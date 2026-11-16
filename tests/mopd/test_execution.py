"""CPU-only contract tests; no torch/Ray, network, PBS, or model downloads."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from d4_mopd import checkpoints, execution
from d4_mopd.metrics import SELECTION_KEY
from d4_mopd.spec import PlannedSampler, load_spec
from d4_teachers.common import digest, files, read, write


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.manifest = dict(identity="frozen-scientific-identity", spec=load_spec(),
                             manifest_path=str(self.root / "campaign/manifest.json"))
        write(self.manifest["manifest_path"], self.manifest)
        self.source = self.root / "checkout"
        for name in (*execution.RUNTIME_PACKAGES, "config/mopd"):
            p = self.source / name / "fixture.py"
            p.parent.mkdir(parents=True)
            p.write_text("immutable runtime fixture\n")
        self.run = self.root / "campaign/runs/domain_mix"
        self.arm = "domain_mix"
        self.sampler = "same-scientific-sampler"

    def prepare(self, name="single_node", route=None):
        record = execution.prepare(self.manifest, name, self.source,
                                   qualification_transition=route or [name, name])
        return execution.attach_execution(self.manifest, record["record_path"])

    def checkpoint(self, step):
        folder = self.run / "checkpoints" / f"global_step_{step}"
        folder.mkdir(parents=True)
        (folder / "data.pt").write_bytes(b"loader cursor and sampler identity")
        (folder / "driver.pt").write_bytes(b"Python NumPy and torch RNG")
        contents = {}
        for name in ("model", "optimizer", "lr_scheduler", "rng_state"):
            relative = name + "/state.bin"
            p = folder / "actor" / relative
            p.parent.mkdir(parents=True)
            p.write_bytes((name + " saved state").encode())
            contents[name] = dict(path=relative)
        write(folder / "actor/ckpt_contents.json",
              dict(framework="megatron", global_step=step, world_size=8, contents=contents))
        checkpoints.commit(folder, self.manifest["identity"], self.arm, step, self.sampler)
        return folder

    def validation(self, step, score):
        reference = checkpoints.evaluation_reference(self.run / "checkpoints",
            self.manifest["identity"], self.arm, step, self.sampler)
        row = dict(status="PASS", step=step, identity=self.manifest["identity"], arm=self.arm,
                   split="validation", test_used_for_selection=False,
                   metrics={SELECTION_KEY: score}, **reference)
        path = self.run / "validation" / f"step_{step}.json"
        write(path, row)
        return dict(step=step, score=score, **reference, validation_receipt=str(path),
                    validation_sha256=digest(row), identity=self.manifest["identity"],
                    arm=self.arm, selection_key=SELECTION_KEY, split="validation",
                    test_used_for_selection=False)

    def receipt(self, manifest):
        route = manifest["execution"]["qualification_transition"]
        return dict(status="PASS", identity=manifest["identity"], arm=self.arm,
                    scope="qualification", step=8, resumed_from=4,
                    **execution.qualification_fields(manifest),
                    qualification_segments=[
                        dict(profile=route[0], attempt_id="fresh-process-A", loaded_step=0, completed_step=4),
                        dict(profile=route[1], attempt_id="fresh-process-B", loaded_step=4, completed_step=8)])

    def test_profiles_preserve_student_layout_and_teacher_pairs(self):
        dual, single = (execution.profile(name) for name in execution.PROFILE_NAMES)
        self.assertEqual((dual["physical_nodes"], single["physical_nodes"]), (2, 1))
        for p in (dual, single):
            self.assertEqual((p["student_world_size"], p["actor_tp"], p["actor_pp"], p["actor_ep"]), (8, 2, 2, 2))
            self.assertEqual(p["teacher_gpu_pairs"], [[0, 1], [2, 3], [4, 5], [6, 7]])
        single["teacher_gpu_pairs"][0].clear()
        self.assertEqual(execution.profile("single_node")["teacher_gpu_pairs"][0], [0, 1])
        with self.assertRaises(RuntimeError):
            execution.profile("eight_rank_student_on_two_nodes")

    def test_snapshot_preserves_manifest_and_ignores_unrelated_checkout_files(self):
        before = Path(self.manifest["manifest_path"]).read_bytes()
        (self.source / "not_runtime.txt").write_text("not part of execution")
        (self.source / "d4_mopd/__pycache__").mkdir()
        (self.source / "d4_mopd/__pycache__/bad.pyc").write_bytes(b"ignored")
        view = self.prepare()
        record = view["execution"]
        self.assertEqual(Path(self.manifest["manifest_path"]).read_bytes(), before)
        self.assertNotIn("execution", self.manifest)
        self.assertEqual(len(record["runtime_files"]), 5)
        self.assertEqual(execution.control_root(self.manifest), self.root / "campaign")
        self.assertEqual(execution.control_root(view), Path(record["control_root"]))
        self.assertEqual(self.prepare()["execution"], record)
        self.assertFalse((self.root / "campaign/launcher_repair.json").exists())

    def test_profile_and_route_ids_are_separate_from_scientific_identity(self):
        single = self.prepare()
        dual = self.prepare("dual_node", ["single_node", "dual_node"])
        cross = self.prepare("single_node", ["dual_node", "single_node"])
        self.assertEqual(len({m["execution"]["execution_identity"] for m in (single, dual, cross)}), 3)
        self.assertEqual(len({m["execution"]["runtime_identity"] for m in (single, dual, cross)}), 1)
        # Use the project's actual domain spelling, without importing a GPU dataset.
        from d4_mopd import DOMAINS
        rows = [dict(domain=d, sample_id=f"{d}-{i}") for d in DOMAINS for i in range(2)]
        samplers = [PlannedSampler(rows, m["spec"], self.arm) for m in (single, dual, cross)]
        self.assertEqual(len({s.identity for s in samplers}), 1)
        self.assertEqual(samplers[0].indices, samplers[1].indices)
        self.assertEqual({m["identity"] for m in (single, dual, cross)}, {self.manifest["identity"]})

    def test_runtime_and_record_tampering_fail_closed(self):
        view = self.prepare()
        record = view["execution"]
        p = Path(record["runtime"]) / "d4_mopd/fixture.py"
        old = p.read_bytes()
        p.write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "runtime changed"):
            execution.verify_record(self.manifest, record["record_path"])
        p.write_bytes(old)
        changed = deepcopy(record)
        changed["qualification_transition"] = ["dual_node", "single_node"]
        write(record["record_path"], changed)
        with self.assertRaisesRegex(RuntimeError, "identity or path"):
            execution.verify_record(self.manifest, record["record_path"])

    def test_route_must_be_explicit_and_end_in_formal_profile(self):
        with self.assertRaises(TypeError):
            execution.prepare(self.manifest, "single_node", self.source)
        with self.assertRaisesRegex(RuntimeError, "formal execution profile"):
            execution.make_record(self.manifest, "single_node", {"a.py": "a" * 64},
                                  qualification_transition=["single_node", "dual_node"])

    def test_full_qualification_rejects_short_recipe_without_modifying_spec(self):
        before = deepcopy(self.manifest["spec"])
        contract = execution.qualification_contract(before)
        self.assertEqual(self.manifest["spec"], before)
        self.assertEqual((contract["updates"], contract["sampler_updates"], contract["pause_after"]), (8, 200, 4))
        self.assertEqual(contract["train_batch_size"], 2048)
        self.assertIsNone(contract["response_length_cap"])
        self.assertFalse(contract["worst_case_64k_capacity_proven"])
        for field, value in (("train_batch_size", 8), ("updates", 8), ("max_context_tokens", 4096)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                execution.qualification_contract(dict(before, **{field: value}))

    def test_qualification_requires_runtime_route_and_actual_fresh_attempt(self):
        view = self.prepare("single_node", ["dual_node", "single_node"])
        receipt = self.receipt(view)
        self.assertEqual(execution.validate_qualification(receipt, view, self.arm), receipt)
        changes = [dict(execution_identity="old"), dict(runtime_identity="different"),
                   dict(qualification_profile_sequence=["single_node", "single_node"]),
                   dict(qualification_contract={}), dict(qualification_segments=[]),
                   dict(resumed_from=0), dict(scope="test")]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                execution.validate_qualification(dict(receipt, **change), view, self.arm)
        old = {k: receipt[k] for k in ("status", "identity", "arm", "scope", "step", "resumed_from")}
        with self.assertRaises(RuntimeError):
            execution.validate_qualification(old, view, self.arm)
        reused = deepcopy(receipt)
        reused["qualification_segments"][1]["attempt_id"] = "fresh-process-A"
        with self.assertRaisesRegex(RuntimeError, "fresh attempt"):
            execution.validate_qualification(reused, view, self.arm)
        forged = deepcopy(receipt)
        forged["qualification_segments"][0]["profile"] = "single_node"
        with self.assertRaises(RuntimeError):
            execution.validate_qualification(forged, view, self.arm)

    def test_matrix_cost_is_explicit_and_never_adds_reverse_direction(self):
        arms = ["domain_mix", "domain_only"]
        one = execution.qualification_plan(self.manifest["spec"], arms, [["dual_node", "single_node"]])
        self.assertEqual((one["case_count"], one["total_optimizer_updates"], one["total_allocated_node_updates"]), (2, 16, 24))
        both = execution.qualification_plan(self.manifest["spec"], arms,
            [["dual_node", "single_node"], ["single_node", "dual_node"]])
        self.assertEqual((both["case_count"], both["total_optimizer_updates"], both["total_allocated_node_updates"]), (4, 32, 48))
        with self.assertRaises(RuntimeError):
            execution.qualification_plan(self.manifest["spec"], arms, [])

    def test_bidirectional_resume_preserves_checkpoint_validation_and_best_bytes(self):
        checkpoint = self.checkpoint(5)
        selection = self.validation(0, .2)
        write(self.run / "best_validation.json", selection)
        before = files(self.run)
        for view in (self.prepare(), self.prepare("dual_node", ["single_node", "dual_node"])):
            evidence = execution.inspect_resume(self.run, view, self.arm, self.sampler)
            self.assertEqual(evidence["step"], 5)
            self.assertEqual(evidence["checkpoint"], str(checkpoint))
            self.assertEqual(evidence["best_selection"], selection)
            self.assertTrue(evidence["best_pointer_current"])
            self.assertFalse(evidence["vllm_sampling_rng_bitwise_restored"])
            self.assertEqual(files(self.run), before)

    def test_corrupt_highest_checkpoint_is_never_silently_rolled_back(self):
        self.checkpoint(5)
        high = self.checkpoint(10)
        self.validation(0, .2)
        (high / "driver.pt").write_bytes(b"changed RNG")
        with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
            execution.inspect_resume(self.run, self.manifest, self.arm, self.sampler)

    def test_resume_rejects_student_world_size_or_sampler_change(self):
        folder = self.checkpoint(5)
        with self.assertRaisesRegex(RuntimeError, "Incompatible"):
            execution.inspect_resume(self.run, self.manifest, self.arm, "wrong sampler")
        native = read(folder / "actor/ckpt_contents.json")
        write(folder / "actor/ckpt_contents.json", dict(native, world_size=16))
        with self.assertRaisesRegex(RuntimeError, "topology mismatch"):
            execution.inspect_resume(self.run, self.manifest, self.arm, self.sampler)

    def test_validation_best_and_completed_test_selection_are_preserved(self):
        self.checkpoint(5)
        self.checkpoint(10)
        best = self.validation(5, .8)
        self.validation(10, .8)  # tie keeps earlier step
        write(self.run / "best_validation.json", best)
        write(self.run / "receipt.json", dict(status="PASS", identity=self.manifest["identity"],
              arm=self.arm, selection=best))
        before = files(self.run)
        result = execution.inspect_resume(self.run, self.manifest, self.arm, self.sampler)
        self.assertEqual(result["best_selection"], best)
        self.assertIsNotNone(result["training_receipt_sha256"])
        self.assertEqual(files(self.run), before)
        changed = read(self.run / "validation/step_5.json")
        changed["checkpoint_identity"] = "wrong checkpoint"
        write(self.run / "validation/step_5.json", changed)
        with self.assertRaisesRegex(RuntimeError, "lineage changed"):
            execution.inspect_resume(self.run, self.manifest, self.arm, self.sampler)

    def test_stale_best_pointer_is_reported_without_mutation(self):
        self.checkpoint(5)
        base = self.validation(0, .1)
        write(self.run / "best_validation.json", base)
        best = self.validation(5, .8)
        before = files(self.run)
        result = execution.inspect_resume(self.run, self.manifest, self.arm, self.sampler)
        self.assertFalse(result["best_pointer_current"])
        self.assertEqual(result["best_selection"], best)
        self.assertEqual(files(self.run), before)

    def test_pruned_validation_history_requires_original_retirement_proof(self):
        old = self.checkpoint(5)
        self.validation(5, .1)
        self.checkpoint(10)
        best = self.validation(10, .8)
        proof = read(old / "mopd_complete.json")
        event = dict(status="DELETED", identity=self.manifest["identity"], arm=self.arm,
                     sampler_identity=self.sampler, step=5, source=str(old), checkpoint_marker=proof)
        write(self.run / "checkpoint_retention/deleted.json", event)
        import shutil
        shutil.rmtree(old)
        result = execution.inspect_resume(self.run, self.manifest, self.arm, self.sampler)
        self.assertEqual(result["best_selection"], best)
        (self.run / "checkpoint_retention/deleted.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "retirement proof"):
            execution.inspect_resume(self.run, self.manifest, self.arm, self.sampler)


if __name__ == "__main__":
    unittest.main()
