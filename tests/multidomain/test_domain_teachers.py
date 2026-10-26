"""CPU integration/fault tests; no synthetic test is claimed as GPU acceptance."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

from test_training_continuation import checkpoint
import test_training_continuation_validation as validation_fixtures
from d4_teachers import DOMAINS
from d4_teachers import (
    config,
    controller,
    data,
    evaluation,
    sampling,
    selection,
    source,
)
from d4_teachers import exporting, trainer as teacher_trainer
from d4_teachers.common import (
    digest,
    file_sha,
    files,
    model_inventory,
    read,
    write,
    is_fenced,
)
from d4_training_continuation import checkpoints as ck
from d4_training_continuation.validation_journal import instrument
from d4_training_continuation.scheduler import Unknown


class TestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class DataAndSampling(TestCase):
    def rows(self):
        return {
            s: [
                dict(
                    sample_id=f"{s}_{d}_{i}",
                    split_group_id=f"{s}_{d}_{i}",
                    domain=d,
                    source="original",
                    prompt_tokens=10,
                    generation_max_tokens=65526,
                    prompt_token_hash="frozen",
                    messages_json='[{"role":"user","content":"x"}]',
                    tools_json="[]",
                    verifier_json='{"answer":42}',
                    category="c",
                    task="t",
                )
                for d in DOMAINS
                for i in range(n)
            ]
            for s, n in (("train", 6), ("validation", 3), ("test", 2))
        }

    def test_direct_arrow_slices_preserve_every_field_and_parent_is_immutable(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        parent = self.root / "parent"
        parent.mkdir()
        rows = self.rows()
        for split, values in rows.items():
            pq.write_table(pa.Table.from_pylist(values), parent / (split + ".parquet"))
        before = {s: file_sha(parent / (s + ".parquet")) for s in rows}
        m = data.prepare(parent, self.root / "slices", 6, {d: 3 for d in DOMAINS})
        for d in DOMAINS:
            for split, values in rows.items():
                self.assertEqual(
                    pq.read_table(
                        self.root / "slices" / d / (split + ".parquet")
                    ).to_pylist(),
                    [r for r in values if r["domain"] == d],
                )
        self.assertEqual(m["parent_sha256"], before)
        self.assertEqual(
            data.prepare(parent, self.root / "slices", 6, {d: 3 for d in DOMAINS}), m
        )
        path = self.root / "slices/math/train.parquet"
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            data.verify(self.root / "slices")

    def test_leakage_and_counts_fail_closed(self):
        rows = self.rows()
        rows["test"][0]["split_group_id"] = rows["train"][0]["split_group_id"]
        with self.assertRaisesRegex(RuntimeError, "leakage"):
            data.validate_rows(rows, 6, {d: 3 for d in DOMAINS})
        rows = self.rows()
        rows["train"].pop()
        with self.assertRaisesRegex(RuntimeError, "counts"):
            data.validate_rows(rows, 6, {d: 3 for d in DOMAINS})

    def test_200_full_batches_and_no_duplicates_within_batch(self):
        rows = [dict(sample_id=str(i), domain="math") for i in range(10000)]
        p = sampling.build_plan(rows, "math")
        self.assertEqual(len(p.indices), 51200)
        self.assertEqual(p, sampling.build_plan(rows, "math"))
        for step in range(1, 201):
            self.assertEqual(len(set(p.batch(step))), 256)
        self.assertEqual(len(set(p.indices[: 39 * 256])), 39 * 256)
        self.assertNotEqual(p, sampling.build_plan(rows, "math", seed=43))
        self.assertEqual(sampling.loader_restart_check()["status"], "PASS")

    def test_actual_batch_must_match_ids_domain_and_eight_responses(self):
        batch = types.SimpleNamespace(
            non_tensor_batch=dict(
                data_source=["math"] * 16,
                extra_info=[dict(sample_id=x) for x in ["a", "b"] * 8],
            )
        )
        sampling.verify_batch(batch, "math", ["a", "b"], 8)
        batch.non_tensor_batch["extra_info"][0]["sample_id"] = "foreign"
        with self.assertRaisesRegex(RuntimeError, "differs"):
            sampling.verify_batch(batch, "math", ["a", "b"], 8)

    def test_policy_is_fixed_and_validation_grid_controls_patience(self):
        p = config.load()
        self.assertFalse(selection.should_stop(40, 0, p))
        self.assertFalse(selection.should_stop(60, 40, p))
        self.assertTrue(selection.should_stop(70, 40, p))
        self.assertTrue(selection.should_stop(50, 0, p))
        p["validate_every"] = 15
        write(self.root / "bad.json", p)
        with self.assertRaisesRegex(RuntimeError, "fixed"):
            config.load(self.root / "bad.json")


class EvaluationRecovery(TestCase):
    def validator(self, n, fail=None):
        obj = validation_fixtures.JournalTests.validator(self, fail)
        obj.val_dataloader = obj.val_dataloader[: (n + 7) // 8]
        for i, batch in enumerate(obj.val_dataloader):
            take = min(8, n - i * 8)
            for key in list(batch):
                batch[key] = batch[key][:take]
            batch["uid"] = np.array([str(i * 8 + j) for j in range(take)], dtype=object)
            batch["data_source"] = np.array(["math"] * take, dtype=object)
        obj._val_metrics_update = lambda domains, uids, extra, turns: {
            "val-core/math/acc/mean@1": sum(extra["acc"]) / len(uids)
        }
        return obj

    def attach(self, obj, n=238, weights="step10"):
        obj.cont_journal = evaluation.Journal(
            self.root / "journal",
            dict(weights=weights, split="validation"),
            dict(rows=n, batches=(n + 7) // 8, ids=[str(i) for i in range(n)]),
        )
        return instrument(obj._validate)

    def test_math_238_rows_resume_completed_batches_before_generation(self):
        first = self.validator(238, fail=8)
        with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "walltime"):
            self.attach(first)()
        self.assertEqual(read(self.root / "journal/progress.json")["completed_rows"], 8)
        obj = self.validator(238)
        with patch("builtins.print"):
            metrics = self.attach(obj)()
        proof, samples = obj.cont_journal.finish()
        self.assertEqual(
            (proof["rows"], proof["batches"], proof["reused_batches"]), (238, 30, 1)
        )
        self.assertEqual(obj.generated[0], 8)
        self.assertEqual(len(samples), 238)
        self.assertEqual(metrics, {"val-core/math/acc/mean@1": 1.0})
        again = self.validator(238)
        with patch("builtins.print"):
            self.attach(again)()
        self.assertFalse(again.generated)
        self.assertEqual(again.cont_journal.finish()[0]["reused_batches"], 30)

    def test_other_domain_256_and_changed_identity_or_order_rejected(self):
        obj = self.validator(256)
        with patch("builtins.print"):
            self.attach(obj, 256)()
        self.assertEqual(obj.cont_journal.finish()[0]["rows"], 256)
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            self.attach(self.validator(256), 256, "other-step")
        changed = self.validator(256)
        changed.val_dataloader[0]["uid"][0] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "reordered"):
            self.attach(changed, 256)()

    def test_failure_after_batch_write_before_progress_does_not_regenerate(self):
        obj = self.validator(238)
        method = self.attach(obj)
        with (
            patch.object(
                evaluation.Journal,
                "progress",
                side_effect=OSError("injected fsync crash"),
            ),
            patch("builtins.print"),
            self.assertRaises(OSError),
        ):
            method()
        resumed = self.validator(238)
        with patch("builtins.print"):
            self.attach(resumed)()
        self.assertEqual(resumed.generated[0], 8)


class SelectionAndRetention(TestCase):
    def request(self):
        return dict(
            role="train",
            run_dir=str(self.root),
            domain="math",
            lineage="lineage",
            base_model="/base",
            base_identity="basehash",
            verifier_identity="verifier",
        )

    def receipt(self, step, score, split="validation"):
        return dict(
            split=split,
            step=step,
            scores={"math": {"mean": score}},
            path="receipt",
            identity="v" + str(step),
        )

    def cp(self, step):
        return checkpoint(
            self.root / f"checkpoints/global_step_{step}", step, lineage="lineage"
        )

    def test_weight_pin_survives_full_checkpoint_pruning_ties_keep_earlier(self):
        req = self.request()
        base = dict(step=0)
        selection.select(req, base, self.receipt(0, 0.1))
        cp = self.cp(10)
        best = selection.select(req, cp, self.receipt(10, 0.5))
        pinned = Path(best["path"]) / "actor/dist/payload.distcp"
        original = Path(cp["path"]) / "actor/dist/payload.distcp"
        self.assertEqual(pinned.stat().st_ino, original.stat().st_ino)
        self.assertFalse((Path(best["path"]) / "data.pt").exists())
        cp20 = self.cp(20)
        self.cp(25)
        ck.retain(self.root, "lineage", 2)
        self.assertFalse(original.exists())
        self.assertTrue(pinned.exists())
        self.assertEqual(selection.select(req, cp20, self.receipt(20, 0.5))["step"], 10)
        better = selection.select(req, cp20, self.receipt(20, 0.6))
        self.assertEqual(better["step"], 20)
        self.assertFalse(Path(best["path"]).exists())
        with self.assertRaisesRegex(RuntimeError, "Only validation"):
            selection.select(req, cp20, self.receipt(20, 1, "test"))

    def test_interrupted_best_pointer_publication_is_recoverable(self):
        req = self.request()
        cp = self.cp(10)
        real = selection.write

        def fail(path, value):
            if Path(path).name == "best.json":
                raise OSError("pointer crash")
            return real(path, value)

        with (
            patch.object(selection, "write", side_effect=fail),
            self.assertRaises(OSError),
        ):
            selection.select(req, cp, self.receipt(10, 0.5))
        self.assertEqual(selection.select(req, cp, self.receipt(10, 0.5))["step"], 10)


class FakePBS:
    def __init__(self):
        self.rows = {}
        self.submissions = []
        self.cancellations = []
        self.unknown = False
        self.ambiguous = False
        self.after_submit = None

    def observe(self, job):
        if self.unknown:
            raise Unknown("injected qstat uncertainty")
        return self.rows[job["job_id"]]

    def recover(self, folder, name):
        path = Path(folder)
        if (path / "job_id.txt").exists():
            return dict(
                folder=str(path),
                job_name=name,
                job_id=(path / "job_id.txt").read_text(),
            )
        if (path / "qsub_attempted.json").exists():
            raise Unknown("ambiguous qsub")

    def submit(self, folder, name):
        job = self.recover(folder, name)
        if job:
            return job
        write(Path(folder) / "qsub_attempted.json", {})
        self.submissions.append(str(folder))
        if self.ambiguous:
            raise Unknown("lost qsub stdout")
        jid = str(len(self.submissions))
        (Path(folder) / "job_id.txt").write_text(jid)
        job = dict(folder=str(folder), job_name=name, job_id=jid)
        self.rows[jid] = dict(job_state="Q")
        if self.after_submit:
            self.after_submit()
        return job

    def cancel(self, job, journal):
        if self.rows[job["job_id"]]["job_state"] == "F":
            return True
        self.cancellations.append(job["job_id"])
        return False


class GuardianFaults(TestCase):
    def setUp(self):
        super().setUp()
        self.c = dict(path=str(self.root), policy=config.load())
        self.pbs = FakePBS()
        self.jobs = 0
        for d in controller.TASKS:
            write(self.root / "runs" / d / "run.json", dict(lineage=d))
            write(
                controller.state_path(self.c, d),
                dict(
                    status="READY",
                    phase="test" if d == "base" else "train",
                    job=None,
                    pending=None,
                    last_progress=0,
                    no_progress_retries=0,
                ),
            )
        self.maker = patch.object(controller, "make_job", side_effect=self.make)
        self.maker.start()
        self.addCleanup(self.maker.stop)
        self.manifest = patch.object(controller, "manifest")
        self.manifest.start()
        self.addCleanup(self.manifest.stop)

    def make(self, c, d, role):
        self.jobs += 1
        folder = self.root / "runs" / d / "jobs" / str(self.jobs)
        folder.mkdir(parents=True)
        return dict(folder=str(folder), job_name="0390_test")

    def tick(self):
        return controller.tick(self.c, self.pbs)

    def test_four_independent_jobs_and_campaign_cap_without_account_queries(self):
        self.c["policy"]["max_concurrent_jobs"] = 2
        s = self.tick()
        self.assertEqual(len(self.pbs.submissions), 2)
        self.tick()
        self.assertEqual(len(self.pbs.submissions), 2)
        self.assertEqual(s["math"]["status"], "QUEUED")
        self.assertEqual(s["logic_algorithmic"]["status"], "READY")

    def test_held_and_unknown_remain_owned_never_duplicate(self):
        s = self.tick()
        job = s["math"]["job"]
        self.pbs.rows[job["job_id"]] = {"job_state": "H"}
        self.assertEqual(self.tick()["math"]["status"], "BLOCKED_HELD")
        self.pbs.unknown = True
        self.assertEqual(self.tick()["math"]["status"], "BLOCKED_UNKNOWN")
        self.tick()
        self.assertEqual(len(self.pbs.submissions), 4)

    def test_ambiguous_submit_intent_survives_restart(self):
        self.pbs.ambiguous = True
        s = self.tick()
        self.assertEqual(s["math"]["status"], "BLOCKED_UNKNOWN")
        self.pbs.ambiguous = False
        self.tick()
        self.assertEqual(len(self.pbs.submissions), 4)
        self.assertIsNotNone(read(controller.state_path(self.c, "math"))["pending"])

    def test_only_observed_admin_or_walltime_retry_and_no_progress_is_bounded(self):
        s = self.tick()
        math = s["math"]["job"]
        other = s["if"]["job"]
        self.pbs.rows[other["job_id"]] = {"job_state": "F", "Exit_status": "1"}
        self.pbs.rows[math["job_id"]] = {
            "job_state": "F",
            "Exit_status": "-29",
            "comment": "walltime exceeded",
        }
        s = self.tick()
        self.assertEqual(s["if"]["status"], "BLOCKED_APPLICATION")
        self.assertEqual(s["math"]["status"], "QUEUED")
        for i in range(2):
            self.pbs.rows[s["math"]["job"]["job_id"]] = {
                "job_state": "F",
                "Exit_status": "271",
            }
            s = self.tick()
        self.assertEqual(s["math"]["status"], "BLOCKED_RETRIES")

    def test_stop_during_qsub_cancels_exact_job_and_waits_for_terminal(self):
        self.pbs.after_submit = lambda: write(self.root / "STOP.json", {})
        s = self.tick()
        self.assertEqual(len(self.pbs.submissions), 1)
        self.assertEqual(s["math"]["status"], "STOPPING")
        self.assertEqual(self.pbs.cancellations, ["1"])
        self.assertTrue(is_fenced(self.root))
        self.pbs.rows["1"] = {"job_state": "F"}
        self.assertEqual(self.tick()["math"]["status"], "STOPPED")
        self.tick()
        self.assertEqual(len(self.pbs.submissions), 1)

    def test_stop_before_intent_submit_never_submits(self):
        pending = self.make(self.c, "math", "train")
        state = read(controller.state_path(self.c, "math"))
        state["pending"] = pending
        write(controller.state_path(self.c, "math"), state)
        write(self.root / "STOP.json", {})
        self.tick()
        self.assertFalse(self.pbs.submissions)

    def test_one_submission_failure_does_not_block_other_teachers(self):
        original = self.make

        def make(c, d, role):
            if d == "math":
                raise RuntimeError("broken snapshot")
            return original(c, d, role)

        with patch.object(controller, "make_job", side_effect=make):
            s = self.tick()
        self.assertEqual(s["math"]["status"], "BLOCKED_ARTIFACT")
        self.assertEqual(len(self.pbs.submissions), 3)

    def test_terminal_application_failure_releases_campaign_slot(self):
        self.c["policy"]["max_concurrent_jobs"] = 1
        s = self.tick()
        self.pbs.rows[s["math"]["job"]["job_id"]] = {
            "job_state": "F",
            "Exit_status": "1",
        }
        s = self.tick()
        self.assertEqual(s["math"]["status"], "BLOCKED_APPLICATION")
        self.assertEqual(s["if"]["status"], "QUEUED")
        self.assertEqual(len(self.pbs.submissions), 2)

    def test_explicit_reenable_does_not_clear_an_unobserved_running_job(self):
        s = self.tick()
        write(self.root / "runs/math/STOP.json", {})
        with patch.object(controller, "load_campaign", return_value=self.c):
            with self.assertRaisesRegex(RuntimeError, "terminal"):
                controller.reenable(self.root, "math")
        self.assertTrue(is_fenced(self.root, "math"))
        self.pbs.rows[s["math"]["job"]["job_id"]] = {"job_state": "F"}
        self.tick()
        with patch.object(controller, "load_campaign", return_value=self.c):
            controller.reenable(self.root, "math")
        self.assertFalse(is_fenced(self.root, "math"))
        self.assertEqual(read(controller.state_path(self.c, "math"))["status"], "READY")

    def test_success_chains_export_then_test_but_test_never_selects(self):
        s = self.tick()
        job = s["math"]["job"]
        write(Path(job["folder"]) / "result.json", {"status": "PASS"})
        self.pbs.rows[job["job_id"]] = {"job_state": "F", "Exit_status": "0"}
        with patch.object(controller, "artifact") as verify:
            s = self.tick()
        verify.assert_called_once_with(self.c, "math", "train")
        self.assertEqual(s["math"]["phase"], "export")
        job = s["math"]["job"]
        write(Path(job["folder"]) / "result.json", {"status": "PASS"})
        self.pbs.rows[job["job_id"]] = {"job_state": "F", "Exit_status": "0"}
        with patch.object(controller, "artifact"):
            s = self.tick()
        self.assertEqual(s["math"]["phase"], "test")


class FrozenDeployment(TestCase):
    def fixture(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from multidomain.common import read_config

        parent = self.root / "parent"
        parent.mkdir()
        job = parent / "old_job"
        job.mkdir()
        cfg = read_config(source.REPO / "config/multidomain/four_domain.yaml")
        cfg["verl_src"] = str(parent / "verl")
        cfg["training"]["model_path"] = str(parent / "base")
        manager = parent / "verl/verl/utils/checkpoint/megatron_checkpoint_manager.py"
        manager.parent.mkdir(parents=True)
        manager.write_text("# fixture manager\n")
        write(parent / "base/config.json", {"model_type": "fixture"})
        (parent / "base/weights.safetensors").write_bytes(b"fixture weights")
        (parent / ".venv-multidomain").symlink_to(sys.prefix, target_is_directory=True)
        compat = parent / "compat/sitecustomize.py"
        compat.parent.mkdir()
        compat.write_text("# frozen compatibility fixture\n")
        roots = {}
        for name in (
            "scripts/multidomain/node.sh",
            "multidomain/__init__.py",
            "config/multidomain/agent.yaml",
        ):
            target = parent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.REPO / name, target)
            roots[name] = file_sha(target)
        for name in ("__init__.py", "core.py", "state_audit.py", "audit_offload.py"):
            target = job / "audit_source" / name
            target.parent.mkdir(exist_ok=True)
            target.write_text("# audit fixture\n")
        backend = job / "runtime/d4_db_mathfix/verifier_backend.zip"
        backend.parent.mkdir(parents=True)
        backend.write_bytes(b"fixture archive")
        reward = job / "md_reward_roundoff_0390/__init__.py"
        reward.parent.mkdir()
        reward.write_text("DIAGNOSTICS = " + repr(str(job / "math_diagnostics")) + "\n")
        rows = DataAndSampling.rows(self)
        dataset = parent / "data"
        dataset.mkdir()
        for split, values in rows.items():
            pq.write_table(pa.Table.from_pylist(values), dataset / (split + ".parquet"))
        write(
            dataset / "manifest.json",
            dict(
                config_sha256=digest(cfg),
                files={
                    s + ".parquet": file_sha(dataset / (s + ".parquet")) for s in rows
                },
            ),
        )
        contract = dict(base_config=cfg, root_files=roots, snapshot_files=files(job))
        request = dict(
            contract=contract,
            identity=digest(contract),
            job_dir=str(job),
            data_dir=str(dataset),
            config=cfg,
        )
        write(job / "request.json", request)
        policy = config.load()
        policy.update(
            parent_root=str(parent),
            parent_data="data",
            source_request="old_job/request.json",
        )
        actual_prepare = data.prepare

        def smaller(p, d):
            return actual_prepare(p, d, 6, {x: 3 for x in DOMAINS})

        with (
            patch.object(
                source, "AUDIT_SHA", file_sha(job / "audit_source/state_audit.py")
            ),
            patch.object(source, "MANAGER_SHA", file_sha(manager)),
            patch.object(data, "prepare", side_effect=smaller),
        ):
            c = source.prepare(self.root / "campaign", policy)
        return c

    def test_prepare_snapshot_job_and_real_isolated_launcher(self):
        c = self.fixture()
        job = source.make_job(c, "math", "train")
        folder = Path(job["folder"])
        r = source.validate_request(folder / "request.json")
        self.assertEqual(r["stage"], "train")
        self.assertEqual(r["expected_nodes"], 1)
        self.assertEqual(r["config"]["training"]["total_training_steps"], 200)
        self.assertEqual(r["config"]["resolved"]["enabled_domains"], ["math"])
        self.assertIn("#PBS -l select=1", (folder / "stage.pbs").read_text())
        self.assertIn("#PBS -l walltime=24:00:00", (folder / "stage.pbs").read_text())
        self.assertNotIn("-m multidomain.stage", (folder / "node.sh").read_text())
        subprocess.run(["bash", "-n", str(folder / "node.sh")], check=True)
        subprocess.run(["bash", "-n", str(folder / "stage.pbs")], check=True)
        result = subprocess.run(
            [sys.executable, "-I", str(folder / "entry.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cpu-checks", result.stdout)
        subprocess.run(
            [sys.executable, "-I", str(Path(c["path"]) / "control.py"), "--help"],
            check=True,
            capture_output=True,
        )
        source.validate_request(folder / "request.json")
        (folder / "runtime/d4_teachers/sampling.py").write_text("# changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            source.validate_request(folder / "request.json")

    def test_fence_prevents_new_job_and_config_mutation_rejected(self):
        c = self.fixture()
        write(Path(c["path"]) / "runs/math/STOP.json", {})
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            source.make_job(c, "math", "train")
        c["contract"]["policy"]["validate_every"] = 15
        write(Path(c["path"]) / "campaign.json", c)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            source.load_campaign(c["path"])


class TrainerGuards(TestCase):
    def test_export_rename_before_pointer_crash_recovers_proof_and_retires_native(self):
        cp = checkpoint(self.root / "checkpoints/global_step_10", 10, lineage="l")
        req = dict(
            role="train",
            run_dir=str(self.root),
            domain="math",
            lineage="l",
            verifier_identity="v",
        )
        receipt = dict(
            split="validation",
            step=10,
            scores={"math": {"mean": 0.5}},
            path="receipt",
            identity="v10",
        )
        best = selection.select(req, cp, receipt)
        candidate = self.root / "candidate"
        write(candidate / "config.json", {"model_type": "fixture"})
        (candidate / "weights.safetensors").write_bytes(b"weights")
        proof = dict(
            status="PASS",
            all_weights_loaded=True,
            finite_forward=True,
            selected_identity=best["weights_identity"],
        )
        original = exporting.write

        def fail(path, value):
            if Path(path).name == "export.json":
                raise OSError("publication crash")
            return original(path, value)

        with (
            patch.object(exporting, "write", side_effect=fail),
            self.assertRaises(OSError),
        ):
            exporting.publish_export(req, best, candidate, proof)
        self.assertFalse(candidate.exists())
        self.assertTrue(Path(best["path"]).exists())
        exported = exporting.publish_export(req, best, candidate, proof)
        self.assertEqual(exported["load_proof"], proof)
        self.assertFalse(Path(best["path"]).exists())
        self.assertEqual(selection.validate_best(best), best)

    def test_hf_inventory_ignores_download_locks_but_binds_model_assets(self):
        write(self.root / "model/config.json", {"model_type": "fixture"})
        (self.root / "model/weights.safetensors").write_bytes(b"weights")
        inv = model_inventory(self.root / "model")
        cache = self.root / "model/.cache/huggingface/download"
        cache.mkdir(parents=True)
        (cache / "weights.lock").touch()
        write(cache / "metadata.json", {"transient": True})
        self.assertEqual(inv, model_inventory(self.root / "model"))
        (self.root / "model/weights.safetensors").write_bytes(b"changed weights")
        self.assertNotEqual(inv, model_inventory(self.root / "model"))

    def test_early_stop_exits_before_next_generation_and_test_cannot_update(self):
        class Base:
            def _get_gen_batch(self, batch):
                raise AssertionError("generated beyond stop")

            def _update_actor(self, batch):
                raise AssertionError("test updated actor")

        t = teacher_trainer.trainer_class(Base)()
        t.teacher_request = dict(
            role="train", campaign_dir=str(self.root), domain="math"
        )
        t.teacher_validating = False
        t.teacher_stop = True
        with self.assertRaises(teacher_trainer.TeacherComplete):
            t._get_gen_batch(None)
        t.teacher_request["role"] = "test"
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            t._update_actor(None)

    def test_transactional_save_requires_all_rank_audits_and_is_idempotent(self):
        class Base:
            def _save_checkpoint(obj):
                cp = checkpoint(
                    Path(obj.config.trainer.default_local_dir) / "global_step_5", 5
                )
                if obj.break_rank:
                    (Path(cp["path"]) / "actor/continuation_state_rank_7.json").unlink()

        t = teacher_trainer.trainer_class(Base)()
        t.global_steps = 5
        t.teacher_saved = None
        t.break_rank = True
        t.config = types.SimpleNamespace(
            trainer=types.SimpleNamespace(
                default_local_dir=str(self.root / "checkpoints")
            )
        )
        t.teacher_request = dict(
            role="train",
            run_dir=str(self.root),
            domain="math",
            lineage="l",
            campaign_identity="c",
        )
        t.teacher_resume = dict(step=0, files={})
        t.teacher_plan = types.SimpleNamespace(
            metadata=lambda: dict(plan_sha256="plan")
        )
        with self.assertRaises(FileNotFoundError):
            t._save_checkpoint()
        self.assertIsNone(ck.latest(self.root, "l"))
        t.break_rank = False
        t._save_checkpoint()
        t._save_checkpoint()
        cp = ck.latest(self.root, "l")
        self.assertEqual(cp["step"], 5)
        self.assertEqual(
            read(Path(cp["path"]) / "teacher.json")["campaign_identity"], "c"
        )

    def test_retiring_native_best_requires_published_load_proof(self):
        cp = checkpoint(self.root / "checkpoints/global_step_10", 10, lineage="l")
        req = dict(
            role="train",
            run_dir=str(self.root),
            domain="math",
            lineage="l",
            verifier_identity="v",
        )
        receipt = dict(
            split="validation",
            step=10,
            scores={"math": {"mean": 0.5}},
            path="receipt",
            identity="v10",
        )
        best = selection.select(req, cp, receipt)
        hf = self.root / "artifacts/hf_step_10"
        write(hf / "config.json", {"model_type": "fixture"})
        (hf / "weights.safetensors").write_bytes(b"weights")
        inv = model_inventory(hf)
        exp = dict(
            status="PASS",
            path=str(hf),
            files=inv,
            weights_identity=digest(inv),
            selected=best,
            load_proof=dict(status="FAIL"),
        )
        with self.assertRaisesRegex(RuntimeError, "proof"):
            exporting.retire_native_best(self.root, exp)
        self.assertTrue(Path(best["path"]).exists())
        exp["load_proof"].update(
            status="PASS",
            all_weights_loaded=True,
            finite_forward=True,
            selected_identity=best["weights_identity"],
        )
        write(self.root / "export.json", exp)
        exporting.retire_native_best(self.root, exp)
        self.assertFalse(Path(best["path"]).exists())
        self.assertEqual(selection.validate_best(best), best)


if __name__ == "__main__":
    unittest.main()
