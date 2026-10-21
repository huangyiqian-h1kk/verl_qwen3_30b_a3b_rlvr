"""CPU-only isolation and transactional deployment tests; no model/cluster use."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/multidomain"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = load("test_audit_repair", SCRIPTS / "repair_domain_audit_binding.py")
u = repair.u
FIXTURES = Path(__file__).parent / "fixtures/switch_worker"


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.job = Path(self.temp.name) / "job"
        self.phase = self.job / "restored_update"
        self.phase.mkdir(parents=True)
        package = self.job / "runtime/d4_two_node_resumefix"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# Synthetic audit dependency package\n")
        (package / "core.py").write_text("def unused(): pass\n")
        (package / "state_audit.py").write_text(
            "OPTIMIZER_AUDIT_POLICY = 'fused-adam-empty-local-group-step-v1'\n"
            "def state_digest(x, sampled=False): return x\n"
            "def engine_state(engine):\n"
            "    return dict(optimizer_audit_policy=OPTIMIZER_AUDIT_POLICY, optimizer=engine)\n")
        (package / "audit_offload.py").write_text(
            "from .state_audit import engine_state\n"
            "def checkpoint_engine_state(engine): return engine_state(engine)\n")
        binder = self.job / "runtime/d4_db_switch/pinned_audit.py"
        binder.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPTS / "domain_pinned_audit.py", binder)
        u.write(self.job / "request.json", dict(job_dir=str(self.job), expected_nodes=2,
            contract={"packages": {"d4_two_node_resumefix": u.packages(package),
                                    "d4_db_switch": u.packages(binder.parent)}}))
        self.binder = load("synthetic_binder", binder)
        self.torch = types.ModuleType("torch")
        self.torch.distributed = types.SimpleNamespace(is_initialized=lambda: True,
                                                       get_world_size=lambda: 16, get_rank=lambda: 6)
        self.env = patch.dict(os.environ, {self.binder.ENV_JOB: str(self.job),
                                          "DB_SWITCH_PHASE_DIR": str(self.phase)}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.modules = patch.dict(sys.modules, {"torch": self.torch})
        self.modules.start(); self.addCleanup(self.modules.stop)

    def test_poisoned_ordinary_import_cannot_select_old_audit(self):
        old = types.ModuleType("d4_two_node_resumefix.state_audit")
        old.__file__ = "/old/project/state_audit.py"
        old.engine_state = lambda _: {"optimizer": "old"}
        with patch.dict(sys.modules, {old.__name__: old}):
            actual = self.binder.checkpoint_engine_state("moments")
            self.assertEqual(actual["optimizer_audit_policy"], repair.POLICY)
            self.assertEqual(actual["optimizer"], "moments")
            self.assertIs(sys.modules[old.__name__], old)
        record = u.read(self.phase / "audit_binding_rank_6.json")
        self.assertEqual(record["ordinary_module_path"], old.__file__)
        self.assertEqual(len(list(self.phase.glob("audit_state_rank_6_*.json"))), 1)
        self.assertEqual(os.environ["MD2_PHASE_DIR"], str(self.phase))

    def test_source_changes_rejected_even_after_successful_cache(self):
        self.binder.checkpoint_engine_state("moments")
        path = self.job / "runtime/d4_two_node_resumefix/state_audit.py"
        path.write_text(path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            self.binder.checkpoint_engine_state("moments")

    def test_cached_function_binding_change_is_detected(self):
        _, modules = self.binder.modules()
        modules["audit_offload.py"].engine_state = lambda _: {}
        with self.assertRaisesRegex(RuntimeError, "Cached audit binding changed"):
            self.binder.checkpoint_engine_state("moments")

    def test_wrong_phase_or_world_rejected(self):
        os.environ["MD2_PHASE_DIR"] = str(self.job / "first_update")
        with self.assertRaisesRegex(RuntimeError, "phase variables"):
            self.binder.require_audit_binding()
        os.environ.pop("MD2_PHASE_DIR")
        self.torch.distributed.get_world_size = lambda: 8
        with self.assertRaisesRegex(RuntimeError, "sixteen-rank"):
            self.binder.require_audit_binding()

    def test_actual_policy_missing_cannot_pass(self):
        with self.assertRaisesRegex(RuntimeError, "policy is missing"):
            self.binder.compare_restored_state(self.job / "absent.json", {"optimizer": "x"})

    def test_legacy_summary_not_relabeled_and_failure_written_before_raise(self):
        saved = self.job / "saved.json"
        u.write(saved, {"optimizer": "moments"})
        before = saved.read_bytes()
        actual = {"optimizer": "moments", "optimizer_audit_policy": repair.POLICY}
        with self.assertRaisesRegex(RuntimeError, "optimizer_audit_policy"):
            self.binder.compare_restored_state(saved, actual)
        result = u.read(self.phase / "restore_comparison_rank_6.json")
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(saved.read_bytes(), before)

    def test_matching_state_passes_but_moments_and_lr_changes_fail(self):
        saved = self.job / "saved.json"
        good = dict(optimizer_audit_policy=repair.POLICY, optimizer="moments", model="weights", lr_scheduler=168)
        u.write(saved, good)
        self.binder.compare_restored_state(saved, good)
        self.assertEqual(u.read(self.phase / "restore_comparison_rank_6.json")["status"], "PASS")
        for key in ("optimizer", "model", "lr_scheduler"):
            bad = dict(good, **{key: "changed"})
            with self.assertRaisesRegex(RuntimeError, key):
                self.binder.compare_restored_state(saved, bad)
            result = u.read(self.phase / "restore_comparison_rank_6.json")
            self.assertEqual(result["changed_sections"], [key])


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.control = self.root / "outputs/multidomain" / repair.RUN / "handoff"
        self.job = self.control.parent / "jobs/failed_train"
        self.jid = "999001.pbs1"
        self.worker = self.control / "runtime/d4_db_switch/worker.py"
        self.worker.parent.mkdir(parents=True)
        self.worker.write_text(u.patched_worker("domain", (FIXTURES / "domain_worker_before.py").read_text()))
        n2 = self.control / "runtime/d4_two_node_resumefix"
        n2.mkdir()
        (n2 / "state_audit.py").write_text("# Synthetic pinned newer audit\n")
        (n2 / "audit_offload.py").write_text("# Synthetic offload adapter\n")
        for attr, value in (("AUDIT_SHA", u.sha(n2 / "state_audit.py")),
                            ("OFFLOAD_SHA", u.sha(n2 / "audit_offload.py"))):
            p = patch.object(repair, attr, value); p.start(); self.addCleanup(p.stop)
        backend = self.control / "runtime/d4_db_mathfix/verifier_backend.zip"
        backend.parent.mkdir(); backend.write_bytes(b"synthetic verifier archive")
        hashes = {p.name: u.packages(p) for p in (self.control / "runtime").iterdir()}
        contract = dict(packages=hashes, check_identity="common", verifier_backend_sha256=u.sha(backend))
        self.plan = dict(identity=u.digest(contract), contract=contract,
                         info={"identity": "common", "identity_inputs": {"package": hashes["d4_two_node_resumefix"]}})
        self.state = dict(identity=self.plan["identity"], phase="WAIT_READY", handoff_blocked="exit 1",
                          dual={"job_id": self.jid, "folder": str(self.job), "job_name": "0390_d4dbn2_train"},
                          single={"job_id": "999000.pbs1"}, deadline_utc="2099-01-01T00:00:00+00:00")
        self.job.mkdir(parents=True)
        shutil.copytree(self.control / "runtime", self.job / "runtime")
        (self.job / "job_id.txt").write_text(self.jid)
        (self.job / "stage.pbs").write_text("# Synthetic PBS fixture")
        (self.job / "pbs.log").write_text(repair.ERROR)
        u.write(self.job / "request.json", dict(identity=self.plan["identity"], contract=contract,
            snapshot_files={"stage.pbs": u.sha(self.job / "stage.pbs")}))
        u.write(self.job / "result.json", dict(status="FAIL", identity=self.plan["identity"], job_id=self.jid))
        self.cp = self.control.parent / "checkpoints/global_step_21"
        (self.cp / "actor").mkdir(parents=True)
        for rank in range(16):
            u.write(self.cp / "actor" / f"db_dual_state_rank_{rank}.json", {"optimizer": "legacy digest"})
        (self.cp / "weights.bin").write_bytes(b"synthetic checkpoint, must be preserved")
        u.write(self.cp / "domain_switch.json", dict(producer_job_id=self.jid, identity=self.plan["identity"]))
        marker = dict(identity=self.plan["identity"], step=21,
                      files={str(p.relative_to(self.cp)): {"sha256": u.sha(p)} for p in (self.cp / "actor").iterdir()})
        u.write(self.cp / "COMPLETE_0390.json", marker)
        selected = dict(path=str(self.cp), step=21, marker_sha256=u.sha(self.cp / "COMPLETE_0390.json"))
        u.write(self.job / "first_update/qualification.json", dict(status="PASS", identity=self.plan["identity"], checkpoint=selected))
        u.write(self.job / "restored_update/resume_source.json", selected)
        limits_text = "# synthetic limits adapter\n"
        limit_sha = u.text_sha(limits_text)
        p = patch.object(u, "LIMITS_SHA", limit_sha); p.start(); self.addCleanup(p.stop)
        self.cfg = dict(root=str(self.root), kind="domain", contract_identity=self.plan["identity"],
                        guarded_packages=hashes, policy_sha256=limit_sha)
        u.atomic_text(self.control / "runtime_limits_30h_24h/limits_policy.py", limits_text)
        u.write(self.control / "runtime_limits_30h_24h/config.json", self.cfg)
        (self.control / "entry.py").write_text("# Synthetic capacity adapter must remain intact\n")
        for name, obj in (("plan.json", self.plan), ("state.json", self.state)):
            u.write(self.control / name, obj)
        check = self.root / "outputs/multidomain" / (u.BASE + "_n2_v1") / "jobs/check"
        check.mkdir(parents=True)
        (check / "job_id.txt").write_text("998999.pbs1")
        (check / "proof.json").write_text("synthetic PASS evidence")
        accepted = dict(status="PASS", grpo_steps_completed=2, identity="common", job_id="998999.pbs1",
                        evidence={str(check / "proof.json"): u.sha(check / "proof.json")})
        u.write(check / "result.json", accepted); u.write(self.control / "acceptance.json", accepted)
        self.observation = dict(job_state="F", Exit_status=1, queue=u.QUEUE, Job_Name="0390_d4dbn2_train",
                                Submit_arguments=str(self.job / "stage.pbs"))

    def inspect(self):
        return repair.inspect(self.root, self.jid, lambda _: self.observation)

    def test_publish_preserves_checkpoint_deadline_single_and_capacity(self):
        c = self.inspect()
        old_worker = self.worker.read_text()
        capacity = (self.control / "entry.py").read_bytes()
        repair.publish(c)
        fixed = self.inspect()
        self.assertTrue(fixed["fixed"])
        self.assertEqual(fixed["state"]["single"], self.state["single"])
        self.assertEqual(fixed["state"]["deadline_utc"], self.state["deadline_utc"])
        self.assertEqual(fixed["state"]["phase"], "WAIT_CHECKPOINT")
        self.assertEqual((self.control / "entry.py").read_bytes(), capacity)
        archived = self.control / repair.AREA / "old_checkpoints/global_step_21/weights.bin"
        self.assertEqual(archived.read_bytes(), b"synthetic checkpoint, must be preserved")
        self.assertFalse(list((self.control.parent / "checkpoints").iterdir()))
        self.assertEqual((self.job / "runtime/d4_db_switch/worker.py").read_text(), old_worker)
        state = u.read(self.control / "state.json")
        state.update(phase="WAIT_READY", dual={"job_id": "new.pbs1"})
        u.write(self.control / "state.json", state)
        self.assertEqual(self.inspect()["state"]["dual"]["job_id"], "new.pbs1")

    def test_partial_write_failure_restores_checkpoint_and_contract(self):
        c = self.inspect(); original = u.atomic_text; fired = False
        def fail_once(path, text):
            nonlocal fired
            if Path(path) == self.control / "state.json" and not fired:
                fired = True
                raise OSError("simulated publication failure")
            return original(path, text)
        with patch.object(u, "atomic_text", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "publication"):
                repair.publish(c)
        self.assertEqual(u.read(self.control / "plan.json"), self.plan)
        self.assertTrue((self.cp / "weights.bin").is_file())
        self.assertFalse((self.control / "runtime/d4_db_switch/pinned_audit.py").exists())
        self.assertFalse(self.inspect()["fixed"])

    def test_interrupted_rename_recovers_without_deleting_checkpoint(self):
        c = self.inspect()
        files = {n: dict(before=(self.control / n).read_text() if (self.control / n).exists() else None, after=s)
                 for n,s in repair.changes(c).items()}
        u.write(self.control / repair.AREA / "transaction.json",
                dict(version=repair.VERSION, status="PREPARED", files=files))
        self.cp.parent.rename(self.control / repair.AREA / "old_checkpoints")
        repair.recover(self.control)
        self.assertTrue((self.cp / "weights.bin").is_file())
        self.assertEqual(u.read(self.control / "state.json"), self.state)

    def test_other_error_live_job_wrong_queue_and_handoff_block_rearm(self):
        original = dict(self.observation)
        for change in ({"job_state":"R"},{"Exit_status":271},{"queue":"rt_HF"}):
            self.observation = {**original, **change}
            with self.assertRaisesRegex(RuntimeError, "PBS"):
                self.inspect()
        self.observation = original
        (self.job / "pbs.log").write_text("another application failure")
        with self.assertRaisesRegex(RuntimeError, "Different candidate error"):
            self.inspect()
        (self.job / "pbs.log").write_text(repair.ERROR)
        u.write(self.control / "qdel.json", {"existing":True})
        with self.assertRaisesRegex(RuntimeError, "handoff"):
            self.inspect()

    def test_no_unrelated_checkpoint_or_policy_override(self):
        other = self.cp.parent / "global_step_22"; other.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Other candidate"):
            self.inspect()
        other.rmdir()
        u.write(self.job / "restored_update/optimizer_audit_rank_6_x.json", {})
        with self.assertRaisesRegex(RuntimeError, "Detailed audits"):
            self.inspect()

    def test_unknown_worker_cannot_be_rewritten(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown domain worker"):
            repair.patched_worker(self.worker.read_text() + "\n# changed\n")


if __name__ == "__main__":
    unittest.main()
