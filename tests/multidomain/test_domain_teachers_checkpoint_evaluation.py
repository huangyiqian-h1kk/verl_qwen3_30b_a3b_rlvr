"""Real snapshot/lease tests with synthetic weights and a fake PBS scheduler."""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_domain_teachers as fixtures
from test_training_continuation import checkpoint
from d4_teachers import (
    checkpoint_evaluation as extra,
    controller,
    exporting,
    source,
    runtime,
)
from d4_teachers.common import files, lock, read, write
from d4_training_continuation import checkpoints as ck


class CheckpointEvaluation(fixtures.TestCase):
    def setUp(self):
        super().setUp()
        self.c = fixtures.FrozenDeployment.fixture(self)
        self.campaign = Path(self.c["path"])
        self.pbs = fixtures.FakePBS()
        for domain in ("math", "if", "conversational_pivot"):
            self.change(domain, status="DONE", phase="test")
            write(self.campaign / "runs" / domain / "best.json", {"original": domain})
            write(self.campaign / "runs" / domain / "test.json", {"original": domain})
        for domain, step in extra.TARGETS.values():
            run = self.campaign / "runs" / domain
            lineage = read(run / "run.json")["lineage"]
            checkpoint(
                run / "checkpoints" / ("global_step_" + str(step)),
                step,
                lineage=lineage,
            )
            write(
                run / "training_complete.json",
                dict(status="PASS", completed_step=step, lineage=lineage),
            )
            (run / "writer.lock").touch()
        self.original = {
            d: files(self.campaign / "runs" / d)
            for d in ("math", "conversational_pivot")
        }

    def change(self, task, **values):
        path = controller.state_path(self.c, task)
        state = read(path)
        state.update(values)
        write(path, state)

    def tick(self):
        return controller.tick(self.c, self.pbs)

    def export(self, task):
        run = self.campaign / "runs" / task
        selected = read(run / "best.json")
        candidate = run / "candidate"
        write(candidate / "config.json", {"model_type": "fixture"})
        (candidate / "weights.safetensors").write_bytes(b"synthetic exported weights")
        request = dict(
            run_dir=str(run),
            verifier_identity=self.c["contract"]["verifier_identity"],
            lineage=selected["lineage"],
        )
        proof = dict(
            status="PASS",
            all_weights_loaded=True,
            finite_forward=True,
            selected_identity=selected["weights_identity"],
        )
        return exporting.publish_export(request, selected, candidate, proof)

    def test_registration_pins_exact_final_models_and_preserves_originals(self):
        result = extra.install(self.c)
        self.assertEqual(result["status"], "REGISTERED")
        for task, (domain, step) in extra.TARGETS.items():
            selected = extra.validate_selected(self.c, task)
            self.assertEqual(selected["step"], step)
            self.assertEqual(selected["selection_source"], "explicit_checkpoint")
            self.assertNotIn("validation_receipt", selected)
            native = Path(selected["source_checkpoint"]["path"])
            self.assertFalse((Path(selected["path"]) / "data.pt").exists())
            self.assertEqual(
                os.stat(native / "actor/dist/payload.distcp").st_ino,
                os.stat(Path(selected["path"]) / "actor/dist/payload.distcp").st_ino,
            )
            self.assertEqual(
                files(self.campaign / "runs" / domain), self.original[domain]
            )
        self.assertEqual(extra.install(self.c)["status"], "ALREADY_REGISTERED")

    def test_missing_checkpoint_fails_before_handoff_or_any_task_publication(self):
        (self.campaign / "runs/math/checkpoints/global_step_50" / ck.MARKER).unlink()
        with patch("d4_teachers.retry_repair.repair") as repair:
            with self.assertRaises(FileNotFoundError):
                extra.launch(self.campaign)
        repair.assert_not_called()
        self.assertFalse((self.campaign / extra.PLAN).exists())
        self.assertFalse((self.campaign / "runs/cp_step80").exists())

    def test_source_writer_or_explicit_stop_prevents_install(self):
        with lock(self.campaign / "runs/math/writer.lock"):
            with self.assertRaises(BlockingIOError):
                extra.install(self.c)
        write(self.campaign / "runs/math/STOP.json", {})
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            extra.install(self.c)
        self.assertFalse((self.campaign / extra.PLAN).exists())

    def test_mid_install_crash_repeats_without_partial_submission(self):
        actual = extra.pin_model

        def fail_second(run, target):
            if target["domain"] == "conversational_pivot":
                raise RuntimeError("crash")
            return actual(run, target)

        with patch.object(extra, "pin_model", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "crash"):
                extra.install(self.c)
        self.assertEqual(controller.tasks(self.c), controller.TASKS)
        extra.install(self.c)
        self.assertIn("math_step50", controller.tasks(self.c))

    def test_base_early_and_four_job_cap_with_logic_and_two_exports(self):
        extra.install(self.c)
        states = self.tick()
        self.assertEqual(len(self.pbs.submissions), 4)
        for task in ("logic_algorithmic", "base", "math_step50", "cp_step80"):
            self.assertEqual(states[task]["status"], "QUEUED")
        self.assertEqual(states["base"]["phase"], "test")
        self.assertEqual(states["math_step50"]["phase"], "export")
        self.tick()
        self.assertEqual(len(self.pbs.submissions), 4)
        self.assertFalse(self.pbs.cancellations)

    def test_existing_base_is_adopted_and_never_rearmed(self):
        pending = source.make_job(self.c, "base", "test")
        job = self.pbs.submit(pending["folder"], pending["job_name"])
        self.change("base", status="RUNNING", job=job)
        extra.install(self.c)
        states = self.tick()
        self.assertEqual(states["base"]["job"], job)
        self.assertEqual(len(self.pbs.submissions), 4)
        self.change("base", status="DONE", job=None)
        before = len(self.pbs.submissions)
        self.tick()
        self.assertEqual(len(self.pbs.submissions), before)

    def test_export_test_jobs_bind_alias_checkpoint_and_frozen_runtime(self):
        extra.install(self.c)
        for task, (domain, step) in extra.TARGETS.items():
            job = source.make_job(self.c, task, "export")
            folder = Path(job["folder"])
            r = source.validate_request(folder / "request.json")
            self.assertEqual(
                (r["task"], r["domain"], r["selected"]["step"]), (task, domain, step)
            )
            self.assertEqual(r["config"]["resolved"]["enabled_domains"], [domain])
            result = subprocess.run(
                [sys.executable, "-I", str(folder / "entry.py"), "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run(["bash", "-n", str(folder / "stage.pbs")], check=True)
            exp = self.export(task)
            self.assertFalse(Path(r["selected"]["path"]).exists())
            job2 = source.make_job(self.c, task, "test")
            r2 = source.validate_request(Path(job2["folder"]) / "request.json")
            self.assertEqual(r2["evaluation_model"]["path"], exp["path"])
            self.assertEqual(r2["evaluation_model"]["step"], step)
            source.validate_request(
                folder / "request.json"
            )  # Retired pin's HF proof remains usable.
            with self.assertRaisesRegex(RuntimeError, "cannot train"):
                source.make_job(self.c, task, "train")
        p = extra.plan(self.c)
        (extra.runtime_directory(self.c, p) / "runtime.py").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            source.make_job(self.c, "base", "test")

    def test_271_export_to_test_then_repeated_test_retries_keep_journal(self):
        extra.install(self.c)
        states = self.tick()
        task = "math_step50"
        job = states[task]["job"]
        self.pbs.rows[job["job_id"]] = dict(job_state="F", Exit_status=271)
        states = self.tick()
        self.assertEqual(states[task]["phase"], "export")
        self.export(task)
        job = states[task]["job"]
        write(Path(job["folder"]) / "result.json", dict(status="PASS"))
        self.pbs.rows[job["job_id"]] = dict(job_state="F", Exit_status=0)
        states = self.tick()
        self.assertEqual(states[task]["phase"], "test")
        first = source.validate_request(
            Path(states[task]["job"]["folder"]) / "request.json"
        )
        journal = self.campaign / "runs" / task / "evaluations/test_50_fixture"
        write(journal / "progress.json", dict(completed_rows=840))
        write(journal / "batch_104.json", dict(cached="keep"))
        before = files(journal)
        for i in range(5):
            self.pbs.rows[states[task]["job"]["job_id"]] = dict(
                job_state="F",
                Exit_status=271 if i % 2 else -29,
                comment="walltime exceeded",
            )
            states = self.tick()
            self.assertEqual(states[task]["status"], "QUEUED")
            r = source.validate_request(
                Path(states[task]["job"]["folder"]) / "request.json"
            )
            for k in (
                "run_dir",
                "evaluation_model",
                "verifier_identity",
                "data_manifest",
            ):
                self.assertEqual(r[k], first[k])
        self.assertEqual(files(journal), before)
        self.assertGreater(states[task]["no_progress_retries"], 3)

    def test_unknown_and_ambiguous_submission_remain_owned(self):
        extra.install(self.c)
        self.pbs.ambiguous = True
        states = self.tick()
        self.assertEqual(states["cp_step80"]["status"], "BLOCKED_UNKNOWN")
        self.pbs.ambiguous = False
        self.tick()
        self.assertEqual(len(self.pbs.submissions), 4)
        self.assertIsNotNone(
            read(controller.state_path(self.c, "cp_step80"))["pending"]
        )

    def test_task_fence_applies_to_runtime_and_does_not_stop_logic(self):
        extra.install(self.c)
        states = self.tick()
        request = read(Path(states["math_step50"]["job"]["folder"]) / "request.json")
        write(self.campaign / "runs/math_step50/STOP.json", {})
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            runtime.allowed(request)
        states = self.tick()
        self.assertEqual(states["math_step50"]["status"], "STOPPING")
        self.assertNotIn(
            states["logic_algorithmic"]["job"]["job_id"], self.pbs.cancellations
        )

    def test_repeat_launch_uses_repair_without_install_callback(self):
        extra.install(self.c)
        with patch("d4_teachers.retry_repair.repair", return_value={}) as repair:
            extra.launch(self.campaign)
        repair.assert_called_once_with(self.campaign, install_under_lease=None)

    def test_changed_verifier_cannot_mix_cached_outputs(self):
        extra.install(self.c)
        with patch.object(extra, "verifier", return_value="different"):
            with self.assertRaisesRegex(RuntimeError, "verifier changed"):
                source.make_job(self.c, "base", "test")


class DeployedCheckpointEvaluation(unittest.TestCase):
    def test_real_guardian_upgrade_adopts_logic_and_repeat_submits_nothing(self):
        import test_domain_teachers_retry_repair as old
        from d4_teachers import retry_repair
        from d4_teachers.common import active

        h = old.RetryRepair(methodName="runTest")
        h.setUp()
        self.addCleanup(h.doCleanups)
        for domain in ("math", "if", "conversational_pivot"):
            path = controller.state_path(h.c, domain)
            state = read(path)
            state.update(status="DONE", phase="test", job=None, pending=None)
            write(path, state)
        for domain, step in extra.TARGETS.values():
            run = h.campaign / "runs" / domain
            lineage = read(run / "run.json")["lineage"]
            checkpoint(
                run / "checkpoints" / ("global_step_" + str(step)),
                step,
                lineage=lineage,
            )
            write(
                run / "training_complete.json",
                dict(status="PASS", completed_step=step, lineage=lineage),
            )
        popen = subprocess.Popen

        def record(*args, **kwargs):
            child = popen(*args, **kwargs)
            h.children.append(child)
            return child

        with patch.object(subprocess, "Popen", side_effect=record):
            # Install the current mailbox-aware guardian, then register from it.
            initial = retry_repair.repair(h.campaign)
            self.assertTrue(
                initial["controller_runtime"]["first_tick_completed"], initial
            )
            logic = h.state()["job"]
            before = files(h.campaign / "template")
            result = extra.launch(h.campaign)
            self.assertTrue(
                result["guardian"]["controller_runtime"]["first_tick_completed"], result
            )
            self.assertEqual(h.state()["job"], logic)
            self.assertEqual(files(h.campaign / "template"), before)
            for task in (*extra.TARGETS, "base"):
                self.assertEqual(result["guardian"]["states"][task]["status"], "QUEUED")
            count = sum(call[0] == "qsub" for call in h.calls())
            self.assertEqual(count, 4)  # Logic plus the three requested targets.
            repeated = extra.launch(h.campaign)
            self.assertEqual(
                repeated["guardian"]["controller_runtime"]["runtime"]["pid"],
                result["guardian"]["controller_runtime"]["runtime"]["pid"],
            )
            self.assertEqual(sum(call[0] == "qsub" for call in h.calls()), count)
            self.assertFalse(any(call[0] == "qdel" for call in h.calls()))
            self.assertTrue(active(h.campaign / "guardian.lock"))
