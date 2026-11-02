"""SIGKILL export recovery with real contracts/locks and synthetic PBS/weights."""

import getpass
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_domain_teachers as fixtures
import test_domain_teachers_checkpoint_evaluation as extra_fixtures
from test_training_continuation import checkpoint
from d4_teachers import checkpoint_evaluation as extra, controller, retry_repair, source
from d4_teachers.common import files, lock, read, write
from d4_training_continuation.scheduler import PBS, Unknown, retryable, validate_job

SCRIPT = source.REPO / "scripts/retry_teacher_export.py"
SPEC = importlib.util.spec_from_file_location("teacher_export_retry", SCRIPT)
retry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)


class BoundPBS(fixtures.FakePBS):
    def submit(self, folder, name):
        job = super().submit(folder, name)
        self.rows[job["job_id"]].update(
            Job_Owner=getpass.getuser() + "@login",
            queue="R9920261000",
            Job_Name=name,
            Submit_arguments=str(Path(folder) / "stage.pbs"),
        )
        return job

    def observe(self, job):
        return validate_job(job, super().observe(job))


class ExportRetry(unittest.TestCase):
    def setUp(self):
        self.h = extra_fixtures.CheckpointEvaluation(methodName="runTest")
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.c, self.root = self.h.c, self.h.campaign
        extra.install(self.c)
        self.pbs = BoundPBS()
        self.h.pbs = self.pbs
        self.h.change("base", status="DONE")
        self.h.change("cp_step80", status="DONE")
        self.logic = self.submit("logic_algorithmic", "train")
        self.h.change("logic_algorithmic", status="RUNNING", job=self.logic)
        self.job = self.submit("math_step50", "export")
        self.jid = self.job["job_id"]
        self.pbs.rows[self.jid].update(job_state="F", Exit_status=265)
        self.h.change(
            "math_step50",
            status="BLOCKED_APPLICATION",
            job=self.job,
            no_progress_retries=3,
            error="unsupported terminal status",
        )
        self.run = self.root / "runs/math_step50"
        self.before = read(self.run / "state.json")

    def submit(self, task, phase):
        intent = source.make_job(self.c, task, phase)
        return self.pbs.submit(intent["folder"], intent["job_name"])

    def rearm(self):
        with lock(self.root / "control.lock"), lock(self.root / "guardian.lock"):
            return retry.rearm_export(self.c, "math_step50", self.jid, self.pbs)

    def test_single_retry_preserves_sources_and_advances_export_to_test(self):
        original = files(self.root / "runs/math")
        job_files = files(self.job["folder"])
        result = self.rearm()
        self.assertEqual(result["status"], "REARMED_EXPORT")
        self.assertEqual(read(result["receipt"])["before"], self.before)
        self.assertEqual(files(self.root / "runs/math"), original)
        self.assertEqual(files(self.job["folder"]), job_files)
        state = read(self.run / "state.json")
        self.assertIsNone(state["job"])
        self.assertEqual(state["previous_job"], self.job)
        self.assertEqual(state["no_progress_retries"], 3)
        state = self.h.tick()["math_step50"]
        export_job = state["job"]
        self.assertEqual(state["phase"], "export")
        request = source.validate_request(Path(export_job["folder"]) / "request.json")
        self.assertEqual(request["selected"]["step"], 50)
        self.h.export("math_step50")
        write(Path(export_job["folder"]) / "result.json", dict(status="PASS"))
        self.pbs.rows[export_job["job_id"]].update(job_state="F", Exit_status=0)
        state = self.h.tick()["math_step50"]
        self.assertEqual((state["status"], state["phase"]), ("QUEUED", "test"))
        request = source.validate_request(Path(state["job"]["folder"]) / "request.json")
        self.assertEqual(request["evaluation_model"]["step"], 50)
        self.assertEqual(
            read(self.root / "runs/logic_algorithmic/state.json")["job"], self.logic
        )
        self.assertEqual(len(self.pbs.submissions), 4)
        self.assertEqual(self.pbs.cancellations, [])

    def test_repeat_does_not_rearm_a_later_265_or_broaden_automatic_policy(self):
        self.rearm()
        state = self.h.tick()["math_step50"]
        later = state["job"]
        self.pbs.rows[later["job_id"]].update(job_state="F", Exit_status=265)
        state = self.h.tick()["math_step50"]
        self.assertEqual(state["status"], "BLOCKED_APPLICATION")
        self.assertEqual(self.rearm()["status"], "ALREADY_REARMED")
        self.h.tick()
        self.assertEqual(len(self.pbs.submissions), 3)
        self.assertFalse(retryable(dict(job_state="F", Exit_status=265)))
        self.assertTrue(retryable(dict(job_state="F", Exit_status=271)))

    def test_new_attempt_retains_271_recovery(self):
        self.rearm()
        state = self.h.tick()["math_step50"]
        self.pbs.rows[state["job"]["job_id"]].update(job_state="F", Exit_status=271)
        state = self.h.tick()["math_step50"]
        self.assertEqual(state["status"], "QUEUED")
        self.assertEqual(len(self.pbs.submissions), 4)
        self.assertEqual(self.pbs.cancellations, [])

    def test_ambiguous_new_submission_stays_owned_on_repeat(self):
        self.rearm()
        self.pbs.ambiguous = True
        state = self.h.tick()["math_step50"]
        self.assertEqual(state["status"], "BLOCKED_UNKNOWN")
        pending = state["pending"]
        self.assertIsNotNone(pending)
        self.assertEqual(self.rearm()["status"], "ALREADY_REARMED")
        again = self.h.tick()["math_step50"]
        self.assertEqual(again["pending"], pending)
        self.assertEqual(len(self.pbs.submissions), 3)
        self.assertEqual(self.pbs.cancellations, [])

    def test_live_unknown_and_other_exit_codes_never_rearm(self):
        for state, code in [
            ("Q", 265),
            ("R", 265),
            ("F", None),
            ("F", 1),
            ("F", 0),
            ("F", 271),
        ]:
            with self.subTest(state=state, code=code):
                self.pbs.rows[self.jid].update(job_state=state, Exit_status=code)
                with self.assertRaisesRegex(RuntimeError, "freshly confirmed"):
                    self.rearm()
                self.assertEqual(read(self.run / "state.json"), self.before)
        self.pbs.unknown = True
        with patch.object(retry_repair, "repair") as repair:
            with self.assertRaises(Unknown):
                retry.retry_export(self.root, "math_step50", self.jid, self.pbs)
            repair.assert_not_called()

    def test_pbs_identity_checks_prevent_foreign_job_retry(self):
        for key, value in [
            ("Job_Owner", "foreign@login"),
            ("queue", "other"),
            ("Job_Name", "other"),
            ("Submit_arguments", "/other/stage.pbs"),
        ]:
            with self.subTest(key=key):
                original = self.pbs.rows[self.jid][key]
                self.pbs.rows[self.jid][key] = value
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    self.rearm()
                self.pbs.rows[self.jid][key] = original
        self.assertEqual(read(self.run / "state.json"), self.before)

    def test_stop_fences_and_writer_ownership_prevent_rearm(self):
        for fence in [self.root / "STOP.json", self.run / "STOP.json"]:
            write(fence, {})
            with self.assertRaisesRegex(RuntimeError, "stop fence"):
                self.rearm()
            fence.unlink()
        with lock(self.run / "writer.lock"):
            with self.assertRaises(BlockingIOError):
                self.rearm()
        self.assertEqual(read(self.run / "state.json"), self.before)

    def test_runtime_or_output_evidence_requires_separate_diagnosis(self):
        for path in [
            Path(self.job["folder"]) / "result.json",
            self.run / "export.json",
            self.run / "test.json",
        ]:
            with self.subTest(path=path.name):
                write(path, dict(status="FAIL"))
                with self.assertRaisesRegex(RuntimeError, "startup-only"):
                    self.rearm()
                path.unlink()
        self.assertEqual(read(self.run / "state.json"), self.before)

    def test_changed_task_pending_request_or_checkpoint_is_rejected(self):
        for change in [
            dict(phase="test"),
            dict(pending={"folder": "ambiguous"}),
            dict(job={**self.job, "job_id": "999.pbs1"}),
        ]:
            write(self.run / "state.json", {**self.before, **change})
            with self.assertRaisesRegex(RuntimeError, "Task changed"):
                self.rearm()
        write(self.run / "state.json", self.before)
        payload = next((self.run / "best_weights").rglob("*.distcp"))
        payload.write_bytes(b"changed model")
        with self.assertRaises(RuntimeError):
            self.rearm()
        self.assertEqual(read(self.run / "state.json"), self.before)

    def test_audit_write_before_state_crash_is_replayable(self):
        from d4_teachers import common

        real_write = common.write

        def fail_state(path, value):
            if Path(path) == self.run / "state.json":
                raise OSError("injected state commit failure")
            return real_write(path, value)

        with patch.object(common, "write", side_effect=fail_state):
            with self.assertRaisesRegex(OSError, "commit failure"):
                self.rearm()
        self.assertEqual(read(self.run / "state.json"), self.before)
        audits = list((self.root / "repairs" / retry.POLICY).glob("*.json"))
        self.assertEqual(len(audits), 1)
        audit = audits[0].read_bytes()
        self.rearm()
        self.assertEqual(audits[0].read_bytes(), audit)
        self.assertEqual(self.rearm()["status"], "ALREADY_REARMED")

    def test_repeat_still_ensures_guardian_after_committed_retry(self):
        self.rearm()
        with patch.object(retry_repair, "repair", return_value={}) as repair:
            result = retry.retry_export(self.root, "math_step50", self.jid, self.pbs)
            self.assertEqual(result["status"], "ALREADY_REARMED")
            repair.assert_called_once_with(
                self.root, pbs=self.pbs, install_under_lease=None
            )


class DeployedExportRetry(unittest.TestCase):
    def test_real_guardian_handoff_submits_once_and_adopts_logic(self):
        import test_domain_teachers_retry_repair as old

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
        extra.install(h.c)
        for task in ("cp_step80", "base"):
            path = controller.state_path(h.c, task)
            write(path, {**read(path), "status": "DONE"})
        path = controller.state_path(h.c, "logic_algorithmic")
        write(
            path,
            {**read(path), "status": "RUNNING", "job": h.jobs["logic_algorithmic"]},
        )
        rows = read(h.root / "pbs_rows.json")
        rows[h.jobs["logic_algorithmic"]["job_id"]].update(
            job_state="R", Exit_status=None
        )
        write(h.root / "pbs_rows.json", rows)
        intent = source.make_job(h.c, "math_step50", "export")
        job = PBS().submit(intent["folder"], intent["job_name"])
        rows = read(h.root / "pbs_rows.json")
        rows[job["job_id"]].update(job_state="F", Exit_status=265)
        write(h.root / "pbs_rows.json", rows)
        path = controller.state_path(h.c, "math_step50")
        write(path, {**read(path), "status": "BLOCKED_APPLICATION", "job": job})
        real_popen = subprocess.Popen

        def track(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            h.children.append(child)
            return child

        with patch.object(subprocess, "Popen", side_effect=track):
            initial = retry_repair.repair(h.campaign)
            self.assertTrue(initial["controller_runtime"]["first_tick_completed"])
            logic = h.state()["job"]
            frozen = files(h.campaign / "template")
            original = files(job["folder"])
            count = sum(call[0] == "qsub" for call in h.calls())
            result = retry.retry_export(h.campaign, "math_step50", job["job_id"])
            guardian = result["guardian"]
            self.assertTrue(
                guardian["controller_runtime"]["first_tick_completed"], result
            )
            self.assertEqual(guardian["states"]["math_step50"]["status"], "QUEUED")
            self.assertEqual(h.state()["job"], logic)
            self.assertEqual(files(h.campaign / "template"), frozen)
            self.assertEqual(files(job["folder"]), original)
            self.assertEqual(sum(call[0] == "qsub" for call in h.calls()), count + 1)
            again = retry.retry_export(h.campaign, "math_step50", job["job_id"])
            self.assertEqual(again["status"], "ALREADY_REARMED")
            self.assertEqual(
                again["guardian"]["controller_runtime"]["runtime"]["pid"],
                guardian["controller_runtime"]["runtime"]["pid"],
            )
            self.assertEqual(sum(call[0] == "qsub" for call in h.calls()), count + 1)
            self.assertFalse(any(call[0] == "qdel" for call in h.calls()))

    def test_isolated_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--expected-job", result.stdout)
