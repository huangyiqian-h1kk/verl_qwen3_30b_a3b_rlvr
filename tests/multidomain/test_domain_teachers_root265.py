"""Root-terminated attempts recover under one real guardian; PBS/GPU are fixtures."""

from pathlib import Path
import shutil
import socket
import subprocess
import unittest
from unittest.mock import patch

import test_domain_teachers_retry_repair as fixtures
from test_training_continuation import checkpoint
from d4_teachers import controller, guardian_handoff, retry_repair as repair
from d4_teachers.common import digest, file_sha, files, read, write
from d4_training_continuation.scheduler import PBS, retryable


ROOT_COMMENT = (
    "Job run at Wed Nov 04 at 16:09 on (node[0]:ngpus=8) and terminated by root@login3"
)
OLD_CONTROLLER = (
    Path(__file__).parent / "fixtures/domain_teachers/controller_before_root265.py"
)
OLD_SHA = "32894bd56d3d395678d4cd838e7a6471d15af590207076a8d98cf97dd3415bfe"


class Root265Recovery(unittest.TestCase):
    def setUp(self):
        self.h = fixtures.RetryRepair(methodName="runTest")
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.c, self.root = self.h.c, self.h.campaign
        self.job = self.h.jobs["logic_algorithmic"]
        self.folder = Path(self.job["folder"])
        self.run = self.root / "runs/logic_algorithmic"
        checkpoint(
            self.run / "checkpoints/global_step_55",
            55,
            lineage=read(self.run / "run.json")["lineage"],
        )
        write(self.run / "progress.json", dict(step=56, last_full_step=55))
        state = self.h.state()
        state.update(
            status="BLOCKED_APPLICATION",
            job=self.job,
            no_progress_retries=54,
            last_progress=5500000,
        )
        write(controller.state_path(self.c, "logic_algorithmic"), state)
        self.h.change_row(Exit_status=265, comment=ROOT_COMMENT)

    def tick(self):
        return controller.tick(self.c, PBS())

    def submissions(self):
        return [call for call in self.h.calls() if call[0] == "qsub"]

    def test_blocked_startup_kill_retries_once_from_durable_progress(self):
        preserved = {
            name: files(self.root / name)
            for name in ("template", "data", "runs/logic_algorithmic/checkpoints")
        }
        attempt = files(self.folder)
        state = self.tick()["logic_algorithmic"]
        self.assertEqual(state["status"], "QUEUED")
        self.assertNotEqual(state["job"]["job_id"], self.job["job_id"])
        self.assertEqual(state["last_progress"], 5500000)
        self.assertEqual(state["no_progress_retries"], 55)
        self.assertEqual(state["last_interruption"]["job_id"], self.job["job_id"])
        self.assertEqual(
            state["last_interruption"]["reason"], "pbs_265_root_termination"
        )
        self.assertEqual(state["last_interruption"]["policy"], controller.RETRY_POLICY)
        self.tick()
        self.assertEqual(len(self.submissions()), 1)
        self.assertFalse(any(call[0] == "qdel" for call in self.h.calls()))
        for name, snapshot in preserved.items():
            self.assertEqual(files(self.root / name), snapshot)
        self.assertEqual(files(self.folder), attempt)
        for task in ("math", "if"):
            self.assertEqual(
                read(controller.state_path(self.c, task))["job"], self.h.jobs[task]
            )

    def test_missing_or_ambiguous_root_evidence_does_not_retry(self):
        for comment in (
            "",
            "killed",
            "terminated by user@login3",
            "not terminated by root@login3",
            "terminated by root@",
            "terminated by root@login3 is unconfirmed",
        ):
            with self.subTest(comment=comment):
                self.h.change_row(comment=comment)
                state = self.tick()["logic_algorithmic"]
                self.assertEqual(state["status"], "BLOCKED_APPLICATION")
                self.assertEqual(state["job"], self.job)
        self.assertEqual(self.submissions(), [])

    def test_known_application_results_and_completion_markers_remain_blocked(self):
        result, marker = self.folder / "result.json", self.folder / "finished"
        cases = [
            (dict(status="FAIL", error="verifier failure"), None),
            (dict(status="PASS"), None),
            (dict(status="RUNNING", error="recorded failure"), None),
            ({}, None),
            (None, "1"),
            (None, "0"),
            (None, "143"),
            (None, "invalid"),
        ]
        for report, exit_code in cases:
            with self.subTest(report=report, exit_code=exit_code):
                result.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                if report is not None:
                    write(result, report)
                if exit_code is not None:
                    marker.write_text(exit_code)
                state = self.tick()["logic_algorithmic"]
                self.assertEqual(state["status"], "BLOCKED_APPLICATION")
                self.assertEqual(state["job"], self.job)
        self.assertEqual(self.submissions(), [])

    def test_running_result_and_signal_marker_allow_recovery(self):
        write(self.folder / "result.json", dict(status="RUNNING"))
        (self.folder / "finished").write_text("137\n")
        self.assertEqual(self.tick()["logic_algorithmic"]["status"], "QUEUED")
        self.assertEqual(len(self.submissions()), 1)

    def test_unreadable_result_preserves_job_and_diagnostic(self):
        (self.folder / "result.json").write_text("{")
        state = self.tick()["logic_algorithmic"]
        self.assertEqual(state["status"], "BLOCKED_ARTIFACT")
        self.assertIn("JSONDecodeError", state["error"])
        self.assertEqual(state["job"], self.job)
        self.assertEqual(self.submissions(), [])

    def test_pbs_owner_queue_name_and_submission_path_must_match(self):
        rows = read(self.h.root / "pbs_rows.json")
        original = dict(rows[self.job["job_id"]])
        for key, value in (
            ("Job_Owner", "different-user@login3"),
            ("queue", "another-queue"),
            ("Job_Name", "another-job"),
            ("Submit_arguments", "/other/stage.pbs"),
        ):
            with self.subTest(field=key):
                rows[self.job["job_id"]] = dict(original, **{key: value})
                write(self.h.root / "pbs_rows.json", rows)
                state = self.tick()["logic_algorithmic"]
                self.assertEqual(state["status"], "BLOCKED_ARTIFACT")
                self.assertEqual(state["job"], self.job)
        self.assertEqual(self.submissions(), [])

    def test_unknown_and_live_observations_never_duplicate(self):
        unavailable = self.h.root / "pbs_unavailable"
        unavailable.touch()
        state = self.tick()["logic_algorithmic"]
        self.assertEqual(state["status"], "BLOCKED_UNKNOWN")
        self.assertEqual(state["job"], self.job)
        unavailable.unlink()
        self.h.change_row(job_state="R")
        self.assertEqual(self.tick()["logic_algorithmic"]["status"], "RUNNING")
        self.assertEqual(self.submissions(), [])
        self.h.change_row(job_state="F")
        self.assertEqual(self.tick()["logic_algorithmic"]["status"], "QUEUED")
        self.tick()
        self.assertEqual(len(self.submissions()), 1)

    def test_stop_fences_prevent_root_kill_recovery(self):
        original = self.h.state()
        for fence in (self.root / "STOP.json", self.run / "STOP.json"):
            with self.subTest(fence=fence.name):
                write(fence, {})
                state = dict(original)
                controller.reconcile(self.c, "logic_algorithmic", state, PBS())
                self.assertEqual(state["status"], "STOPPED")
                self.assertIsNone(state["job"])
                fence.unlink()
        self.assertEqual(self.submissions(), [])

    def test_original_271_walltime_and_other_exit_codes_keep_their_rules(self):
        for code, comment, eligible in (
            (271, "", True),
            (-29, "walltime exceeded", True),
            (-29, "killed", False),
            (1, ROOT_COMMENT, False),
            (0, ROOT_COMMENT, False),
            (None, ROOT_COMMENT, False),
            (265, ROOT_COMMENT, True),
        ):
            with self.subTest(code=code, comment=comment):
                row = dict(job_state="F", Exit_status=code, comment=comment)
                self.assertEqual(bool(controller.retry_reason(row, self.job)), eligible)
        self.assertFalse(
            retryable(dict(job_state="F", Exit_status=265, comment=ROOT_COMMENT))
        )

    def test_all_phases_recover_without_changing_stage_or_evaluation_journal(self):
        original = self.h.state()
        write(
            self.run / "evaluations/test_30_fixture/progress.json",
            dict(completed_rows=8),
        )
        journal = files(self.run / "evaluations")
        for phase in ("train", "export", "test"):
            with self.subTest(phase=phase):
                state = dict(original, phase=phase)
                controller.reconcile(self.c, "logic_algorithmic", state, PBS())
                self.assertEqual((state["status"], state["phase"]), ("READY", phase))
                self.assertEqual(files(self.run / "evaluations"), journal)
        self.assertEqual(self.submissions(), [])

    def test_published_template_controller_is_accepted(self):
        self.assertEqual(file_sha(OLD_CONTROLLER), OLD_SHA)
        shutil.copy2(
            OLD_CONTROLLER, self.root / "template/runtime/d4_teachers/controller.py"
        )
        revision = repair.prepare_revision(self.c)
        target = self.root / "controller_revisions" / revision["identity"] / "runtime"
        self.assertEqual(
            file_sha(target / "d4_teachers/controller.py"),
            file_sha(controller.__file__),
        )

    def test_real_previous_owner_handoff_resubmits_once_from_another_host(self):
        current = repair.prepare_revision(self.c)
        payload = {k: v for k, v in current.items() if k != "identity"}
        payload["files"] = dict(
            payload["files"], **{"d4_teachers/controller.py": OLD_SHA}
        )
        previous = dict(payload, identity=digest(payload))
        revisions = self.root / "controller_revisions"
        previous_dir = revisions / previous["identity"]
        shutil.copytree(
            revisions / current["identity"] / "runtime", previous_dir / "runtime"
        )
        shutil.copy2(OLD_CONTROLLER, previous_dir / "runtime/d4_teachers/controller.py")
        write(previous_dir / "manifest.json", previous)
        write(self.root / "controller_revision.json", previous)
        (self.root / "control.py").write_text(repair.ENTRY)
        old = self.h.start_old()
        self.assertEqual(
            read(self.root / "guardian_runtime.json")["retry_policy"],
            "four-domain-external-interruption-v1",
        )
        preserved = {
            name: files(self.root / name)
            for name in (
                "template",
                "data",
                "runs/logic_algorithmic/checkpoints",
            )
        }
        popen = subprocess.Popen

        def record(*args, **kwargs):
            child = popen(*args, **kwargs)
            self.h.children.append(child)
            return child

        with (
            patch.object(subprocess, "Popen", side_effect=record),
            patch.object(socket, "gethostname", return_value="login4"),
            patch.object(
                repair, "signal_local_owner", side_effect=AssertionError("remote PID")
            ),
            patch.object(
                guardian_handoff, "remote_signal", side_effect=AssertionError("SSH")
            ),
        ):
            first = repair.repair(self.root, wait_seconds=10)
            self.assertEqual(first["status"], "REPAIRED_GUARDIAN_RUNNING", first)
            self.assertEqual(old.wait(timeout=5), 0)
            self.assertEqual(first["retry_policy"], controller.RETRY_POLICY)
            self.assertEqual(first["states"]["logic_algorithmic"]["status"], "QUEUED")
            pid = first["controller_runtime"]["runtime"]["pid"]
            repeated = repair.repair(self.root, wait_seconds=2)
            self.assertEqual(repeated["controller_runtime"]["runtime"]["pid"], pid)
        self.assertEqual(len(self.submissions()), 1)
        self.assertEqual(self.h.state()["last_progress"], 5500000)
        self.assertFalse(any(call[0] == "qdel" for call in self.h.calls()))
        for name, snapshot in preserved.items():
            self.assertEqual(files(self.root / name), snapshot)
        for task in ("math", "if"):
            self.assertEqual(
                read(controller.state_path(self.c, task))["job"], self.h.jobs[task]
            )
        self.assertEqual(
            read(controller.state_path(self.c, "conversational_pivot"))["status"],
            "BLOCKED_APPLICATION",
        )
