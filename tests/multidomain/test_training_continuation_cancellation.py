"""PBS F-without-exit regression and frozen-controller repair fault tests."""
import copy
import errno
import getpass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from d4_training_continuation import cli, cancellation_repair as repair, process_signal
from d4_training_continuation.common import acquire, digest, files, now, paths, read, sha, verify_files, write
from d4_training_continuation.scheduler import PBS, Unknown, finished, retryable, terminal
from test_training_continuation import Harness


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = dict(job_id="123.pbs1", folder=str(self.root / "old"), job_name="0390_test")
        self.row = dict(Job_Owner=getpass.getuser() + "@login", queue="R9920261000",
                        Job_Name="0390_test", Submit_arguments=str(self.root / "old/stage.pbs"),
                        job_state="F", substate=91, exec_vnode=None,
                        comment="Job held by user and terminated")

    def test_finished_without_exit_can_retire_but_is_never_retryable(self):
        for state in ("F", "C"):
            for code in ("absent", None, ""):
                with self.subTest(state=state, code=code):
                    row = dict(self.row, job_state=state)
                    if code != "absent":
                        row["Exit_status"] = code
                    pbs = PBS()
                    journal = self.root / "cancel.json"
                    with patch.object(pbs, "command", return_value=json.dumps({"Jobs": {"123.pbs1": row}})) as command:
                        self.assertTrue(pbs.cancel(self.job, journal))
                    self.assertEqual(command.call_args.args[0], ["qstat", "-xf", "-F", "json", "123.pbs1"])
                    self.assertEqual(command.call_count, 1)
                    self.assertFalse(journal.exists())
                    self.assertTrue(finished(row))
                    self.assertFalse(terminal(row))
                    self.assertFalse(retryable(row))

    def test_finished_record_still_requires_owner_queue_name_and_path(self):
        for key in ("Job_Owner", "queue", "Job_Name", "Submit_arguments"):
            with self.subTest(key=key):
                pbs = PBS()
                row = dict(self.row, **{key: "wrong"})
                with patch.object(pbs, "jobs", return_value={"123.pbs1": row}), patch.object(pbs, "command") as command:
                    with self.assertRaises(RuntimeError):
                        pbs.cancel(self.job, self.root / "cancel.json")
                command.assert_not_called()

    def test_unknown_query_or_state_never_cancels_and_reports_raw_state(self):
        pbs = PBS()
        with patch.object(pbs, "jobs", side_effect=Unknown("timeout")), patch.object(pbs, "command") as command:
            with self.assertRaises(Unknown):
                pbs.cancel(self.job, self.root / "cancel.json")
        command.assert_not_called()
        row = dict(self.row, job_state="X")
        with patch.object(pbs, "jobs", return_value={"123.pbs1": row}), patch.object(pbs, "command") as command:
            with self.assertRaisesRegex(Unknown, '"job_state": "X"'):
                pbs.cancel(self.job, self.root / "cancel.json")
        command.assert_not_called()
        self.assertFalse((self.root / "cancel.json").exists())

    def test_qdel_requires_a_later_finished_observation(self):
        pbs = PBS()
        rows = [{"123.pbs1": dict(self.row, job_state="H")}, {"123.pbs1": self.row}]
        with patch.object(pbs, "jobs", side_effect=rows), patch.object(pbs, "command", return_value="") as command:
            self.assertFalse(pbs.cancel(self.job, self.root / "cancel.json"))
            self.assertTrue(pbs.cancel(self.job, self.root / "cancel.json"))
        command.assert_called_once_with(["qdel", "123.pbs1"])


class RetireIntegrationTests(Harness):
    def test_real_cancel_unblocks_old_candidate_then_submits_exactly_once(self):
        c = self.controller()
        old = dict(job_id="99.pbs1", job_name="0390_old_candidate", folder=str(self.root / "old_candidate"))
        c.save(old_candidate=old, old_candidate_retired=False)
        self.pbs.rows[old["job_id"]] = dict(job_state="F", substate=91, Exit_status=None)
        self.pbs.cancel = lambda job, journal: PBS.cancel(self.pbs, job, journal)
        single = copy.deepcopy(c.state["active"])
        deadline = c.state["deadline_utc"]
        c.tick(); c.tick()
        self.assertTrue(c.state["old_candidate_retired"])
        self.assertEqual(c.state["active"], single)
        self.assertEqual(c.state["deadline_utc"], deadline)
        self.assertFalse(c.state["handoff_failed"])
        self.assertEqual(len(self.pbs.submissions), 1)
        self.assertEqual(self.pbs.submissions[0]["role"], "candidate")
        self.assertEqual(self.pbs.submissions[0]["expected_nodes"], 2)
        self.assertEqual(self.pbs.cancellations, [])

    def test_missing_training_exit_code_does_not_authorize_a_formal_retry(self):
        c = self.controller()
        c.save(candidate_admitted=False)
        self.pbs.rows[c.state["active"]["job_id"]] = dict(job_state="F", Exit_status=None)
        c.tick(); c.tick()
        self.assertEqual(self.pbs.submissions, [])
        self.assertFalse(c.state.get("training_done", False))


class RepairTests(Harness):
    def frozen(self, kind="random"):
        c = self.controller(kind)
        operation = c.control / "operations/synthetic"
        template = operation / "template"
        scheduler = template / repair.SCHEDULER
        scheduler.parent.mkdir(parents=True)
        scheduler.write_text("# Synthetic original frozen scheduler\n")
        (template / repair.CLI).write_text("# Synthetic original frozen CLI\n")
        (scheduler.parent / "worker.py").write_text("# Training implementation must stay unchanged\n")
        (template / "sampling.json").write_text('{"plan": "unchanged"}\n')
        c.plan.update(template=str(template), template_files=files(template))
        write(operation / "plan.json", c.plan)
        c.save(migration_complete=True, controller_status="STOPPED", pid=os.getpid(), host=socket.gethostname())
        self.old_patch = patch.object(repair, "ORIGINAL_SHA", sha(scheduler))
        self.old_patch.start()
        self.addCleanup(self.old_patch.stop)
        cli_patch = patch.object(repair, "ORIGINAL_CLI_SHAS", {sha(template / repair.CLI)})
        cli_patch.start()
        self.addCleanup(cli_patch.stop)
        return c, template

    def spawn(self, root, kind):
        p = paths(root, kind)
        state = read(p["control"] / "state.json")
        state["controller_status"] = "RUNNING"
        plan = read(p["control"] / "operations" / state["operation"] / "plan.json")
        state["runtime_binding"] = dict(version=cli.RUNTIME_BINDING, pid=state["pid"], host=state["host"],
            at_utc=now(), operation=state["operation"], template=plan["template"],
            template_digest=digest(plan["template_files"]), modules={
                "d4_training_continuation." + Path(relative).stem:
                    dict(path=str(Path(plan["template"]) / relative), sha256=plan["template_files"][relative])
                for relative in (repair.CLI, repair.SCHEDULER)})
        write(p["control"] / "state.json", state)

    def test_repair_publishes_only_controller_changes_and_repetition_preserves_state(self):
        c, original = self.frozen()
        domain = self.controller("domain")
        domain.fail_candidate("Candidate ended before handoff commit: 271")
        domain_before = read(domain.control / "state.json")
        before = read(c.control / "state.json")
        old_files = files(original)
        job = c.p["run"] / "jobs/already_submitted/request.json"
        write(job, {"snapshot": "must remain immutable"})
        before_job = job.read_bytes()
        with patch.object(cli, "spawn", side_effect=self.spawn) as spawn, patch.object(cli, "active", return_value=True):
            first = repair.repair(self.root, "random")
            second = repair.repair(self.root, "random")
        self.assertTrue(first["template_updated"])
        self.assertFalse(second["template_updated"])
        self.assertEqual(first["status"], "REPAIRED_CONTROLLER_RUNNING")
        self.assertEqual(spawn.call_count, 1)
        self.assertFalse(second["controller_restart_requested"])
        self.assertTrue(all(call.args[1] == "random" for call in spawn.call_args_list))
        plan = read(c.control / "operations/synthetic/plan.json")
        changed = {k for k in old_files if old_files[k] != plan["template_files"][k]}
        self.assertEqual(changed, {repair.SCHEDULER, repair.CLI})
        verify_files(plan["template"], plan["template_files"])
        self.assertEqual(files(original), old_files)
        self.assertEqual(job.read_bytes(), before_job)
        after = read(c.control / "state.json")
        after.pop("runtime_binding")
        after["controller_status"] = before["controller_status"]
        self.assertEqual(after, before)
        self.assertEqual(read(domain.control / "state.json"), domain_before)

    def test_failed_handoff_is_not_rearmed_even_when_repair_targets_it(self):
        c, _ = self.frozen("domain")
        c.fail_candidate("Candidate ended before handoff commit: 271")
        with patch.object(cli, "spawn", side_effect=self.spawn), patch.object(cli, "active", return_value=True):
            result = repair.repair(self.root, "domain")
        self.assertTrue(result["handoff_failed"])
        self.assertFalse(result["promoted"])
        self.assertEqual(result["phase"], "SINGLE_ONLY")

    def test_restart_failure_can_resume_from_the_already_published_plan(self):
        c, original = self.frozen()
        with patch.object(cli, "spawn", side_effect=OSError("fork unavailable")):
            with self.assertRaisesRegex(OSError, "fork unavailable"):
                repair.repair(self.root, "random")
        first = read(c.control / "operations/synthetic/plan.json")
        self.assertNotEqual(first["template"], str(original))
        with patch.object(cli, "spawn", side_effect=self.spawn), patch.object(cli, "active", return_value=True):
            result = repair.repair(self.root, "random")
        self.assertFalse(result["template_updated"])
        self.assertEqual(first, read(c.control / "operations/synthetic/plan.json"))

    def test_changed_or_unsupported_source_rejected_before_stopping_owner(self):
        c, template = self.frozen()
        scheduler = template / repair.SCHEDULER
        scheduler.write_text("# Unrecognized edit\n")
        with patch.object(repair, "stop_and_acquire") as stop, patch.object(cli, "spawn") as spawn:
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                repair.repair(self.root, "random")
            c.plan["template_files"] = files(template)
            write(c.control / "operations/synthetic/plan.json", c.plan)
            with self.assertRaisesRegex(RuntimeError, "Unsupported frozen scheduler"):
                repair.repair(self.root, "random")
        stop.assert_not_called(); spawn.assert_not_called()

    def test_interruption_before_publication_leaves_old_plan_and_can_retry(self):
        c, template = self.frozen()
        planfile = c.control / "operations/synthetic/plan.json"
        before = planfile.read_bytes()
        with patch.object(repair, "stop_and_acquire", side_effect=RuntimeError("still releasing")):
            with self.assertRaisesRegex(RuntimeError, "still releasing"):
                repair.repair(self.root, "random")
        self.assertEqual(planfile.read_bytes(), before)
        self.assertEqual(sha(template / repair.SCHEDULER), repair.ORIGINAL_SHA)
        with patch.object(cli, "spawn", side_effect=self.spawn), patch.object(cli, "active", return_value=True):
            self.assertTrue(repair.repair(self.root, "random")["template_updated"])

    def test_foreign_login_host_is_never_signalled(self):
        with patch.object(process_signal, "open_process") as open_pid:
            with self.assertRaisesRegex(RuntimeError, "controller host"):
                repair._signal_owner(dict(host="another-login.invalid", pid=42), Path("/entry"), self.root, "random")
        open_pid.assert_not_called()

    def _live_controller_handoff(self):
        c, _ = self.frozen()
        entry = c.control / "operations/synthetic/controller_entry.py"
        entry.write_text(
            "import fcntl, json, os, signal, socket, time\n"
            "from pathlib import Path\n"
            f"control=Path({str(c.control)!r})\n"
            "lock=(control/'controller.lock').open('a')\n"
            "fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "state=json.loads((control/'state.json').read_text())\n"
            "state.update(pid=os.getpid(), host=socket.gethostname(), controller_status='RUNNING')\n"
            "stop=[False]\n"
            "signal.signal(signal.SIGTERM, lambda *args: stop.__setitem__(0, True))\n"
            "(control/'state.json').write_text(json.dumps(state))\n"
            "print(json.dumps({'uid': os.getuid(), 'argv': [os.fsdecode(x) for x in Path('/proc/self/cmdline').read_bytes().split(b'\\0') if x]}), flush=True)\n"
            "while not stop[0]: time.sleep(0.01)\n"
            "state['controller_status']='STOPPED'\n"
            "(control/'state.json').write_text(json.dumps(state))\n"
            "lock.close()\n"
        )
        child = subprocess.Popen([sys.executable, "-I", str(entry), "_run", "--root", str(self.root), "--kind", "random"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: child.kill() if child.poll() is None else None)
        identity = json.loads(child.stdout.readline())
        state = read(c.control / "state.json")
        # The test executor's /proc/PID mount can use a different PID namespace;
        # use the real child's self-reported /proc/self identity. The pidfd signal,
        # SIGTERM handler and ownership-lock handoff below are real.
        with patch.object(repair, "_process_identity", return_value=(identity["uid"], identity["argv"])):
            with self.assertRaisesRegex(RuntimeError, "command line mismatch"):
                repair._signal_owner(state, entry, self.root, "domain")
            self.assertIsNone(child.poll())
            with repair.stop_and_acquire(c.control, self.root, "random", timeout=3):
                self.assertEqual(read(c.control / "state.json")["controller_status"], "STOPPED")
                with self.assertRaises(BlockingIOError):
                    acquire(c.control / "controller.lock")
        self.assertEqual(child.wait(timeout=3), 0)
        child.stdout.close(); child.stderr.close()

    def test_live_controller_stops_gracefully_and_exact_command_is_required(self):
        self._live_controller_handoff()

    def test_live_controller_stops_when_python_pidfd_wrappers_are_missing(self):
        with patch.object(os, "pidfd_open", None, create=True), \
             patch.object(signal, "pidfd_send_signal", None, create=True), \
             patch.object(os, "kill", side_effect=AssertionError("Numeric-PID fallback is forbidden")):
            self._live_controller_handoff()

    def test_kernel_probe_failure_leaves_live_owner_and_plan_unchanged(self):
        c, _ = self.frozen()
        original = read(c.control / "operations/synthetic/plan.json")
        for error in (errno.ENOSYS, errno.EPERM):
            with self.subTest(error=error), \
                 patch.object(process_signal, "probe", side_effect=OSError(error, "kernel check failed")), \
                 patch.object(repair, "stop_and_acquire") as stop, patch.object(cli, "spawn") as spawn:
                with self.assertRaises(OSError):
                    repair.repair(self.root, "random")
                stop.assert_not_called(); spawn.assert_not_called()
                self.assertEqual(read(c.control / "operations/synthetic/plan.json"), original)


class ProcessSignalTests(unittest.TestCase):
    def test_probe_without_python_bindings_uses_kernel_pidfd_and_signal_zero(self):
        with patch.object(os, "pidfd_open", None, create=True), \
             patch.object(signal, "pidfd_send_signal", None, create=True):
            report = process_signal.probe()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["open_backend"], "linux-syscall")
        self.assertEqual(report["signal_backend"], "linux-syscall")

    def test_each_python_wrapper_can_be_missing_independently(self):
        for module, name in ((os, "pidfd_open"), (signal, "pidfd_send_signal")):
            with self.subTest(name=name), patch.object(module, name, None, create=True):
                self.assertEqual(process_signal.probe()["status"], "PASS")

    def test_kernel_errors_propagate_without_numeric_pid_fallback(self):
        for error in (errno.ENOSYS, errno.EPERM, errno.ESRCH):
            with self.subTest(error=error), \
                 patch.object(os, "pidfd_open", None, create=True), \
                 patch.object(process_signal, "_syscall", side_effect=OSError(error, "kernel error")), \
                 patch.object(os, "kill") as kill:
                with self.assertRaises(OSError) as caught:
                    process_signal.open_process(42)
                self.assertEqual(caught.exception.errno, error)
                kill.assert_not_called()

    def test_unknown_architecture_is_rejected_before_any_native_call(self):
        with patch.object(os, "pidfd_open", None, create=True), \
             patch.object(process_signal.platform, "machine", return_value="unsupported"), \
             patch.object(process_signal.ctypes, "CDLL") as libc:
            with self.assertRaisesRegex(RuntimeError, "compatibility supports"):
                process_signal.open_process(42)
        libc.assert_not_called()

    def test_probe_closes_descriptor_if_signal_check_fails(self):
        with patch.object(process_signal, "open_process", return_value=123), \
             patch.object(process_signal, "send_signal", side_effect=PermissionError(errno.EPERM, "denied")), \
             patch.object(os, "close") as close:
            with self.assertRaises(PermissionError):
                process_signal.probe()
        close.assert_called_once_with(123)


if __name__ == "__main__":
    unittest.main()
