"""Real process/lock handoffs; PBS and SSH transport use inspectable fixtures."""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_domain_teachers_retry_repair as fixtures
from d4_teachers import controller, guardian_handoff as handoff, retry_repair as repair
from d4_teachers.common import active, digest, file_sha, files, lock, read, write


def install_ssh_transport(h):
    """Execute the real helper; adapt hostname and executor /proc only."""
    child_code = (
        "import json,os,socket,sys\nfrom pathlib import Path\n"
        "socket.gethostname=lambda:'login1'\n"
        "sys.path.insert(0,str(Path(sys.argv[1]).resolve().parents[1]))\n"
        "from d4_teachers import guardian_handoff,retry_repair\n"
        "identity=json.loads((Path(os.environ['TEACHER_TEST_PBS'])/'old_identity.json').read_text())\n"
        "retry_repair._process_identity=lambda pid:(identity['uid'],identity['argv'])\n"
        "sys.argv=sys.argv[1:]\nguardian_handoff.main()\n"
    )
    program = (
        "import json,os,shlex,subprocess,sys\nfrom pathlib import Path\n"
        "root=Path(os.environ['TEACHER_TEST_PBS'])\n"
        "with (root/'ssh_calls.jsonl').open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if (root/'ssh_fail').exists():sys.stderr.write('fixture SSH unavailable');raise SystemExit(255)\n"
        "args=shlex.split(sys.argv[-1])\n"
        "assert sys.argv[-2]=='login1' and args[1]=='-I'\n"
        f"code={child_code!r}\n"
        "result=subprocess.run([args[0],'-I','-c',code,*args[2:]])\n"
        "raise SystemExit(result.returncode)\n"
    )
    ssh = h.root / "fake_bin/ssh"
    ssh.write_text("#!" + sys.executable + "\n" + program)
    ssh.chmod(0o700)


class HostIndependentRepair(unittest.TestCase):
    def setUp(self):
        self.h = fixtures.RetryRepair(methodName="runTest")
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.root = self.h.campaign
        self.popen = subprocess.Popen
        install_ssh_transport(self.h)

    def record(self, *args, **kwargs):
        child = self.popen(*args, **kwargs)
        self.h.children.append(child)
        return child

    def legacy_owner(self):
        # Deploy the exact abe93f4/ebcc5cd controller, including its runtime file.
        previous = (
            Path(__file__).parent
            / "fixtures/domain_teachers/controller_before_handoff.py"
        )
        self.assertEqual(file_sha(previous), repair.PREVIOUS_OBSERVER_SHA)
        revision = repair.prepare_revision(self.h.c)
        payload = {k: v for k, v in revision.items() if k != "identity"}
        payload["files"] = dict(
            payload["files"], **{"d4_teachers/controller.py": file_sha(previous)}
        )
        older = dict(payload, identity=digest(payload))
        revisions = self.root / "controller_revisions"
        target = revisions / older["identity"]
        shutil.copytree(
            revisions / revision["identity"] / "runtime", target / "runtime"
        )
        shutil.copy2(previous, target / "runtime/d4_teachers/controller.py")
        write(target / "manifest.json", older)
        write(self.root / "controller_revision.json", older)
        (self.root / "control.py").write_text(repair.ENTRY)
        child = self.h.start_old()
        for name in ("guardian_runtime.json", "heartbeat.json"):
            value = read(self.root / name)
            value["host"] = "login1"
            write(self.root / name, value)
        return child

    def test_legacy_remote_handoff_then_cooperative_handoff_and_repeat(self):
        old = self.legacy_owner()
        preserved = {
            name: files(self.root / name)
            for name in ("template", "data", "runs/logic_algorithmic/checkpoints")
        }
        campaign = file_sha(self.root / "campaign.json")
        requests = {
            d: file_sha(Path(j["folder"]) / "request.json")
            for d, j in self.h.jobs.items()
        }
        with (
            patch.object(subprocess, "Popen", side_effect=self.record),
            patch.object(socket, "gethostname", return_value="login2"),
            patch.object(
                repair,
                "signal_local_owner",
                side_effect=AssertionError("client must not signal remote PID"),
            ),
        ):
            first = repair.repair(self.root, wait_seconds=10)
            self.assertEqual(first["status"], "REPAIRED_GUARDIAN_RUNNING", first)
            self.assertEqual(old.wait(timeout=5), -15)
            first_pid = first["controller_runtime"]["runtime"]["pid"]
            first_process = next(p for p in self.h.children if p.pid == first_pid)
            job = self.h.state()["job"]
            # A new owner hands off via shared files despite a different client
            # hostname and completely unavailable SSH/signal paths.
            with patch.object(
                handoff, "remote_signal", side_effect=AssertionError("SSH must not run")
            ):
                second = repair.repair(
                    self.root,
                    wait_seconds=10,
                    install_under_lease=lambda c: dict(status="INSTALLED"),
                )
                self.assertEqual(first_process.wait(timeout=5), 0)
                self.assertNotEqual(
                    second["controller_runtime"]["runtime"]["pid"], first_pid
                )
                self.assertEqual(self.h.state()["job"], job)
                repeated = repair.repair(self.root, wait_seconds=2)
                self.assertEqual(
                    repeated["controller_runtime"]["runtime"]["pid"],
                    second["controller_runtime"]["runtime"]["pid"],
                )
        receipts = [
            read(p)
            for p in (self.root / "repairs/guardian_handoffs").glob("*.json")
            if not p.name.endswith(".owner.json")
        ]
        self.assertEqual(
            {r["method"] for r in receipts}, {"ssh_legacy", "shared_filesystem"}
        )
        self.assertTrue(all(r["status"] == "LOCK_ACQUIRED" for r in receipts))
        self.assertEqual(
            len((self.h.root / "ssh_calls.jsonl").read_text().splitlines()), 1
        )
        self.assertFalse((self.root / handoff.REQUEST).exists())
        self.assertTrue(active(self.root / "guardian.lock"))
        self.assertEqual([c[0] for c in self.h.calls()].count("qsub"), 1)
        self.assertFalse(any(c[0] == "qdel" for c in self.h.calls()))
        self.assertEqual(file_sha(self.root / "campaign.json"), campaign)
        for name, before in preserved.items():
            self.assertEqual(files(self.root / name), before)
        self.assertEqual(
            {
                d: file_sha(Path(j["folder"]) / "request.json")
                for d, j in self.h.jobs.items()
            },
            requests,
        )
        self.assertEqual(
            read(controller.state_path(self.h.c, "conversational_pivot"))["status"],
            "BLOCKED_APPLICATION",
        )

    def test_ssh_failure_preserves_owner_jobs_and_cancels_request(self):
        old = self.legacy_owner()
        pointer = read(self.root / "controller_revision.json")
        before = {d: read(controller.state_path(self.h.c, d)) for d in controller.TASKS}
        (self.h.root / "ssh_fail").touch()
        with patch.object(socket, "gethostname", return_value="login2"):
            with self.assertRaisesRegex(
                RuntimeError, "lock is still held.*SSH unavailable"
            ):
                repair.repair(self.root, wait_seconds=0)
        self.assertIsNone(old.poll())
        self.assertTrue(active(self.root / "guardian.lock"))
        self.assertEqual(read(self.root / "controller_revision.json"), pointer)
        self.assertEqual(
            {d: read(controller.state_path(self.h.c, d)) for d in controller.TASKS},
            before,
        )
        self.assertFalse((self.root / handoff.REQUEST).exists())
        self.assertFalse(any(c[0] in ("qsub", "qdel") for c in self.h.calls()))

    def test_lost_ssh_reply_uses_released_lock_as_authority(self):
        old = self.h.start_old()
        target = handoff.owner(self.root)
        target["value"]["host"] = "login1"
        write(self.root / target["record"], target["value"])

        def lost_reply(root, request, timeout):
            # The remote stop completed, but its transport reply was lost.
            old.terminate()
            old.wait(timeout=5)
            raise subprocess.TimeoutExpired("ssh", timeout)

        with lock(self.root / "launch.lock"), lock(self.root / "control.lock"):
            with (
                patch.object(socket, "gethostname", return_value="login2"),
                patch.object(handoff, "remote_signal", side_effect=lost_reply),
            ):
                lease = repair.stop_and_acquire(self.root, timeout=1)
                self.assertTrue(active(self.root / "guardian.lock"))
                lease.close()
        self.assertFalse((self.root / handoff.REQUEST).exists())

    def test_cooperative_requests_require_exact_owner_live_coordinator_and_freshness(
        self,
    ):
        runtime = dict(
            host="login1", pid=1234, incarnation="first", handoff_policy=handoff.POLICY
        )
        write(self.root / "guardian_runtime.json", runtime)
        with lock(self.root / "launch.lock"), lock(self.root / "control.lock"):
            request, lease = handoff.create(self.root, handoff.owner(self.root), 10)
            try:
                self.assertTrue(handoff.relinquish(self.root, runtime))
                self.assertFalse(
                    handoff.relinquish(self.root, dict(runtime, incarnation="second"))
                )
                expired = dict(request, expires_epoch=0)
                write(self.root / handoff.REQUEST, expired)
                self.assertFalse(handoff.relinquish(self.root, runtime))
                write(self.root / handoff.REQUEST, request)
                lease.close()  # Simulate a dead coordinator with a leftover mailbox.
                self.assertFalse(handoff.relinquish(self.root, runtime))
            finally:
                lease.close()

    def test_helper_rejects_wrong_host_changed_request_or_wrong_process_owner(self):
        old = self.h.start_old()
        with lock(self.root / "launch.lock"), lock(self.root / "control.lock"):
            request, lease = handoff.create(self.root, handoff.owner(self.root), 10)
            try:
                with patch.object(socket, "gethostname", return_value="wrong-host"):
                    with self.assertRaisesRegex(RuntimeError, "wrong host"):
                        repair.signal_local_owner(self.root, request["owner"])
                write(self.root / handoff.REQUEST, dict(request, nonce="0" * 32))
                with self.assertRaisesRegex(RuntimeError, "replaced or cancelled"):
                    handoff.check(self.root, request)
                write(self.root / handoff.REQUEST, request)
                identity = read(self.h.root / "old_identity.json")
                with patch.object(
                    repair,
                    "_process_identity",
                    return_value=(os.getuid() + 1, identity["argv"]),
                ):
                    with self.assertRaisesRegex(RuntimeError, "owner mismatch"):
                        repair.signal_local_owner(self.root, request["owner"])
            finally:
                lease.close()
        self.assertIsNone(old.poll())

    def test_nonresponsive_cooperative_owner_is_never_force_signalled(self):
        old = self.h.start_old()
        record = handoff.owner(self.root)["value"]
        write(
            self.root / "guardian_runtime.json",
            dict(record, handoff_policy=handoff.POLICY, incarnation="unresponsive"),
        )
        with lock(self.root / "launch.lock"), lock(self.root / "control.lock"):
            with (
                patch.object(repair, "signal_local_owner") as signal,
                patch.object(handoff, "remote_signal") as remote,
            ):
                with self.assertRaisesRegex(RuntimeError, "handoff timed out"):
                    repair.stop_and_acquire(self.root, timeout=0.1)
                signal.assert_not_called()
                remote.assert_not_called()
        self.assertIsNone(old.poll())
        self.assertFalse((self.root / handoff.REQUEST).exists())
