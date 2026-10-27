"""Real frozen-owner replacement; PBS executables and checkpoint payloads are fixtures."""

import getpass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import test_domain_teachers as fixtures
from d4_teachers import controller, retry_repair as repair, source
from d4_teachers.common import active, digest, file_sha, files, read, write
from d4_training_continuation.scheduler import PBS
from test_training_continuation import checkpoint


class RetryRepair(fixtures.TestCase):
    def setUp(self):
        super().setUp()
        self.c = fixtures.FrozenDeployment.fixture(self)
        self.campaign = Path(self.c["path"])
        self.children = []
        self.addCleanup(self.stop_children)
        # Restore the exact initial controller/CLI, including the retry-cap bug.
        old = Path(__file__).parent / "fixtures/domain_teachers"
        for name in ("controller", "cli"):
            target = self.campaign / "template/runtime/d4_teachers" / (name + ".py")
            shutil.copy2(old / (name + "_before.py"), target)
            self.assertEqual(file_sha(target), repair.ORIGINAL_SOURCES[name + ".py"])
        (self.campaign / "template/runtime/d4_teachers/retry_repair.py").unlink()
        self.c["policy"].update(max_no_progress_retries=3, poll_seconds=5)
        self.c["contract"]["template_files"] = files(self.campaign / "template")
        self.c["identity"] = digest(self.c["contract"])
        write(self.campaign / "campaign.json", self.c)
        write(
            self.campaign / "preflight.json",
            dict(status="PASS", campaign_identity=self.c["identity"]),
        )
        self.jobs, rows = {}, {}
        for i, domain in enumerate(controller.TASKS[:-1], 1):
            run = self.campaign / "runs" / domain
            run_info = read(run / "run.json")
            run_info["campaign_identity"] = self.c["identity"]
            write(run / "run.json", run_info)
            job = dict(source.make_job(self.c, domain, "train"), job_id=f"{i}.pbs1")
            self.jobs[domain] = job
            state = read(run / "state.json")
            state.update(status="RUNNING", job=job)
            row = dict(
                Job_Owner=getpass.getuser() + "@login",
                queue="R9920261000",
                Job_Name=job["job_name"],
                Submit_arguments=str(Path(job["folder"]) / "stage.pbs"),
                job_state="R",
            )
            if domain == "conversational_pivot":
                state.update(
                    status="BLOCKED_APPLICATION", error="verifier fixture failure"
                )
                row.update(job_state="F", Exit_status=1)
            if domain == "logic_algorithmic":
                state.update(
                    status="BLOCKED_RETRIES",
                    job=None,
                    previous_job=job,
                    no_progress_retries=3,
                    last_progress=1000000,
                    error="old qstat timeout",
                )
                row.update(
                    job_state="F", Exit_status=271, comment="terminated by root@login1"
                )
                checkpoint(
                    run / "checkpoints/global_step_10", 10, lineage=run_info["lineage"]
                )
                write(
                    run / "progress.json", dict(step=12)
                )  # Unsaved work is not resumable.
            write(run / "state.json", state)
            rows[job["job_id"]] = row
        write(self.root / "pbs_rows.json", rows)
        queue = (
            "Queue: R9920261000\n    queue_type = Execution\n    enabled = True\n    started = True\n"
            "    default_chunk.node_group = nii\n    acl_group_enable = True\n    acl_groups = gcg51557\n"
            "    resources_available.ncpus = 1920"
        )
        program = (
            "import json, os, sys\nfrom pathlib import Path\n"
            "root=Path(os.environ['TEACHER_TEST_PBS'])\nname=Path(sys.argv[0]).name\n"
            "with (root/'pbs_calls.jsonl').open('a') as f: f.write(json.dumps([name]+sys.argv[1:])+'\\n')\n"
            "rows=json.loads((root/'pbs_rows.json').read_text())\n"
            "if name=='qstat':\n"
            f"    print({queue!r} if '-Qf' in sys.argv else json.dumps({{'Jobs':rows}}))\n"
            "elif name=='qsub':\n"
            "    request=json.loads((Path(sys.argv[1]).parent/'request.json').read_text())\n"
            "    jid=str(900000+len(rows))+'.pbs1'\n"
            "    rows[jid]=dict(Job_Owner=os.environ['TEACHER_TEST_USER']+'@login',queue='R9920261000',\n"
            "                   Job_Name=request['job_name'],Submit_arguments=sys.argv[1],job_state='Q')\n"
            "    (root/'pbs_rows.json').write_text(json.dumps(rows))\n"
            "    print(jid)\n"
            "else: raise RuntimeError('Unexpected PBS mutation: '+name)\n"
        )
        binary = self.root / "fake_bin"
        binary.mkdir()
        for name in ("qstat", "qsub", "qdel", "pbsnodes"):
            path = binary / name
            path.write_text("#!" + sys.executable + "\n" + program)
            path.chmod(0o700)
        env = patch.dict(
            os.environ,
            dict(
                PATH=str(binary) + os.pathsep + os.environ.get("PATH", ""),
                TEACHER_TEST_PBS=str(self.root),
                TEACHER_TEST_USER=getpass.getuser(),
            ),
        )
        env.start()
        self.addCleanup(env.stop)

    def stop_children(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)

    def state(self):
        return read(self.campaign / "runs/logic_algorithmic/state.json")

    def change_row(self, **changes):
        rows = read(self.root / "pbs_rows.json")
        rows[self.jobs["logic_algorithmic"]["job_id"]].update(changes)
        write(self.root / "pbs_rows.json", rows)

    def wait_for(self, predicate):
        until = time.monotonic() + 12
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(0.05)
        log = self.campaign / "guardian.log"
        self.fail(log.read_text() if log.exists() else "No guardian log")

    def start_old(self):
        # This executor's /proc can expose host PIDs. Obtain identity from the
        # child's own /proc/self; pidfd signalling and flock remain real.
        entry = self.campaign / "control.py"
        original = entry.read_text()
        instrumented = (
            "import os, json\nfrom pathlib import Path\n"
            f"Path({str(self.root / 'old_identity.json')!r}).write_text(json.dumps(dict(uid=os.getuid(), "
            "argv=[os.fsdecode(x) for x in Path('/proc/self/cmdline').read_bytes().split(b'\\0') if x])))\n"
        ) + original
        entry.write_text(instrumented)
        allowed_entry = patch.object(repair, "ORIGINAL_ENTRY", instrumented)
        allowed_entry.start()
        self.addCleanup(allowed_entry.stop)
        with (self.campaign / "guardian.log").open("a") as stream:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    str(entry),
                    "guardian",
                    "--campaign",
                    str(self.campaign),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        self.children.append(child)
        self.wait_for(lambda: (self.campaign / "heartbeat.json").exists())
        identity = read(self.root / "old_identity.json")
        proc = patch.object(
            repair,
            "_process_identity",
            return_value=(identity["uid"], identity["argv"]),
        )
        proc.start()
        self.addCleanup(proc.stop)
        return child

    def test_recover_uses_full_checkpoint_and_is_repeatable(self):
        result = repair.recover_blocks(self.c, PBS())
        self.assertEqual(result["logic_algorithmic"]["status"], "REARMED")
        self.assertEqual(self.state()["last_progress"], 1000000)
        self.assertEqual(self.state()["no_progress_retries"], 3)
        self.assertNotIn("error", self.state())
        self.assertEqual(repair.recover_blocks(self.c, PBS()), {})
        self.assertFalse(any(x[0] in ("qsub", "qdel") for x in self.calls()))

    def calls(self):
        path = self.root / "pbs_calls.jsonl"
        return (
            [json.loads(s) for s in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def test_walltime_is_also_eligible(self):
        self.change_row(Exit_status=-29, comment="job walltime exceeded limit")
        self.assertEqual(
            repair.recover_blocks(self.c, PBS())["logic_algorithmic"]["status"],
            "REARMED",
        )

    def test_application_live_unknown_and_unconfirmed_walltime_stay_blocked(self):
        before = self.state()
        for changes in (
            dict(Exit_status=1),
            dict(Exit_status=-29, comment="killed"),
            dict(Exit_status=None),
            dict(job_state="R", Exit_status=271),
        ):
            with self.subTest(changes=changes):
                self.change_row(
                    job_state="F",
                    **{k: v for k, v in changes.items() if k != "job_state"},
                )
                if "job_state" in changes:
                    self.change_row(job_state=changes["job_state"])
                result = repair.recover_blocks(self.c, PBS())
                self.assertEqual(result["logic_algorithmic"]["status"], "LEFT_BLOCKED")
                self.assertEqual(self.state(), before)

    def test_fresh_pbs_identity_is_required(self):
        self.change_row(Job_Owner="different-user@login")
        result = repair.recover_blocks(self.c, PBS())
        self.assertIn("owner mismatch", result["logic_algorithmic"]["reason"])
        self.assertEqual(self.state()["status"], "BLOCKED_RETRIES")

    def test_missing_pbs_record_and_ambiguous_intent_stay_blocked(self):
        before = self.state()
        rows = read(self.root / "pbs_rows.json")
        del rows[self.jobs["logic_algorithmic"]["job_id"]]
        write(self.root / "pbs_rows.json", rows)
        result = repair.recover_blocks(self.c, PBS())
        self.assertIn("omitted registered job", result["logic_algorithmic"]["reason"])
        self.assertEqual(self.state(), before)
        before["pending"] = dict(folder="unresolved submission")
        write(controller.state_path(self.c, "logic_algorithmic"), before)
        self.assertIn(
            "Job/intent still owned",
            repair.recover_blocks(self.c, PBS())["logic_algorithmic"]["reason"],
        )

    def test_foreign_job_contract_stays_blocked(self):
        folder = Path(self.jobs["logic_algorithmic"]["folder"])
        request = read(folder / "request.json")
        request["contract"]["campaign_identity"] = "another-campaign"
        request["campaign_identity"] = "another-campaign"
        request["identity"] = digest(request["contract"])
        write(folder / "request.json", request)
        self.assertIn(
            "another run",
            repair.recover_blocks(self.c, PBS())["logic_algorithmic"]["reason"],
        )
        self.assertEqual(self.calls(), [])

    def test_explicit_stop_fence_wins(self):
        write(self.campaign / "runs/logic_algorithmic/STOP.json", {})
        self.assertEqual(
            repair.recover_blocks(self.c, PBS())["logic_algorithmic"]["status"],
            "LEFT_BLOCKED",
        )
        write(self.campaign / "STOP.json", {})
        with self.assertRaisesRegex(RuntimeError, "Campaign stopped"):
            repair.repair(self.campaign, wait_seconds=0)
        self.assertFalse((self.campaign / "controller_revision.json").exists())

    def test_corrupt_training_snapshot_fails_before_signalling(self):
        (self.campaign / "template/runtime/d4_teachers/sampling.py").write_text(
            "# changed"
        )
        with patch.object(repair, "stop_and_acquire") as stop:
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                repair.repair(self.campaign, wait_seconds=0)
            stop.assert_not_called()

    def test_remote_guardian_is_not_signalled(self):
        child = self.start_old()
        heartbeat = read(self.campaign / "heartbeat.json")
        heartbeat["host"] = "another-login-host"
        write(self.campaign / "heartbeat.json", heartbeat)
        with self.assertRaisesRegex(RuntimeError, "Run repair on guardian host"):
            repair.repair(self.campaign, wait_seconds=0)
        self.assertIsNone(child.poll())
        self.assertFalse((self.campaign / "controller_revision.json").exists())

    def test_wrong_command_is_not_signalled(self):
        child = self.start_old()
        with patch.object(
            repair,
            "_process_identity",
            return_value=(os.getuid(), [sys.executable, "other.py"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "command line mismatch"):
                repair.repair(self.campaign, wait_seconds=0)
        self.assertIsNone(child.poll())

    def handoff(self, native=True, interrupt=False, legacy_resume=False):
        old = self.start_old()
        preserved = {
            "campaign": file_sha(self.campaign / "campaign.json"),
            "template": files(self.campaign / "template"),
            "data": files(self.campaign / "data"),
            "checkpoints": files(self.campaign / "runs/logic_algorithmic/checkpoints"),
            "requests": {
                d: file_sha(Path(j["folder"]) / "request.json")
                for d, j in self.jobs.items()
            },
        }
        popen = subprocess.Popen

        def record(*args, **kwargs):
            child = popen(*args, **kwargs)
            self.children.append(child)
            return child

        with patch.object(subprocess, "Popen", side_effect=record):
            if not native:
                with (
                    patch.object(os, "pidfd_open", None, create=True),
                    patch.object(signal, "pidfd_send_signal", None, create=True),
                ):
                    result = repair.repair(self.campaign, wait_seconds=10)
            elif interrupt:
                with patch.object(
                    repair,
                    "recover_blocks",
                    side_effect=RuntimeError("interrupted after publication"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "interrupted after publication"
                    ):
                        repair.repair(self.campaign, wait_seconds=0)
                self.assertEqual(old.wait(timeout=5), -signal.SIGTERM)
                result = repair.repair(self.campaign, wait_seconds=10)
            else:
                result = repair.repair(self.campaign, wait_seconds=10)
            self.assertEqual(old.wait(timeout=5), -signal.SIGTERM)
            self.assertEqual(result["status"], "REPAIRED_GUARDIAN_RUNNING", result)
            self.assertEqual(
                result["recovered"]["logic_algorithmic"]["status"], "REARMED"
            )
            self.assertEqual(self.state()["status"], "QUEUED")
            for d in ("math", "if"):
                self.assertEqual(
                    read(controller.state_path(self.c, d))["job"], self.jobs[d]
                )
            self.assertEqual(
                read(controller.state_path(self.c, "conversational_pivot"))["status"],
                "BLOCKED_APPLICATION",
            )
            pid = result["controller_runtime"]["runtime"]["pid"]
            repeated = repair.repair(self.campaign, wait_seconds=2)
            self.assertEqual(repeated["controller_runtime"]["runtime"]["pid"], pid)
            self.assertEqual(repeated["recovered"], {})
            if legacy_resume:
                current = next(child for child in self.children if child.pid == pid)
                current.terminate()
                current.wait(timeout=5)
                spec = importlib.util.spec_from_file_location(
                    "d4_teachers._legacy_cli",
                    Path(__file__).parent / "fixtures/domain_teachers/cli_before.py",
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.start(self.campaign, True)
                revision = read(self.campaign / "controller_revision.json")
                self.wait_for(
                    lambda: repair.runtime_status(self.c, revision)[
                        "first_tick_completed"
                    ]
                )
                self.assertNotEqual(
                    repair.runtime_status(self.c, revision)["runtime"]["pid"], pid
                )
        self.assertEqual([x[0] for x in self.calls()].count("qsub"), 1)
        self.assertFalse(any(x[0] in ("qdel", "pbsnodes") for x in self.calls()))
        self.assertEqual(
            file_sha(self.campaign / "campaign.json"), preserved["campaign"]
        )
        self.assertEqual(files(self.campaign / "template"), preserved["template"])
        self.assertEqual(files(self.campaign / "data"), preserved["data"])
        self.assertEqual(
            files(self.campaign / "runs/logic_algorithmic/checkpoints"),
            preserved["checkpoints"],
        )
        self.assertEqual(
            {
                d: file_sha(Path(j["folder"]) / "request.json")
                for d, j in self.jobs.items()
            },
            preserved["requests"],
        )
        self.assertTrue(active(self.campaign / "guardian.lock"))

    def test_real_old_owner_handoff_no_duplicate_jobs(self):
        self.handoff()

    def test_real_owner_handoff_without_python_pidfd_wrappers(self):
        self.handoff(native=False)

    def test_interrupted_publication_recovers_on_same_command(self):
        self.handoff(interrupt=True)

    def test_old_start_resume_entry_cannot_resurrect_retry_cap(self):
        self.handoff(legacy_resume=True)

    def test_launcher_rejects_modified_controller_snapshot(self):
        revision = repair.prepare_revision(self.c)
        write(self.campaign / "controller_revision.json", revision)
        (self.campaign / "control.py").write_text(repair.ENTRY)
        target = (
            self.campaign
            / "controller_revisions"
            / revision["identity"]
            / "runtime/d4_teachers/controller.py"
        )
        target.write_text("# tampered")
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(self.campaign / "control.py"),
                "status",
                "--campaign",
                str(self.campaign),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Controller source changed", result.stderr)
        self.assertEqual(self.calls(), [])
