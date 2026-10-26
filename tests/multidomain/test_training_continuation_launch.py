"""Real isolated interpreter/entry/snapshot tests; PBS commands are local fixtures."""
import copy
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from d4_training_continuation import cli, cancellation_repair as repair
from d4_training_continuation.common import active, files, read, sha, write
from test_training_continuation import Harness
import test_training_continuation as fixtures


class LaunchTests(Harness):
    def setUp(self):
        super().setUp()
        self.children = []
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            else:
                child.wait()

    def wait_for(self, predicate, control, timeout=10):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(0.05)
        log = control / "controller.log"
        self.fail("Owner did not reach expected state: " + (log.read_text() if log.exists() else "no log"))

    def fixture(self):
        c = self.controller("random")
        plan = fixtures.IntegrationTests.template_plan(self, "random")
        operation = c.control / "operations/synthetic"
        original = operation / "template"
        shutil.move(plan["template"], original)
        # A small legacy-owner fixture reproduces the deployed old CLI's lack
        # of runtime binding. Its exact command, lease and SIGTERM are real.
        legacy = original / repair.CLI
        legacy.write_text(
            "import os, signal, socket, time\n"
            "from pathlib import Path\n"
            "from .common import locked, read, write\n"
            "def main():\n"
            f"    control=Path({str(c.control)!r})\n"
            "    with locked(control/'controller.lock'):\n"
            "        state=read(control/'state.json')\n"
            "        state.update(pid=os.getpid(), host=socket.gethostname(), controller_status='RUNNING',\n"
            "                     last_error='RuntimeError: Unknown PBS cancellation state')\n"
            "        state.pop('runtime_binding', None)\n"
            "        stop=[False]\n"
            "        signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))\n"
            "        write(control/'state.json', state)\n"
            "        write(control/'legacy_identity.json', {'uid': os.getuid(), 'argv':\n"
            "            [os.fsdecode(x) for x in Path('/proc/self/cmdline').read_bytes().split(b'\\0') if x]})\n"
            "        while not stop[0]: time.sleep(0.01)\n"
            "        state['controller_status']='STOPPED'\n"
            "        write(control/'state.json', state)\n"
        )
        cli_pin = patch.object(repair, "ORIGINAL_CLI_SHAS", {sha(legacy)})
        cli_pin.start()
        self.addCleanup(cli_pin.stop)
        plan.update(template=str(original), template_files=files(original))
        planfile = operation / "plan.json"
        # Reproduce the first repair: plan already points to a new scheduler
        # snapshot, but controller_entry.py still imports operation/template.
        v1 = operation / "repairs/pbs-finished-without-exit-v1/template"
        shutil.copytree(original, v1)
        plan.update(template=str(v1), template_files=files(v1))
        write(planfile, plan)
        old = dict(job_id="99.pbs1", job_name="0390_old_candidate", folder=str(self.root / "old_candidate"))
        c.save(migration_complete=True, old_candidate=old, old_candidate_retired=False)
        c.p["old_control"].mkdir(parents=True)
        write(c.p["monitor"] / "state.json", {"status": "RUNNING"})
        rows = {}
        for job, state in ((c.state["active"], "R"), (old, "F")):
            rows[job["job_id"]] = dict(Job_Owner=getpass.getuser() + "@login", queue="R9920261000",
                Job_Name=job["job_name"], Submit_arguments=str(Path(job["folder"]) / "stage.pbs"),
                job_state=state, substate=91 if state == "F" else 42, Exit_status=None)
        write(self.root / "pbs_rows.json", rows)
        binary = self.root / "fake_bin"
        binary.mkdir()
        queue = ("Queue: R9920261000\n    queue_type = Execution\n    enabled = True\n    started = True\n"
                 "    default_chunk.node_group = nii\n    acl_group_enable = True\n    acl_groups = gcg51557\n"
                 "    resources_available.ncpus = 1920")
        fixture = (
            "import json, os, sys\nfrom pathlib import Path\n"
            "root=Path(os.environ['D4_TEST_PBS'])\nname=Path(sys.argv[0]).name\n"
            "with (root/'pbs_calls.jsonl').open('a') as out: out.write(json.dumps([name]+sys.argv[1:])+'\\n')\n"
            "if name=='qstat':\n"
            "    if '-Qf' in sys.argv:\n"
            f"        print({queue!r})\n"
            "    else: print(json.dumps({'Jobs':json.loads((root/'pbs_rows.json').read_text())}))\n"
            "elif name=='qsub': print('900001.pbs1')\n"
            "else: raise RuntimeError('Unexpected PBS mutation: '+name)\n"
        )
        for name in ("qstat", "qsub", "qdel", "pbsnodes"):
            file = binary / name
            file.write_text("#!" + sys.executable + "\n" + fixture)
            file.chmod(0o700)
        env = patch.dict(os.environ, {"PATH": str(binary) + os.pathsep + os.environ.get("PATH", ""),
                                     "D4_TEST_PBS": str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        # The fixture owns the only PBS command paths visible to new children.
        entry = operation / "controller_entry.py"
        entry.write_text("import sys\n" + f"sys.path.insert(0, {str(original / 'runtime')!r})\n"
                         "from d4_training_continuation.cli import main\nmain()\n")
        return c, planfile, original, entry

    def run_repair(self, already_current):
        c, planfile, original, entry = self.fixture()
        domain = self.controller("domain")
        domain.fail_candidate("Candidate ended before handoff commit: 271")
        domain_before = (domain.control / "state.json").read_bytes()
        before = read(c.control / "state.json")
        original_files = files(original)
        if already_current:
            _, _, revised = repair.prepare_template(self.root, "random")
            write(planfile, revised)
        # Start the actual stale entry in python -I, with the already updated plan.
        with (c.control / "controller.log").open("a") as out:
            child = subprocess.Popen([sys.executable, "-I", str(entry), "_run", "--root", str(self.root),
                                      "--kind", "random"], stdout=out, stderr=subprocess.STDOUT)
        self.children.append(child)
        self.wait_for(lambda: (c.control / "legacy_identity.json").exists(), c.control)
        identity = read(c.control / "legacy_identity.json")
        self.assertFalse(cli.runtime_status(self.root, "random")["verified"])
        real_spawn = cli.spawn

        def recording_spawn(root, kind):
            process = real_spawn(root, kind)  # Real entry, import, daemon and PBS adapter.
            if process is not None:
                self.children.append(process)
            return process

        # Only /proc identity is supplied by the child's own /proc/self because
        # this test executor can mount /proc in a different PID namespace.
        # pidfd, SIGTERM, leases, Popen, imports and scheduler calls remain real.
        with patch.object(repair, "_process_identity", return_value=(identity["uid"], identity["argv"])), \
             patch.object(cli, "spawn", side_effect=recording_spawn):
            result = repair.repair(self.root, "random")
        self.assertEqual(child.wait(timeout=3), 0)
        self.assertEqual(result["template_updated"], not already_current)
        self.assertTrue(result["controller_restart_requested"])
        self.assertEqual(result["status"], "REPAIRED_CONTROLLER_RUNNING")
        self.assertTrue(result["controller_runtime"]["verified"])
        self.assertNotEqual(result["controller_runtime"]["pid"], child.pid)
        self.wait_for(lambda: bool(read(c.control / "state.json").get("candidate"))
                      and read(c.control / "state.json").get("last_error") is None, c.control)
        state = read(c.control / "state.json")
        self.assertTrue(state["old_candidate_retired"])
        self.assertFalse(state["promoted"])
        self.assertFalse(state["handoff_failed"])
        self.assertEqual(state["active"], before["active"])
        self.assertEqual(state["deadline_utc"], before["deadline_utc"])
        request = read(Path(state["candidate"]["folder"]) / "request.json")
        self.assertEqual(request["expected_nodes"], 2)
        self.assertEqual(request["role"], "candidate")
        self.assertEqual(request["config"]["training"]["train_batch_size"], 256)
        current = read(planfile)
        for module in ("cli", "scheduler", "controller"):
            path = state["runtime_binding"]["modules"]["d4_training_continuation." + module]["path"]
            self.assertEqual(path, str(Path(current["template"]) / "runtime/d4_training_continuation" / (module + ".py")))
        again = repair.repair(self.root, "random")
        self.assertFalse(again["controller_restart_requested"])
        self.assertEqual(again["controller_runtime"]["pid"], state["pid"])
        calls = [json.loads(line) for line in (self.root / "pbs_calls.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row[0] == "qsub" for row in calls), 1)
        self.assertFalse(any(row[0] in ("qdel", "pbsnodes") for row in calls))
        self.assertEqual(files(original), original_files)
        self.assertEqual((domain.control / "state.json").read_bytes(), domain_before)
        return c, current

    def test_real_entry_repair_loads_new_snapshot_and_queues_one_candidate(self):
        self.run_repair(already_current=False)

    def test_already_updated_plan_still_restarts_unbound_old_process(self):
        self.run_repair(already_current=True)

    def test_wrong_import_root_is_rejected_before_any_pbs_action(self):
        c, planfile, original, entry = self.fixture()
        _, _, revised = repair.prepare_template(self.root, "random")
        write(planfile, revised)
        # Deliberately run the fixed daemon from a different snapshot than plan.
        # It must reject that entry before touching PBS or claiming ownership.
        shutil.copy2(ROOT / "d4_training_continuation/cli.py", original / repair.CLI)
        before = (c.control / "state.json").read_bytes()
        result = subprocess.run([sys.executable, "-I", str(entry), "_run", "--root", str(self.root),
                                 "--kind", "random"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Path escapes owned directory", result.stderr)
        self.assertFalse((self.root / "pbs_calls.jsonl").exists())
        self.assertEqual((c.control / "state.json").read_bytes(), before)
        self.assertFalse(active(c.control / "controller.lock"))

    def test_runtime_receipt_rejects_stale_owner_plan_hash_and_stopped_process(self):
        c, _ = self.run_repair(already_current=False)
        original = read(c.control / "state.json")
        cases = []
        for key, value in (("pid", -1), ("host", "other-login"), ("controller_status", "STOPPED")):
            state = copy.deepcopy(original)
            state[key] = value
            cases.append(state)
        for key, value in (("template", "/wrong"), ("template_digest", "wrong")):
            state = copy.deepcopy(original)
            state["runtime_binding"][key] = value
            cases.append(state)
        state = copy.deepcopy(original)
        state["runtime_binding"]["modules"]["d4_training_continuation.scheduler"]["sha256"] = "wrong"
        cases.append(state)
        for state in cases:
            self.assertFalse(cli.runtime_status(self.root, "random", state)["verified"])
        self.stop_children()
        self.assertFalse(active(c.control / "controller.lock"))
        self.assertFalse(cli.runtime_status(self.root, "random", original)["verified"])
