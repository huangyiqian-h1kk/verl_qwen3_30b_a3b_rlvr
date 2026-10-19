"""CPU fixtures for scheduler ambiguity and non-mutating incident collection."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/multidomain/diagnose_jobs.py"
spec = importlib.util.spec_from_file_location("d4_diagnose_jobs", SCRIPT)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticsTests(unittest.TestCase):
    def test_pbs_json_with_prefix_and_flattened_resources(self):
        row = {"job_state": "F", "Exit_status": 271, "Resource_List.walltime": "24:00:00",
               "resources_used": {"walltime": "02:13:00"}}
        text = "Job id  Name\nwarning {not JSON}\n" + json.dumps({"Jobs": {"7.pbs1": row}})
        with patch.object(diag, "command", return_value={"returncode": 0, "stdout": text, "stderr": ""}):
            actual = diag.query_job(Path.cwd(), "7.pbs1")
        self.assertEqual(actual["Exit_status"], 271)
        self.assertEqual(actual["Resource_List"]["walltime"], "24:00:00")
        self.assertEqual(actual["resources_used"]["walltime"], "02:13:00")

    def test_unknown_malformed_and_wrong_job_are_not_success(self):
        responses = [
            {"returncode": 153, "stdout": "", "stderr": "Unknown Job Id"},
            {"returncode": 0, "stdout": "not json", "stderr": ""},
            {"returncode": 0, "stdout": '{"Jobs":{"7.other":{"Exit_status":0}}}', "stderr": ""},
            {"returncode": 0, "stdout": '{"Jobs":{}}\n{"Jobs":{}}', "stderr": ""},
        ]
        for response in responses:
            with self.subTest(response=response), patch.object(diag, "command", return_value=response):
                result = diag.query_job(Path.cwd(), "7.pbs1")
                self.assertIn("query_error", result)
                self.assertNotIn("Exit_status", result)

    def test_lock_observation_does_not_create_lock_and_detects_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "controller.lock"
            self.assertFalse(diag.lock_status(lock)["held"])
            self.assertFalse(lock.exists())
            with lock.open("w") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(diag.lock_status(lock)["held"])
            self.assertFalse(diag.lock_status(lock)["held"])

    def test_large_log_read_is_bounded_and_shows_error_before_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pbs.log"
            path.write_text("old\n" * 100000 + "Traceback (most recent call last):\nValueError: broken restore\n"
                            + "shutdown cleanup\n" * 100)
            evidence = diag.log_evidence(path, 8)
            self.assertEqual(len(evidence["lines"]), 8)
            self.assertTrue(evidence["truncated"])
            self.assertIn("ValueError: broken restore", evidence["error_context_from_final_256KiB"])

    def test_collection_retains_both_lineages_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root / "outputs/multidomain"
            def write(path, obj):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(obj))
            random = parent / (diag.BASE + "_n2_once_v1")
            domain = parent / (diag.BASE + "_db_v1")
            old = random / "jobs/20261019T142000_train"
            new = domain / "jobs/20261019T170000_train"
            write(random / "handoff/state.json", {
                "phase": "WAIT_READY", "controller_status": "RUNNING", "host": "another-login",
                "dual": {"job_id": "71.pbs1", "folder": str(old)},
                "handoff_blocked": "Candidate failed with exit 1"})
            write(domain / "monitor/state.json", {"job_id": "72.pbs1", "job_dir": str(new)})
            write(domain / "monitor/pending_submission.json", {"job_id": "73.pbs1"})
            for folder, jid, status in ((old, "71.pbs1", "FAIL"), (new, "72.pbs1", "RUNNING")):
                write(folder / "request.json", {"stage": "train"})
                write(folder / "result.json", {"status": status})
                (folder / "job_id.txt").write_text(jid + "\n")
                (folder / "pbs.log").write_text("ValueError: test failure\n")
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            calls = []
            def fake(args, cwd, timeout=15):
                calls.append(args)
                if args[:4] == ["qstat", "-xf", "-F", "json"]:
                    jid = args[4]
                    value = {"job_state": "R"} if jid == "72.pbs1" else {"job_state": "F", "Exit_status": 1}
                    return {"returncode": 0, "stdout": json.dumps({"Jobs": {jid: value}}), "stderr": ""}
                self.assertIn(args[0], ("qstat", "git"))
                return {"returncode": 0, "stdout": "", "stderr": ""}
            with patch.object(diag, "command", side_effect=fake):
                report = diag.collect(root, ("70.pbs1",))
            self.assertEqual(set(report["jobs"]), {"70.pbs1", "71.pbs1", "72.pbs1", "73.pbs1"})
            self.assertEqual(report["jobs"]["71.pbs1"]["pbs"]["Exit_status"], 1)
            self.assertEqual(report["jobs"]["72.pbs1"]["pbs"]["job_state"], "R")
            self.assertFalse(report["controllers"]["random_dual"]["lock"]["held"])
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertTrue(all(c[0] in ("git", "qstat") for c in calls))

    def test_stdin_execution_needs_no_repo_imports_or_new_files(self):
        with tempfile.TemporaryDirectory() as temp:
            # Do not run real scheduler commands; verify the standalone CLI via help.
            result = subprocess.run(["python3", "-I", "-", "--help"], input=SCRIPT.read_text(),
                                    cwd=temp, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--job", result.stdout)
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
