"""Logic30 reuses frozen test rows and the real guardian; GPU/PBS are synthetic."""

import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_domain_teachers_deepmath_test_only as fixtures
from test_training_continuation import checkpoint
from d4_teachers import controller, deepmath_data, deepmath_evaluation as deep
from d4_teachers import evaluation, selection, source
from d4_teachers.common import files, read, write

TASK = "deepmath_logic30"


class LogicSupplement(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestOnlySupplements(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.h, self.c, self.root, self.pbs = (
            self.fixture.h,
            self.fixture.c,
            self.fixture.root,
            self.fixture.pbs,
        )
        run = self.root / "runs/logic_algorithmic"
        cp = checkpoint(
            run / "checkpoints/global_step_30",
            30,
            lineage=read(run / "run.json")["lineage"],
        )
        selection.select(
            dict(
                role="train",
                run_dir=str(run),
                domain="logic_algorithmic",
                lineage=cp["lineage"],
                verifier_identity=self.c["contract"]["verifier_identity"],
            ),
            cp,
            dict(
                step=30,
                split="validation",
                scores={"logic_algorithmic": {"mean": 0.75}},
                path="fixture",
                identity="fixture",
            ),
        )
        self.h.export("logic_algorithmic")
        deep.install(self.c, test_only=True)
        for task in controller.tasks(self.c):
            self.h.change(task, status="DONE", phase="test", job=None, pending=None)

    def test_registration_preserves_completed_campaign_and_pins_only_logic30_test(self):
        before = {
            name: files(self.root / name) for name in ("runs", "data", "template")
        }
        plans = {n: (self.root / n).read_bytes() for n in (deep.PLAN, deep.TEST_PLAN)}
        deep.install(self.c, logic30_test=True)
        for name, snapshot in before.items():
            actual = files(self.root / name)
            if name == "runs":
                actual = {
                    k: v for k, v in actual.items() if not k.startswith(TASK + "/")
                }
            self.assertEqual(actual, snapshot)
        self.assertEqual(plans, {n: (self.root / n).read_bytes() for n in plans})
        req = self.fixture.request(TASK)
        self.assertEqual(req["job_name"], "0390_dm_la30")
        self.assertEqual(req["evaluation_model"]["step"], 30)
        self.assertEqual(
            req["evaluation_model"]["path"],
            read(self.root / "runs/logic_algorithmic/export.json")["path"],
        )
        manifest = deepmath_data.verify(self.c)
        paths, ids = deep.files_and_ids(self.c, TASK)
        self.assertEqual(paths, [str(deepmath_data.directory(self.c) / "test.parquet")])
        self.assertEqual(ids, manifest["splits"]["test"]["ids"])
        self.assertFalse(set(ids) & set(manifest["splits"]["validation"]["ids"]))
        for role in ("train", "export"):
            with self.assertRaisesRegex(RuntimeError, "evaluation only"):
                source.make_job(self.c, TASK, role)
        self.h.tick()
        state = controller.state_path(self.c, TASK).read_bytes()
        self.assertEqual(
            deep.install(self.c, logic30_test=True)["status"], "ALREADY_REGISTERED"
        )
        self.assertEqual(controller.state_path(self.c, TASK).read_bytes(), state)
        self.h.tick()
        self.assertEqual(len(self.pbs.submissions), 1)

    def test_interrupted_validator_resumes_only_missing_batches_and_reports_levels(
        self,
    ):
        deep.install(self.c, logic30_test=True)
        with patch.object(
            evaluation, "dependency_versions", return_value={"fixture": "1"}
        ), patch("builtins.print"), patch.object(selection, "select") as select:
            first = self.fixture.runner(self.fixture.request(TASK), fail=8)
            with self.assertRaisesRegex(RuntimeError, "walltime"):
                evaluation.validate(first, first._validate)
            self.assertEqual(controller.progress(self.c, TASK, "test"), 8)
            resumed = self.fixture.runner(self.fixture.request(TASK))
            evaluation.validate(resumed, resumed._validate)
            receipt, rows, _ = deep.validate_result(self.c, TASK)
            self.assertEqual(resumed.generated, [8])
            self.assertEqual(receipt["journal"]["reused_batches"], 1)
            self.assertEqual(set(receipt["split_scores"]), {"test"})
            self.assertEqual(
                receipt["split_scores"]["test"]["math"], {"n": 16, "mean": 1.0}
            )
            self.assertEqual({r["difficulty_level"] for r in rows}, {8, 9})
            controller.artifact(self.c, TASK, "test")
            again = self.fixture.runner(self.fixture.request(TASK))
            evaluation.validate(again, again._validate)
            self.assertFalse(again.generated)
            select.assert_not_called()
        report = deep.report(self.root, logic30_test=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(set(report["models"]), {TASK})
        self.assertTrue(
            (self.root / "deepmath_evaluation/logic30_comparison.json").is_file()
        )
        for name in ("test.json", "best.json", "validation_state.json"):
            self.assertFalse((self.root / "runs" / TASK / name).exists())

    def test_271_and_confirmed_root265_retry_but_unknown_and_plain265_do_not(self):
        deep.install(self.c, logic30_test=True)
        state = self.h.tick()[TASK]
        req = source.validate_request(Path(state["job"]["folder"]) / "request.json")
        cached = self.root / "runs" / TASK / "evaluations/heldout_partial"
        write(cached / "progress.json", {"completed_rows": 8})
        write(cached / "batch_000001.json", {"kept": True})
        before = files(cached)
        for code in (271, 265, 271, 265):
            old = state["job"]["job_id"]
            self.pbs.rows[old] = dict(
                job_state="F", Exit_status=code, comment="terminated by root@login-fixture"
            )
            state = self.h.tick()[TASK]
            self.assertEqual(state["status"], "QUEUED")
            self.assertNotEqual(state["job"]["job_id"], old)
            newer = source.validate_request(
                Path(state["job"]["folder"]) / "request.json"
            )
            for key in (
                "evaluation_model",
                "deepmath_evaluation",
                "heldout_manifest",
                "verifier_identity",
            ):
                self.assertEqual(newer[key], req[key])
        self.assertEqual(files(cached), before)
        count = len(self.pbs.submissions)
        self.pbs.unknown = True
        self.assertEqual(self.h.tick()[TASK]["status"], "BLOCKED_UNKNOWN")
        self.assertEqual(len(self.pbs.submissions), count)
        self.pbs.unknown = False
        self.pbs.rows[state["job"]["job_id"]] = dict(job_state="F", Exit_status=265)
        self.assertEqual(self.h.tick()[TASK]["status"], "BLOCKED_APPLICATION")
        self.assertEqual(len(self.pbs.submissions), count)
        self.assertFalse(self.pbs.cancellations)

    def test_wrong_step_or_changed_weights_rejected_before_submission(self):
        path = self.root / "runs/logic_algorithmic/export.json"
        saved = read(path)
        changed = read(path)
        changed["selected"]["step"] = 60
        write(path, changed)
        with patch("d4_teachers.retry_repair.repair") as repair:
            with self.assertRaisesRegex(RuntimeError, "checkpoint unavailable"):
                deep.launch(self.root, logic30_test=True)
            repair.assert_not_called()
        write(path, saved)
        deep.install(self.c, logic30_test=True)
        (Path(saved["path"]) / "weights.safetensors").write_bytes(b"changed model")
        with self.assertRaisesRegex(RuntimeError, "model changed"):
            self.fixture.request(TASK)
        self.assertFalse(self.pbs.submissions)

    def test_inspector_and_cli_show_logic_test_without_validation(self):
        deep.install(self.c, logic30_test=True)
        repo = Path(source.__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(repo / "scripts/evaluate_teacher_deepmath.py"),
                "status",
                "--campaign",
                str(self.root),
                "--logic30-test",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(json.loads(result.stdout)["tasks"]), {TASK})
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(repo / "scripts/inspect_domain_teachers.py"),
                "--campaign",
                str(self.root),
                "--no-pbs",
                "--logs",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(TASK + " | test |", result.stdout)
        self.assertNotIn(TASK + " | validation |", result.stdout)
        self.assertFalse(self.pbs.submissions)

    def test_real_guardian_restarts_finished_campaign_and_repeat_start_is_idempotent(
        self,
    ):
        from d4_training_continuation import QUEUE
        from d4_training_continuation.capacity import NODE_GROUP, PROJECT

        write(
            self.root / "preflight.json",
            dict(status="PASS", campaign_identity=self.c["identity"]),
        )
        area = self.root.parent / "logic-pbs"
        binary = area / "bin"
        binary.mkdir(parents=True)
        write(area / "rows.json", {})
        (area / "queue.txt").write_text(
            f"Queue: {QUEUE}\n    queue_type = Execution\n    enabled = True\n"
            f"    started = True\n    default_chunk.node_group = {NODE_GROUP}\n"
            f"    acl_group_enable = True\n    acl_groups = {PROJECT}\n"
            "    resources_available.ncpus = 1920"
        )
        program = """import json, os, sys
from pathlib import Path
root = Path(os.environ["LOGIC_TEST_PBS"])
rows = json.loads((root / "rows.json").read_text())
name = Path(sys.argv[0]).name
with (root / "calls.jsonl").open("a") as f:
    f.write(json.dumps([name] + sys.argv[1:]) + "\\n")
if name == "qstat":
    print((root / "queue.txt").read_text() if "-Qf" in sys.argv else json.dumps({"Jobs": rows}))
elif name == "qsub":
    request = json.loads((Path(sys.argv[1]).parent / "request.json").read_text())
    jid = str(900000 + len(rows)) + ".pbs1"
    rows[jid] = dict(Job_Owner=os.environ["LOGIC_TEST_USER"] + "@login", queue=os.environ["LOGIC_TEST_QUEUE"], Job_Name=request["job_name"], Submit_arguments=sys.argv[1], job_state="Q")
    (root / "rows.json").write_text(json.dumps(rows))
    print(jid)
else:
    raise RuntimeError("Unexpected scheduler mutation")
"""
        for name in ("qstat", "qsub", "qdel"):
            path = binary / name
            path.write_text("#!" + sys.executable + "\n" + program)
            path.chmod(0o700)
        children = []
        popen = subprocess.Popen

        def track(*a, **kw):
            child = popen(*a, **kw)
            children.append(child)
            return child

        def cleanup():
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)

        self.addCleanup(cleanup)
        with patch.dict(
            os.environ,
            dict(
                PATH=str(binary) + os.pathsep + os.environ["PATH"],
                LOGIC_TEST_PBS=str(area),
                LOGIC_TEST_USER=getpass.getuser(),
                LOGIC_TEST_QUEUE=QUEUE,
            ),
        ), patch.object(subprocess, "Popen", side_effect=track):
            result = deep.launch(self.root, logic30_test=True)
            self.assertTrue(
                result["guardian"]["controller_runtime"]["first_tick_completed"]
            )
            self.assertEqual(result["guardian"]["states"][TASK]["status"], "QUEUED")
            again = deep.launch(self.root, logic30_test=True)
            self.assertEqual(
                again["guardian"]["controller_runtime"]["runtime"]["pid"],
                result["guardian"]["controller_runtime"]["runtime"]["pid"],
            )
        calls = [
            json.loads(line) for line in (area / "calls.jsonl").read_text().splitlines()
        ]
        self.assertEqual(sum(c[0] == "qsub" for c in calls), 1)
        self.assertFalse(any(c[0] == "qdel" for c in calls))
