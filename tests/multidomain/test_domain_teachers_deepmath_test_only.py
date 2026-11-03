"""Test-only supplements use real snapshots/journals; PBS and GPU are synthetic."""

import json
import getpass
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq

import test_domain_teachers as fixtures
import test_domain_teachers_deepmath as deep_fixtures
import test_domain_teachers_followup as follow_fixtures
from d4_teachers import controller, deepmath_data, deepmath_evaluation as deep
from d4_teachers import evaluation, followup, runtime, selection, source
from d4_teachers.common import digest, files, read, write


class TestOnlySupplements(unittest.TestCase):
    def setUp(self):
        self.fixture = follow_fixtures.Followup(methodName="runTest")

        def prepare(obj):
            return deepmath_data.prepare(
                obj.c,
                obj.inputs,
                size=16,
                tokenizer=deep_fixtures.Tokenizer(),
                score_reference=lambda _: dict(status="ok", score=1.0),
            )

        with patch.object(deep_fixtures.DeepMathFixture, "prepare", prepare):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.h, self.c, self.root, self.pbs = (
            self.fixture.h,
            self.fixture.c,
            self.fixture.root,
            self.fixture.pbs,
        )
        followup.install(self.c)
        self.h.export(followup.IF)
        self.h.change(followup.IF, status="READY", phase="test")
        deep.install(self.c)
        for task in deep.TARGETS:
            self.h.change(task, status="DONE", phase="test")

    def request(self, task):
        job = source.make_job(self.c, task, "test")
        return source.validate_request(Path(job["folder"]) / "request.json")

    def runner(self, request, fail=None):
        rows = pq.read_table(
            deepmath_data.directory(self.c) / "test.parquet"
        ).to_pylist()
        obj = fixtures.EvaluationRecovery(methodName="runTest").validator(
            len(rows), fail=fail
        )
        obj.config = OmegaConf.create(json.loads(json.dumps(obj.config)))
        obj.config.actor_rollout_ref.rollout.val_kwargs.temperature = 0
        obj.config.data = dict(validation_shuffle=False, val_batch_size=8)
        obj.teacher_request = request
        obj.global_steps = request["evaluation_model"]["step"]

        class Rows:
            def __init__(self):
                self.rows = rows

            def __len__(self):
                return len(rows)

        obj.val_dataset = Rows()
        obj.current_checkpoint = lambda step: None
        obj.weights_identity = lambda cp: request["evaluation_model"][
            "weights_identity"
        ]
        obj._dump_generations = lambda **kwargs: None
        for i, batch in enumerate(obj.val_dataloader):
            batch["uid"] = np.array(
                [r["sample_id"] for r in rows[i * 8 : i * 8 + 8]], dtype=object
            )
        return obj

    def test_registration_preserves_existing_plans_models_jobs_and_four_job_cap(self):
        states = self.h.tick()
        existing = {
            t: states[t]["job"]
            for t in ("logic_algorithmic", followup.IF, followup.MATH)
        }
        before = {
            name: files(self.root / name)
            for name in (
                "template",
                "data",
                "runs/if_step70",
                "runs/cp_step80",
                "runs/math_continued",
            )
        }
        plans = {
            name: (self.root / name).read_bytes() for name in (deep.PLAN, followup.PLAN)
        }
        deep.install(self.c, test_only=True)
        states = self.h.tick()
        self.assertEqual(len(self.pbs.submissions), 4)
        self.assertEqual(states["deepmath_if70"]["status"], "QUEUED")
        self.assertEqual(states["deepmath_cp80"]["status"], "READY")
        for task, job in existing.items():
            self.assertEqual(states[task]["job"], job)
        for name, snapshot in before.items():
            self.assertEqual(
                {k: v for k, v in files(self.root / name).items() if k != "state.json"},
                {k: v for k, v in snapshot.items() if k != "state.json"},
            )
        self.assertEqual(
            plans, {name: (self.root / name).read_bytes() for name in plans}
        )
        state_bytes = {
            t: controller.state_path(self.c, t).read_bytes() for t in deep.TEST_TARGETS
        }
        self.assertEqual(
            deep.install(self.c, test_only=True)["status"], "ALREADY_REGISTERED"
        )
        self.assertEqual(
            state_bytes,
            {
                t: controller.state_path(self.c, t).read_bytes()
                for t in deep.TEST_TARGETS
            },
        )
        self.h.tick()
        self.assertEqual(len(self.pbs.submissions), 4)
        self.assertFalse(self.pbs.cancellations)

    def test_requests_and_runtime_use_exact_test_only_hf_models(self):
        deep.install(self.c, test_only=True)
        manifest = deepmath_data.verify(self.c)
        for task, (origin, step) in deep.TEST_TARGETS.items():
            with self.subTest(task=task):
                req = self.request(task)
                self.assertEqual(req["evaluation_model"]["step"], step)
                self.assertEqual(
                    req["evaluation_model"]["path"],
                    read(self.root / "runs" / origin / "export.json")["path"],
                )
                self.assertNotIn("teacher_followup", req)
                paths, ids = deep.files_and_ids(self.c, task)
                self.assertEqual(
                    paths, [str(deepmath_data.directory(self.c) / "test.parquet")]
                )
                self.assertEqual(ids, manifest["splits"]["test"]["ids"])
                self.assertFalse(
                    set(ids) & set(manifest["splits"]["validation"]["ids"])
                )
                folder = Path(req["job_dir"])
                subprocess.run(["bash", "-n", str(folder / "stage.pbs")], check=True)
                self.assertIn("walltime=24:00:00", (folder / "stage.pbs").read_text())
                child = subprocess.run(
                    [sys.executable, "-I", str(folder / "entry.py"), "--help"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(child.returncode, 0, child.stderr)
                for role in ("train", "export"):
                    with self.assertRaisesRegex(RuntimeError, "evaluation only"):
                        source.make_job(self.c, task, role)
                write(folder / "allocation.json", dict(execution_id="fake-allocation"))
                cfg = OmegaConf.create(
                    dict(
                        trainer={},
                        data={},
                        reward=dict(custom_reward_function={}),
                        ray_kwargs=dict(ray_init=dict(runtime_env=dict(env_vars={}))),
                    )
                )
                with (
                    patch.object(runtime, "allowed"),
                    patch.object(runtime, "fixed_recipe"),
                    patch("multidomain.runtime_config.compose", return_value=cfg),
                    patch("d4_training_continuation.optimizer_repair.configure"),
                    patch.object(runtime, "run_trainer") as run,
                    patch.dict(os.environ, {}),
                ):
                    runtime.phase(folder / "request.json")
                actual = run.call_args.args[0]
                self.assertTrue(actual.trainer.val_only)
                self.assertEqual(list(actual.data.val_files), paths)

    def test_actual_validator_recovers_batches_and_publishes_only_test_scores(self):
        deep.install(self.c, test_only=True)
        with (
            patch.object(
                evaluation, "dependency_versions", return_value={"fixture": "1"}
            ),
            patch("builtins.print"),
            patch.object(selection, "select") as select,
        ):
            for task in deep.TEST_TARGETS:
                with self.subTest(task=task):
                    req = self.request(task)
                    first = self.runner(req, fail=8)
                    with self.assertRaisesRegex(RuntimeError, "walltime"):
                        evaluation.validate(first, first._validate)
                    self.assertEqual(controller.progress(self.c, task, "test"), 8)
                    self.assertFalse((Path(req["run_dir"]) / "heldout.json").exists())
                    resumed = self.runner(self.request(task))
                    evaluation.validate(resumed, resumed._validate)
                    receipt, rows, _ = deep.validate_result(self.c, task)
                    self.assertEqual(resumed.generated, [8])
                    self.assertEqual(receipt["journal"]["reused_batches"], 1)
                    self.assertEqual(set(receipt["split_scores"]), {"test"})
                    self.assertEqual(
                        receipt["split_scores"]["test"]["math"], dict(n=16, mean=1.0)
                    )
                    self.assertTrue(all(r["evaluation_split"] == "test" for r in rows))
                    controller.artifact(self.c, task, "test")
                    again = self.runner(self.request(task))
                    evaluation.validate(again, again._validate)
                    self.assertFalse(again.generated)
                    for name in ("test.json", "best.json", "validation_state.json"):
                        self.assertFalse((Path(req["run_dir"]) / name).exists())
            select.assert_not_called()
        report = deep.report(self.root, test_only=True)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(r["split"] == "test" for r in report["paired_comparisons"]))

    def test_271_retries_and_unknown_observations_preserve_work(self):
        deep.install(self.c, test_only=True)
        for task in ("logic_algorithmic", followup.IF, followup.MATH):
            self.h.change(task, status="DONE", job=None)
        states = self.h.tick()
        for task in deep.TEST_TARGETS:
            initial = source.validate_request(
                Path(states[task]["job"]["folder"]) / "request.json"
            )
            cached = self.root / "runs" / task / "evaluations/heldout_progress"
            write(cached / "progress.json", dict(completed_rows=8))
            write(cached / "batch_000001.json", dict(cached="preserved"))
            before = files(cached)
            for _ in range(6):
                self.pbs.rows[states[task]["job"]["job_id"]] = dict(
                    job_state="F", Exit_status=271
                )
                states = self.h.tick()
                resumed = source.validate_request(
                    Path(states[task]["job"]["folder"]) / "request.json"
                )
                for key in (
                    "deepmath_evaluation",
                    "evaluation_model",
                    "heldout_manifest",
                    "verifier_identity",
                ):
                    self.assertEqual(resumed[key], initial[key])
            self.assertGreater(states[task]["no_progress_retries"], 3)
            self.assertEqual(files(cached), before)
        before = len(self.pbs.submissions)
        self.pbs.unknown = True
        states = self.h.tick()
        self.assertEqual(len(self.pbs.submissions), before)
        self.assertTrue(
            all(states[t]["status"] == "BLOCKED_UNKNOWN" for t in deep.TEST_TARGETS)
        )
        self.pbs.unknown = False
        self.h.tick()
        self.assertEqual(len(self.pbs.submissions), before)

    def test_crash_before_publication_never_registers_partial_tasks(self):
        original = deep.write

        def fail_plan(path, value):
            if Path(path).name == deep.TEST_PLAN:
                raise OSError("publication crash")
            return original(path, value)

        with patch.object(deep, "write", side_effect=fail_plan):
            with self.assertRaisesRegex(OSError, "publication crash"):
                deep.install(self.c, test_only=True)
        self.assertTrue(set(deep.TEST_TARGETS).isdisjoint(controller.tasks(self.c)))
        deep.install(self.c, test_only=True)
        self.assertTrue(set(deep.TEST_TARGETS).issubset(controller.tasks(self.c)))

    def test_missing_if70_fails_before_handoff_and_data_change_is_rejected(self):
        best = self.root / "runs/if_step70/best.json"
        saved = best.read_bytes()
        best.unlink()
        with patch("d4_teachers.retry_repair.repair") as repair:
            with self.assertRaises(FileNotFoundError):
                deep.launch(self.root, test_only=True)
        repair.assert_not_called()
        best.write_bytes(saved)
        deep.install(self.c, test_only=True)
        (deepmath_data.directory(self.c) / "test.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.request("deepmath_if70")

    def test_waiting_export_keeps_original_if_job_and_rejects_wrong_selection(self):
        # Model-only pins remain usable while the independently scheduled export finishes.
        from d4_teachers import checkpoint_evaluation

        record = read(self.root / "runs/if_step70/export.json")
        checkpoint_evaluation.pin_model(
            self.root / "runs/if_step70", followup.plan(self.c)["targets"][followup.IF]
        )
        (self.root / "runs/if_step70/export.json").unlink()
        deep.install(self.c, test_only=True)
        state = {}
        self.assertFalse(deep.ready(self.c, "deepmath_if70", state))
        self.assertEqual(state["status"], "WAITING_MODEL")
        write(self.root / "runs/if_step70/export.json", record)
        self.assertTrue(deep.ready(self.c, "deepmath_if70", state))
        p = deep.plan(self.c, test_only=True)
        p["targets"]["deepmath_if70"]["selected_identity"] = "wrong-weights"
        p["identity"] = digest({k: v for k, v in p.items() if k != "identity"})
        write(self.root / deep.TEST_PLAN, p)
        with self.assertRaisesRegex(RuntimeError, "selection changed"):
            self.request("deepmath_if70")

    def test_inspector_and_status_recognize_supplements_without_submitting(self):
        deep.install(self.c, test_only=True)
        repo = Path(source.__file__).resolve().parents[1]
        status = subprocess.run(
            [
                sys.executable,
                "-I",
                str(repo / "scripts/evaluate_teacher_deepmath.py"),
                "status",
                "--campaign",
                str(self.root),
                "--if70-cp80-test",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(
            set(json.loads(status.stdout)["tasks"]), set(deep.TEST_TARGETS)
        )
        inspector = subprocess.run(
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
        self.assertEqual(inspector.returncode, 0, inspector.stderr)
        for task in deep.TEST_TARGETS:
            self.assertIn(task + ": READY / test", inspector.stdout)
            self.assertIn(task + " | test |", inspector.stdout)
            self.assertNotIn(task + " | validation |", inspector.stdout)
        self.assertFalse(self.pbs.submissions)

    def test_real_guardian_handoff_preserves_three_existing_jobs_and_repeat_start(self):
        from d4_teachers import retry_repair
        from d4_training_continuation import QUEUE
        from d4_training_continuation.capacity import NODE_GROUP, PROJECT

        write(
            self.root / "preflight.json",
            dict(status="PASS", campaign_identity=self.c["identity"]),
        )
        area = self.root.parent / "pbs-test-only"
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
root = Path(os.environ["TEST_ONLY_PBS"])
name = Path(sys.argv[0]).name
rows = json.loads((root / "rows.json").read_text())
with (root / "calls.jsonl").open("a") as f:
    f.write(json.dumps([name] + sys.argv[1:]) + "\\n")
if name == "qstat":
    print((root / "queue.txt").read_text() if "-Qf" in sys.argv else json.dumps({"Jobs": rows}))
elif name == "qsub":
    request = json.loads((Path(sys.argv[1]).parent / "request.json").read_text())
    jid = str(900000 + len(rows)) + ".pbs1"
    rows[jid] = dict(Job_Owner=os.environ["TEST_ONLY_USER"] + "@login", queue=os.environ["TEST_ONLY_QUEUE"], Job_Name=request["job_name"], Submit_arguments=sys.argv[1], job_state="Q")
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

        def cleanup():
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)

        self.addCleanup(cleanup)
        popen = subprocess.Popen

        def track(*a, **kw):
            child = popen(*a, **kw)
            children.append(child)
            return child

        with (
            patch.dict(
                os.environ,
                dict(
                    PATH=str(binary) + os.pathsep + os.environ["PATH"],
                    TEST_ONLY_PBS=str(area),
                    TEST_ONLY_USER=getpass.getuser(),
                    TEST_ONLY_QUEUE=QUEUE,
                ),
            ),
            patch.object(subprocess, "Popen", side_effect=track),
        ):
            first = retry_repair.repair(self.root)
            self.assertTrue(first["controller_runtime"]["first_tick_completed"])
            old = {
                t: read(controller.state_path(self.c, t))["job"]
                for t in ("logic_algorithmic", followup.IF, followup.MATH)
            }
            result = deep.launch(self.root, test_only=True)
            self.assertTrue(
                result["guardian"]["controller_runtime"]["first_tick_completed"], result
            )
            for task, job in old.items():
                self.assertEqual(read(controller.state_path(self.c, task))["job"], job)
            self.assertEqual(
                result["guardian"]["states"]["deepmath_if70"]["status"], "QUEUED"
            )
            self.assertEqual(
                result["guardian"]["states"]["deepmath_cp80"]["status"], "READY"
            )
            again = deep.launch(self.root, test_only=True)
            self.assertEqual(
                again["guardian"]["controller_runtime"]["runtime"]["pid"],
                result["guardian"]["controller_runtime"]["runtime"]["pid"],
            )
            calls = [
                json.loads(line)
                for line in (area / "calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(c[0] == "qsub" for c in calls), 4)
            self.assertFalse(any(c[0] == "qdel" for c in calls))
