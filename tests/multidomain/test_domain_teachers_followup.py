"""Follow-up contracts, full-state pins and interruption recovery; synthetic PBS/GPU."""

import os
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pyarrow.parquet as pq
import test_domain_teachers_deepmath as fixtures
from test_training_continuation import checkpoint
from d4_teachers import (
    controller,
    evaluation,
    exporting,
    followup,
    sampling,
    selection,
    source,
    trainer,
)
from d4_teachers.common import digest, files, read, write
from d4_training_continuation import checkpoints as ck


def completed_test(c, task, rows, follow=None):
    model = exporting.evaluation_model(c, task)
    domain = read(Path(c["path"]) / "runs" / task / "run.json")["domain"]
    binding = dict(
        campaign=c["identity"],
        domain=domain,
        split="test",
        step=model["step"],
        weights=model["weights_identity"],
        verifier=c["contract"]["verifier_identity"],
    )
    if follow:
        binding["teacher_followup"] = follow["identity"]
    ids = [r["sample_id"] for r in rows]
    runtime = dict(ids=ids, rows=len(rows), batches=1, dataset_sha256=digest(rows))
    folder = Path(c["path"]) / "runs" / task / "evaluations/test_fixture"
    journal = evaluation.Journal(folder, binding, runtime)
    payload = dict(
        batch=1,
        rows=len(rows),
        input_digest=digest(ids),
        extras={},
        lists=dict(
            sample_uids=ids,
            sample_scores=[1.0] * len(rows),
            sample_inputs=["q"] * len(rows),
            sample_outputs=["a"] * len(rows),
            sample_gts=["a"] * len(rows),
            sample_turns=[],
            data_source_lst=[[r["domain"] for r in rows]],
        ),
    )
    write(folder / "batch_000001.json", dict(payload=payload, sha256=digest(payload)))
    journal.index, journal.rows = 1, len(rows)
    proof, samples = journal.finish()
    receipt = dict(
        status="PASS",
        split="test",
        step=model["step"],
        binding=binding,
        identity=digest(dict(binding=binding, runtime=runtime)),
        path=str(folder / "receipt.json"),
        scores=evaluation.statistics(samples, rows),
        journal=proof,
    )
    if task == followup.MATH:
        receipt["math_sources"] = followup.source_statistics(samples, rows)
    receipt["content_sha256"] = digest(receipt)
    write(folder / "receipt.json", receipt)
    write(Path(c["path"]) / "runs" / task / "test.json", receipt)
    return receipt


class Followup(fixtures.DeepMathFixture):
    def setUp(self):
        super().setUp()
        self.prepare()
        self.h.change("math_step50", status="DONE", phase="test")
        self.run = self.root / "runs" / followup.MATH
        # The shared Arrow fixture has six train rows. Use two prompts per batch
        # in this CPU fixture only, keeping the actual 200-step stream identity.
        original = sampling.build_plan
        small = patch.object(
            sampling,
            "build_plan",
            side_effect=lambda rows, domain, **kw: original(
                rows, domain, batch_size=2, **kw
            ),
        )
        small.start()
        self.addCleanup(small.stop)
        parent = self.root / "runs/math/checkpoints/global_step_50"
        info = read(self.root / "runs/math/run.json")
        rows = pq.read_table(self.root / "data/math/train.parquet").to_pylist()
        write(parent / "sampling.json", sampling.build_plan(rows, "math").metadata())
        write(
            parent / "teacher.json",
            dict(
                domain="math",
                lineage=info["lineage"],
                campaign_identity=self.c["identity"],
            ),
        )
        marker = read(parent / ck.MARKER)
        marker["files"] = ck.inventory(parent)
        write(parent / ck.MARKER, marker)
        run = self.root / "runs/if"
        lineage = read(run / "run.json")["lineage"]
        checkpoint(run / "checkpoints/global_step_70", 70, lineage=lineage)
        write(
            run / "training_complete.json",
            dict(status="PASS", completed_step=70, lineage=lineage),
        )
        self.h.export("cp_step80")
        self.original_rows = [
            row
            for d in controller.DOMAINS
            for row in pq.read_table(
                self.root / "data" / d / "test.parquet"
            ).to_pylist()
        ]
        completed_test(self.c, "cp_step80", self.original_rows)
        # Remove deliberately invalid placeholders from the older shared fixture.
        for domain in ("math", "if"):
            (self.root / "runs" / domain / "test.json").unlink()
        self.before = {
            d: files(self.root / "runs" / d) for d in ("math", "if", "cp_step80")
        }

    def request(self, task=followup.MATH, role="train"):
        job = source.make_job(self.c, task, role)
        return source.validate_request(Path(job["folder"]) / "request.json")

    def test_full_pin_preserves_sources_and_registration_is_idempotent(self):
        followup.install(self.c)
        p = followup.plan(self.c)
        resume = followup.initial_resume(self.c)
        original = p["targets"][followup.MATH]["checkpoint"]
        self.assertEqual(set(resume["files"]), set(original["files"]))
        for name in resume["files"]:
            self.assertEqual(
                os.stat(Path(resume["path"]) / name).st_ino,
                os.stat(Path(original["path"]) / name).st_ino,
            )
        self.assertIn("data.pt", resume["files"])
        self.assertIn("sampling.json", resume["files"])
        self.assertNotEqual(read(self.run / "run.json")["lineage"], resume["lineage"])
        self.h.tick()
        states = {t: read(controller.state_path(self.c, t)) for t in followup.TARGETS}
        self.assertEqual(followup.install(self.c)["status"], "ALREADY_REGISTERED")
        self.assertEqual(
            states,
            {t: read(controller.state_path(self.c, t)) for t in followup.TARGETS},
        )
        for domain, before in self.before.items():
            # Controller observations can advance; original model/data records cannot.
            current = files(self.root / "runs" / domain)
            self.assertEqual(
                {
                    k: v
                    for k, v in current.items()
                    if k not in ("state.json", "writer.lock")
                },
                {
                    k: v
                    for k, v in before.items()
                    if k not in ("state.json", "writer.lock")
                },
            )

    def test_missing_checkpoint_or_changed_cursor_fails_before_handoff(self):
        cp = self.root / "runs/math/checkpoints/global_step_50"
        (cp / "data.pt").unlink()
        with patch("d4_teachers.retry_repair.repair") as repair:
            with self.assertRaisesRegex(RuntimeError, "Missing dataloader"):
                followup.launch(self.root)
        repair.assert_not_called()
        self.assertFalse((self.root / followup.PLAN).exists())

    def test_crash_before_plan_publication_never_registers_partial_tasks(self):
        real = followup.write

        def fail_plan(path, value):
            if Path(path).name == followup.PLAN:
                raise OSError("publication crash")
            return real(path, value)

        with patch.object(followup, "write", side_effect=fail_plan):
            with self.assertRaisesRegex(OSError, "publication crash"):
                followup.install(self.c)
        self.assertNotIn(followup.MATH, controller.tasks(self.c))
        followup.install(self.c)
        self.assertIn(followup.MATH, controller.tasks(self.c))

    def test_frozen_requests_include_full_resume_and_split_specific_math_data(self):
        followup.install(self.c)
        r = self.request()
        self.assertEqual(r["policy"]["max_steps"], 200)
        self.assertEqual(r["config"]["training"]["total_training_steps"], 200)
        self.assertEqual(followup.end_step(r), 100)
        validation_files, validation_ids = followup.files_and_ids(self.c, "validation")
        test_files, test_ids = followup.files_and_ids(self.c, "test")
        self.assertTrue(
            all(Path(f).name == "validation.parquet" for f in validation_files)
        )
        self.assertFalse(set(validation_ids).intersection(test_ids))
        self.assertEqual(len(test_files), 5)
        self.assertEqual(len(validation_files), 2)
        self.assertEqual(len(test_ids), len(self.original_rows) + 8)
        if_request = self.request(followup.IF, "export")
        self.assertEqual(if_request["selected"]["step"], 70)
        self.h.export(followup.IF)
        after = self.request(followup.IF, "test")
        self.assertEqual(after["evaluation_model"]["step"], 70)
        self.assertNotIn("heldout_manifest", after)
        with self.assertRaisesRegex(RuntimeError, "cannot train"):
            self.request(followup.IF, "train")
        for request in (r, if_request, after):
            folder = Path(request["job_dir"])
            child = subprocess.run(
                [sys.executable, "-I", str(folder / "entry.py"), "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(child.returncode, 0, child.stderr)
            subprocess.run(["bash", "-n", str(folder / "stage.pbs")], check=True)
            self.assertIn("walltime=24:00:00", (folder / "stage.pbs").read_text())

    def test_two_source_selection_uses_equal_means_and_never_test(self):
        followup.install(self.c)
        r, cp = self.request(), followup.initial_resume(self.c)
        receipt = dict(
            status="PASS",
            split="validation",
            step=50,
            path="validation50",
            identity="validation50",
            scores={"math": {"mean": 0.39}},
            math_sources={
                "nemotron": {"mean": 0.9, "n": 3},
                "deepmath": {"mean": 0.2, "n": 8},
            },
        )
        best = selection.select(r, cp, receipt)
        self.assertAlmostEqual(best["score"], 0.55)
        cp80 = checkpoint(
            self.run / "checkpoints/global_step_80", 80, lineage=r["lineage"]
        )
        later = dict(
            receipt,
            step=80,
            scores={"math": {"mean": 0.61}},
            math_sources={
                "nemotron": {"mean": 0.1, "n": 3},
                "deepmath": {"mean": 0.8, "n": 8},
            },
        )
        self.assertEqual(selection.select(r, cp80, later)["step"], 50)
        with self.assertRaisesRegex(RuntimeError, "Only validation"):
            selection.select(r, cp80, dict(later, split="test"))
        self.assertEqual(
            followup.stop_state(r, 70, 50), dict(early_stop=False, budget_reached=False)
        )
        self.assertTrue(followup.stop_state(r, 80, 50)["early_stop"])
        self.assertFalse(followup.stop_state(r, 80, 60)["early_stop"])
        self.assertEqual(
            followup.stop_state(r, 100, 100),
            dict(early_stop=False, budget_reached=True),
        )

    def test_baseline_50_and_later_resume_use_correct_full_checkpoint(self):
        followup.install(self.c)
        request = self.request()
        instance = trainer.trainer_class(object)()
        instance.teacher_request = request
        instance.teacher_resume = followup.initial_resume(self.c)
        self.assertEqual(instance.current_checkpoint(50), instance.teacher_resume)
        later = checkpoint(
            self.run / "checkpoints/global_step_55", 55, lineage=request["lineage"]
        )
        self.assertEqual(instance.current_checkpoint(55), later)
        self.assertEqual(controller.progress(self.c, followup.MATH, "train"), 5500000)
        with self.assertRaisesRegex(RuntimeError, "committed full checkpoint"):
            instance.current_checkpoint(57)

    def test_real_torchdata_cursor_restores_at_50_and_next_batch_is_51(self):
        import torch
        from types import SimpleNamespace
        from torch.utils.data import SequentialSampler
        from torchdata.stateful_dataloader import StatefulDataLoader

        rows = pq.read_table(self.root / "data/math/train.parquet").to_pylist()
        plan = sampling.build_plan(rows, "math")

        def loader():
            ds = sampling.PlannedDataset(list(range(len(rows))), plan)
            return StatefulDataLoader(
                ds,
                batch_size=2,
                sampler=SequentialSampler(ds),
                num_workers=0,
                drop_last=True,
            )

        original = loader()
        iterator = iter(original)
        for _ in range(50):
            next(iterator)
        parent = self.root / "runs/math/checkpoints/global_step_50"
        torch.save(original.state_dict(), parent / "data.pt")
        marker = read(parent / ck.MARKER)
        marker["files"] = ck.inventory(parent)
        write(parent / ck.MARKER, marker)
        followup.install(self.c)
        request = self.request()
        resume = followup.initial_resume(self.c)

        class Base:
            def _load_checkpoint(obj):
                obj.train_dataloader.load_state_dict(
                    torch.load(Path(resume["path"]) / "data.pt", weights_only=False)
                )
                obj.global_steps = 50

        obj = trainer.trainer_class(Base)()
        obj.total_training_steps, obj.train_dataloader = 200, loader()
        obj.config = SimpleNamespace(trainer=SimpleNamespace(val_only=False))
        obj.checkpoint_manager = SimpleNamespace(update_weights=lambda step: None)
        phase = Path(request["job_dir"]) / "train"
        obj.configure_teacher(request, phase, resume, plan)
        # The cluster audit binder requires a job-local imported module. The
        # actual torchdata state here is JSON-compatible, so compare it exactly.
        with patch(
            "d4_training_continuation.pinned_audit.state_digest", side_effect=digest
        ):
            obj._load_checkpoint()
        self.assertEqual(read(phase / "resume.json")["step"], 50)
        self.assertEqual(
            next(iter(obj.train_dataloader)).tolist(), list(plan.batch(51))
        )

    def test_271_retries_preserve_logic_and_completed_eval_batches(self):
        logic = source.make_job(self.c, "logic_algorithmic", "train")
        job = self.pbs.submit(logic["folder"], logic["job_name"])
        self.h.change("logic_algorithmic", status="RUNNING", job=job)
        followup.install(self.c)
        states = self.h.tick()
        self.assertEqual(len(self.pbs.submissions), 3)
        self.assertEqual(states["logic_algorithmic"]["job"], job)
        journal = self.run / "evaluations/validation_50_fixture"
        write(journal / "progress.json", dict(completed_rows=8))
        write(journal / "batch_000001.json", dict(keep=True))
        before = files(journal)
        for attempt in range(5):
            for task in followup.TARGETS:
                self.pbs.rows[states[task]["job"]["job_id"]] = dict(
                    job_state="F", Exit_status=271
                )
            states = self.h.tick()
            self.assertEqual(states[followup.MATH]["status"], "QUEUED")
            self.assertEqual(states[followup.IF]["phase"], "export")
        self.assertGreater(states[followup.MATH]["no_progress_retries"], 3)
        self.assertEqual(files(journal), before)
        count = len(self.pbs.submissions)
        self.pbs.unknown = True
        self.h.tick()
        self.h.tick()
        self.assertEqual(len(self.pbs.submissions), count)
        self.assertEqual(
            read(controller.state_path(self.c, "logic_algorithmic"))["job"], job
        )
        self.assertFalse(self.pbs.cancellations)

    def test_if_export_then_complete_four_domain_test(self):
        followup.install(self.c)
        state = self.h.tick()[followup.IF]
        self.h.export(followup.IF)
        write(Path(state["job"]["folder"]) / "result.json", dict(status="PASS"))
        self.pbs.rows[state["job"]["job_id"]] = dict(job_state="F", Exit_status=0)
        state = self.h.tick()[followup.IF]
        self.assertEqual(state["phase"], "test")
        receipt = completed_test(
            self.c, followup.IF, self.original_rows, followup.plan(self.c)
        )
        self.assertEqual(followup.validate_test(self.c, followup.IF), receipt)
        write(Path(state["job"]["folder"]) / "result.json", dict(status="PASS"))
        self.pbs.rows[state["job"]["job_id"]] = dict(job_state="F", Exit_status=0)
        self.assertEqual(self.h.tick()[followup.IF]["status"], "DONE")
        changed = dict(
            receipt, scores=dict(receipt["scores"], math=dict(n=2, mean=0.0))
        )
        changed["content_sha256"] = digest(
            {k: v for k, v in changed.items() if k != "content_sha256"}
        )
        write(self.root / "runs" / followup.IF / "test.json", changed)
        with self.assertRaisesRegex(RuntimeError, "scores changed"):
            followup.validate_test(self.c, followup.IF)

    def test_cp80_is_adopted_with_honest_provenance_before_logic_finishes(self):
        followup.install(self.c)
        self.assertIsNone(exporting.manifest(self.c))
        selected = read(self.root / "selected_teachers.json")
        cp = selected["teachers"]["conversational_pivot"]
        self.assertEqual(cp["source_task"], "cp_step80")
        self.assertEqual(cp["selected_step"], 80)
        self.assertIsNone(cp["selection_split"])
        self.assertIsNone(cp["selection_score"])
        self.assertEqual(
            cp["original_validation_best"], {"original": "conversational_pivot"}
        )
        self.assertIn("logic_algorithmic", selected["missing_teachers"])
        self.assertEqual(
            selected["teachers"]["math"]["pending_replacement"], followup.MATH
        )
        self.assertEqual(read(self.root / "runs/if/best.json"), {"original": "if"})

    def test_math_test_separates_sources_and_inspector_displays_new_tasks(self):
        followup.install(self.c)
        r, cp = self.request(), followup.initial_resume(self.c)
        selection.select(
            r,
            cp,
            dict(
                status="PASS",
                step=50,
                split="validation",
                path="val",
                identity="val",
                scores={"math": {"mean": 0.5}},
                math_sources={"nemotron": {"mean": 0.2}, "deepmath": {"mean": 0.8}},
            ),
        )
        self.h.export(followup.MATH)
        self.assertEqual(followup.chosen_task(self.c, "math"), followup.MATH)
        paths, _ = followup.files_and_ids(self.c, "test")
        rows = [r for p in paths for r in pq.read_table(p).to_pylist()]
        receipt = completed_test(self.c, followup.MATH, rows, followup.plan(self.c))
        self.assertEqual(followup.validate_test(self.c, followup.MATH), receipt)
        self.assertEqual(receipt["math_sources"]["nemotron"]["n"], 2)
        self.assertEqual(receipt["math_sources"]["deepmath"]["n"], 8)
        self.assertEqual(followup.four_domain_scores(receipt)["math"]["n"], 2)
        script = source.REPO / "scripts/inspect_domain_teachers.py"
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                str(script),
                "--campaign",
                str(self.root),
                "--no-pbs",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        for expected in (
            "math_continued",
            "if_step70",
            "Math/Nemotron",
            "来源等权均值",
            "用户指定 cp_step80",
        ):
            self.assertIn(expected, child.stdout)

    def test_repeat_launch_has_no_install_callback_or_reset(self):
        followup.install(self.c)
        with patch("d4_teachers.retry_repair.repair", return_value={}) as repair:
            followup.launch(self.root)
        repair.assert_called_once_with(self.root, install_under_lease=None)

    def test_real_validation_loop_resumes_two_source_baseline_then_reuses_receipt(self):
        from omegaconf import OmegaConf
        import numpy as np

        followup.install(self.c)
        request, cp = self.request(), followup.initial_resume(self.c)
        paths, _ = followup.files_and_ids(self.c, "validation")
        rows = [r for path in paths for r in pq.read_table(path).to_pylist()]

        class Rows:
            def __len__(self):
                return len(rows)

        harness = fixtures.fixtures.EvaluationRecovery(methodName="runTest")

        def runner(fail=None):
            obj = harness.validator(len(rows), fail=fail)
            obj.config = OmegaConf.create(json.loads(json.dumps(obj.config)))
            obj.config.actor_rollout_ref.rollout.val_kwargs.temperature = 0
            obj.config.data = dict(validation_shuffle=False, val_batch_size=8)
            obj.teacher_request, obj.global_steps = request, 50
            obj.val_dataset = Rows()
            obj.val_dataset.rows = rows
            obj.current_checkpoint = lambda step: cp
            obj.weights_identity = lambda cp: digest(cp["files"])
            obj._dump_generations = lambda **kw: None
            for i, batch in enumerate(obj.val_dataloader):
                batch["uid"] = np.array(
                    [r["sample_id"] for r in rows[i * 8 : i * 8 + 8]], dtype=object
                )
            return obj

        with (
            patch.object(
                evaluation, "dependency_versions", return_value={"fixture": "1"}
            ),
            patch("builtins.print"),
        ):
            first = runner(fail=8)
            with self.assertRaisesRegex(RuntimeError, "walltime"):
                evaluation.validate(first, first._validate)
            self.assertFalse((self.run / "best.json").exists())
            second = runner()
            evaluation.validate(second, second._validate)
            state = read(self.run / "validation_state.json")
            receipt = evaluation.read_receipt(state["receipt"])
            self.assertEqual(receipt["journal"]["reused_batches"], 1)
            self.assertEqual(second.generated, [8])
            self.assertEqual(receipt["math_sources"]["nemotron"], {"n": 3, "mean": 1.0})
            self.assertEqual(receipt["math_sources"]["deepmath"], {"n": 8, "mean": 1.0})
            self.assertEqual(read(self.run / "best.json")["score"], 1.0)
            self.assertFalse(second.teacher_stop)
            third = runner()
            evaluation.validate(third, third._validate)
            self.assertEqual(third.generated, [])

    def test_real_guardian_registers_new_tasks_and_repeat_does_not_resubmit(self):
        import test_domain_teachers_retry_repair as old
        from d4_teachers import retry_repair

        transport = old.RetryRepair(methodName="runTest")
        transport.setUp()
        self.addCleanup(transport.doCleanups)
        write(
            self.root / "preflight.json",
            dict(status="PASS", campaign_identity=self.c["identity"]),
        )
        actual = subprocess.Popen

        def record(*args, **kw):
            child = actual(*args, **kw)
            transport.children.append(child)
            return child

        with patch.object(subprocess, "Popen", side_effect=record):
            initial = retry_repair.repair(self.root)
            self.assertTrue(initial["controller_runtime"]["first_tick_completed"])
            logic = read(controller.state_path(self.c, "logic_algorithmic"))["job"]
            result = followup.launch(self.root)
            self.assertTrue(
                result["guardian"]["controller_runtime"]["first_tick_completed"], result
            )
            for task in followup.TARGETS:
                self.assertEqual(result["guardian"]["states"][task]["status"], "QUEUED")
            self.assertEqual(
                read(controller.state_path(self.c, "logic_algorithmic"))["job"], logic
            )
            self.assertEqual(sum(call[0] == "qsub" for call in transport.calls()), 3)
            repeated = followup.launch(self.root)
            self.assertEqual(
                repeated["guardian"]["controller_runtime"]["runtime"]["pid"],
                result["guardian"]["controller_runtime"]["runtime"]["pid"],
            )
            self.assertEqual(sum(call[0] == "qsub" for call in transport.calls()), 3)
            self.assertFalse(any(call[0] == "qdel" for call in transport.calls()))
