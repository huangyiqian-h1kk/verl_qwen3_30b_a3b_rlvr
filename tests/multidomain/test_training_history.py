"""Regression tests for scientific accounting across interrupted D4 attempts."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/multidomain/collect_training_history.py"
SPEC = importlib.util.spec_from_file_location("training_history", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def job(self, role, name, jid, log, stage="train"):
        folder = self.root / "outputs/multidomain" / (m.BASE + m.RUNS[role]) / "jobs" / name
        self.write(folder / "job_id.txt", jid)
        self.write(folder / "request.json", json.dumps({"stage": stage}))
        self.write(folder / "pbs.log", log)
        return folder

    def collect(self):
        collector = m.Collector(self.root, self.root / "analysis", include_pbs=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = collector.run()
        return collector, summary

    def test_console_ansi_scientific_negative_and_validation(self):
        row = m.parse_metric("\x1b[36m(Runner pid=1)\x1b[0m step:5 - actor/pg_loss:-1.3e-05 - actor/entropy:0.432 - val-core/math/acc/mean@1:0.6 - timing_s/step:2000.0")
        self.assertEqual(row["step"], 5)
        self.assertAlmostEqual(row["metrics"]["actor/pg_loss"], -0.000013)
        self.assertEqual(row["metrics"]["val-core/math/acc/mean@1"], 0.6)
        self.assertTrue(m.is_update(row))

    def test_nonfinite_is_explicit_not_silent_zero(self):
        row = m.parse_metric("step:3 - actor/pg_loss:nan - actor/grad_norm:inf - actor/entropy:0.1")
        self.assertIsNone(row["metrics"]["actor/pg_loss"])
        self.assertEqual(row["nonfinite_keys"], ["actor/pg_loss", "actor/grad_norm"])
        json.dumps(row, allow_nan=False)

    def test_numpy_scalar_reprs_from_pinned_verl_are_not_dropped(self):
        row = m.parse_metric("step:25 - actor/pg_loss:np.float64(-0.002) - actor/lr:np.float64(1e-6) - val-core/math/acc/mean@1:np.float64(0.5) - num_turns/min:np.int32(2) - sampling/domain/math:numpy.float32(1.0)")
        self.assertEqual(len(row["metrics"]), 5)
        self.assertEqual(row["metrics"]["actor/pg_loss"], -0.002)
        self.assertEqual(row["metrics"]["num_turns/min"], 2)
        self.assertTrue(m.is_update(row))
        self.assertIsNone(m.parse_metric("step:0 - val-core/math/acc/mean@1:np.float64(__import__('os'))"))

    def test_progress_restore_and_sampling_are_not_completed_steps(self):
        for line in ("Training Progress: 14% 65/468", "load checkpoint step 65", "[0390] sampling step 66: domain=if"):
            self.assertIsNone(m.parse_metric(line))
        self.assertFalse(m.is_update(m.parse_metric("step:0 - val-core/math/acc/mean@1:0.4")))
        self.assertFalse(m.is_update(m.parse_metric("step:66 - actor/lr:1e-6")))

    def test_all_attempts_preserved_without_merging_candidates_or_checks(self):
        self.job("random_single", "a_train", "1.pbs1", "step:0 - val-core/math/acc/mean@1:0.4\nstep:1 - actor/pg_loss:0.1 - timing_s/step:20\n")
        self.job("random_single", "b_train", "2.pbs1", "step:1 - actor/pg_loss:0.2 - timing_s/step:30\nstep:2 - actor/pg_loss:0.3 - timing_s/step:40\n")
        self.job("random_single", "c_check", "3.pbs1", "step:99 - actor/pg_loss:1\n", stage="smoke")
        self.job("random_dual", "d_train", "4.pbs1", "step:3 - actor/pg_loss:0.5\n")
        collector, summary = self.collect()
        s = summary["roles"]["random_single"]
        self.assertEqual(s["max_logged_update_step"], 2)
        self.assertEqual(s["unique_logged_update_step_count"], 2)
        self.assertEqual(s["repeated_step_candidates"], {"1": 2})
        self.assertEqual(s["metric_endpoints_all_attempts"]["actor/pg_loss"]["first_values"], [0.1, 0.2])
        self.assertEqual(s["logged_step_seconds_all_attempts"]["sum"], 90)
        self.assertEqual(summary["roles"]["random_dual"]["max_logged_update_step"], 3)
        self.assertEqual(len(collector.metrics), 6)

    def test_same_job_restart_segments_and_partial_tail(self):
        self.job("domain_single", "a_train", "1.pbs1",
            "step:4 - actor/pg_loss:0.1\nstep:5 - actor/pg_loss:0.1\nstep:4 - actor/pg_loss:0.2\nstep:6 - actor/pg_loss:0.3")
        collector, summary = self.collect()
        self.assertEqual([r["segment"] for r in collector.metrics], [0, 0, 1])
        self.assertEqual(summary["roles"]["domain_single"]["max_logged_update_step"], 5)
        self.assertTrue(any("unterminated" in w.get("note", "") for w in collector.warnings))

    def test_generations_numeric_stats_samples_and_no_fake_token_counts(self):
        folder = self.job("random_single", "a_train", "1.pbs1", "")
        gen = folder.parent.parent / "generations/train/0.jsonl"
        rows = [{"input": "question", "output": "A" * 13000, "score": 1, "domain": "math", "step": 0},
                {"input": "question2", "output": "B", "score": 0, "domain": "math", "step": 0}]
        self.write(gen, "".join(json.dumps(r) + "\n" for r in rows) + '{"partial":')
        collector, summary = self.collect()
        score = [r for r in collector.generation_stats if r["metric"] == "score"][0]
        self.assertEqual(score["mean"], 0.5)
        self.assertEqual(score["count"], 2)
        self.assertFalse(any("token" in r["metric"] for r in collector.generation_stats))
        sample = json.loads((collector.output / "output_samples.jsonl").read_text().splitlines()[0])
        self.assertEqual(sample["record"]["output"]["characters"], 13000)
        self.assertTrue(sample["record"]["output"]["truncated"])
        self.assertIsNone(summary["roles"]["random_single"]["max_logged_update_step"])

    def test_full_logs_archived_checkpoint_weights_and_symlinks_not_read(self):
        folder = self.job("random_single", "a_train", "1.pbs1", "step:1 - actor/pg_loss:0.1\n")
        run = folder.parent.parent
        cp = run / "checkpoints/global_step_5"
        self.write(cp / "actor/model.pt", "DO NOT READ THIS")
        self.write(cp / "actor/ckpt_contents.json", '{"global_step":5}')
        self.write(cp / "COMPLETE_0390.json", '{"step":5}')
        (folder / "linked.log").symlink_to(cp / "actor/model.pt")
        digest = hashlib.sha256((folder / "pbs.log").read_bytes()).hexdigest()
        collector, _ = self.collect()
        with tarfile.open(collector.output.with_suffix(".tar.gz")) as archive:
            names = archive.getnames()
            self.assertFalse(any(name.endswith(("model.pt", "linked.log")) for name in names))
            name = "evidence/" + str((folder / "pbs.log").relative_to(self.root))
            self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), digest)
            self.assertIn("analysis/metrics_long.csv", names)
        self.assertTrue(collector.checkpoints[0]["metadata_only_not_validated"])
        self.assertEqual(hashlib.sha256((folder / "pbs.log").read_bytes()).hexdigest(), digest)

    def test_exact_log_copies_do_not_double_count(self):
        text = "step:1 - actor/pg_loss:0.1 - timing_s/step:20\n"
        folder = self.job("random_single", "a_train", "1.pbs1", text)
        self.write(folder / "train.log", text)
        collector, summary = self.collect()
        self.assertEqual(len(collector.metrics), 1)
        self.assertEqual(summary["roles"]["random_single"]["logged_step_seconds_all_attempts"]["sum"], 20)

    def test_nonidentical_mirror_does_not_inflate_timing(self):
        text = "step:1 - actor/pg_loss:0.1 - timing_s/step:20\n"
        folder = self.job("random_single", "a_train", "1.pbs1", text)
        self.write(folder / "train.log", "other header\n" + text)
        collector, summary = self.collect()
        self.assertEqual(len(collector.metrics), 2)
        self.assertEqual(summary["roles"]["random_single"]["logged_step_seconds_all_attempts"]["sum"], 20)

    def test_pbs_walltime_and_gpu_hours_never_cput(self):
        collector = m.Collector(self.root, self.root / "analysis", include_pbs=False)
        collector.jobs = [{"role": "random_single", "stage": "train", "job_id": "1.pbs1",
            "pbs": {"resources_used": {"walltime": "01:30:00", "cput": "99:00:00"}, "Resource_List": {"ngpus": 8}}},
            {"role": "random_single", "stage": "train", "job_id": "2.pbs1", "pbs": {"query_error": "expired"}}]
        row = collector.summarize()["roles"]["random_single"]
        self.assertEqual(row["allocated_job_hours_known"], 1.5)
        self.assertEqual(row["allocated_gpu_hours_known"], 12)
        self.assertEqual(row["pbs_walltime_coverage"], {"known": 1, "jobs": 2, "gpu_known": 1})

    def continuation_job(self, role, name, jid, job_role="formal", committed=False):
        folder = self.job(role, name, jid, "")
        lineage = name + "_lineage"
        self.write(folder / "request.json", json.dumps(dict(stage="train", role=job_role,
            identity=name, lineage=lineage, expected_nodes=2 if job_role == "candidate" else 1)))
        if committed:
            commit = dict(identity=name, job={"job_id": jid}, authority={"lineage": lineage}, checkpoint={"step": 10})
            self.write(folder / "committed.json", json.dumps(commit))
            self.write(folder.parent.parent / "control/handoff_commit.json", json.dumps(commit))
        return folder, lineage

    def step(self, folder, phase, step, world, **values):
        self.write(folder / phase / "phase.json", json.dumps(dict(target_world=world, source_step=step - 1,
                                                                execution_id="exec_" + phase)))
        self.write(folder / phase / f"step_{step}.json", json.dumps(dict(step=step, status="PASS",
            actor_update_completed=True, weights_synchronized=True, prompts=256, responses=2048, **values)))

    def test_continuation_handoff_phase_metrics_and_failed_candidate_separation(self):
        self.job("random_single", "old_train", "1.pbs1", "step:0 - val-core/math/acc/mean@1:0.4\nstep:5 - actor/pg_loss:0.1\n")
        random, _ = self.continuation_job("random_continuation", "candidate_a", "2.pbs1", "candidate", True)
        domain, _ = self.continuation_job("domain_continuation", "formal_b", "3.pbs1")
        failed, _ = self.continuation_job("domain_continuation", "candidate_c", "4.pbs1", "candidate")
        self.step(random, "C", 8, 8, actor_metrics={"actor/entropy": [0.3, 0.5], "actor/lr": 1e-6},
                  response_domain_counts={"math": 1024, "if": 1024}, reward_mean=0.75)
        self.step(random, "E", 11, 16, actor_metrics={"actor/pg_loss": -0.01})
        self.write(random / "E/phase.json", json.dumps(dict(target_world=16, source_step=8, execution_id="exec_E")))
        self.step(domain, "train", 7, 8, domain="math", actor_metrics={"actor/entropy": [0.4]})
        self.step(failed, "A", 99, 16)
        self.write(random / "pbs.log", "step:8 - actor/pg_loss:0.02 - timing_s/step:20\nstep:11 - actor/pg_loss:-0.01 - timing_s/step:30 - val-core/math/acc/mean@1:0.6\n")
        self.write(random.parent.parent / "control/state.json", json.dumps(dict(active={"job_id": "2.pbs1"}, phase="ACTIVE", promoted=True)))
        collector, summary = self.collect()
        self.assertEqual(summary["setups"]["random"]["max_completed_step_on_formal_attempts"], 11)
        self.assertEqual(summary["setups"]["random"]["validation_steps"], [0, 11])
        self.assertEqual(summary["setups"]["random"]["handoff_checkpoint_step"], 10)
        self.assertEqual(summary["setups"]["domain"]["max_completed_step_on_formal_attempts"], 7)
        self.assertEqual(summary["roles"]["domain_continuation"]["max_structured_completed_step_all_attempts"], 99)
        rows = {r["step"]: r for r in collector.step_records}
        self.assertEqual(rows[8]["allocation_nodes"], 2)
        self.assertEqual(rows[8]["target_world"], 8)
        self.assertEqual(rows[8]["step_usage"], "qualification")
        self.assertEqual(rows[11]["step_usage"], "formal_after_handoff")
        self.assertAlmostEqual(rows[8]["metrics"]["actor_samples/actor/entropy/mean"], 0.4)
        self.assertEqual(rows[8]["record"]["actor_metrics"]["actor/entropy"], [0.3, 0.5])
        self.assertEqual([r for r in collector.metrics if r["job_id"] == "2.pbs1"][0]["phase"], "C")
        self.assertEqual(summary["roles"]["random_continuation"]["logged_step_seconds_all_attempts"]["sum"], 50)
        self.assertTrue((collector.output / "step_metrics_long.csv").is_file())

    def test_lineage_checkpoint_generation_and_all_phase_outputs_are_collected(self):
        folder, lineage = self.continuation_job("random_continuation", "candidate_a", "2.pbs1", "candidate", True)
        run = folder.parent.parent / "lineages" / lineage
        self.write(run / "checkpoints/global_step_10/actor/ckpt_contents.json", '{"global_step":10}')
        self.write(run / "checkpoints/global_step_10/actor/model.pt", "NEVER ARCHIVE WEIGHTS")
        self.write(run / "generations/train/10.jsonl", '{"step":10,"score":1,"domain":"math","output":"answer"}\n')
        self.write(folder / "B/generations/train/7.jsonl", '{"step":7,"score":0,"domain":"if"}\n')
        self.write(folder / "E/raw_generations/train/11.jsonl", '{"step":11,"score":1,"domain":"math"}\n')
        self.step(folder, "B", 7, 16)
        collector, _ = self.collect()
        self.assertEqual(len(collector.generation_stats), 4)  # three scores and one character count
        self.assertEqual(collector.checkpoints[0]["lineage"], lineage)
        rows = [json.loads(x) for x in (collector.output / "generation_rows.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(next(r for r in rows if r["step"] == 10)["lineage"], lineage)
        self.assertEqual(next(r for r in rows if r["step"] == 7)["job_id"], "2.pbs1")
        with tarfile.open(collector.output.with_suffix(".tar.gz")) as archive:
            self.assertFalse(any(p.endswith("model.pt") for p in archive.getnames()))
            self.assertIn("analysis/continuation_status.json", archive.getnames())

    def test_partial_step_evidence_and_wrong_commit_do_not_claim_formal_progress(self):
        folder, _ = self.continuation_job("domain_continuation", "candidate_a", "2.pbs1", "candidate", True)
        self.write(folder / "committed.json", json.dumps(dict(identity="different", job={"job_id": "2.pbs1"},
                                                             authority={"lineage": "candidate_a_lineage"}, checkpoint={"step": 10})))
        self.step(folder, "E", 10, 16, actor_metrics={"actor/entropy": [0.2, float("nan")]})
        formal, _ = self.continuation_job("domain_continuation", "formal_b", "3.pbs1")
        self.write(formal / "train/step_11.json", '{"step":11,"status":"PASS","actor_update_completed":true}')
        self.write(formal / "train/step_12.json", '{"step":')
        collector, summary = self.collect()
        self.assertIsNone(summary["setups"]["domain"]["max_completed_step_on_formal_attempts"])
        self.assertFalse(next(r for r in collector.step_records if r["step"] == 10)["handoff_committed"])
        self.assertTrue(any("step_12.json" in w.get("path", "") for w in collector.warnings))
        self.assertFalse(any("actor_samples/actor/entropy/mean" in r["metrics"] for r in collector.step_records))

    def test_structured_and_console_replay_are_not_added_as_net_steps(self):
        first, _ = self.continuation_job("random_continuation", "formal_a", "1.pbs1")
        second, _ = self.continuation_job("random_continuation", "formal_b", "2.pbs1")
        for folder in (first, second):
            self.step(folder, "train", 30, 8, actor_metrics={"actor/pg_loss": 0.1})
            self.write(folder / "pbs.log", "step:30 - actor/pg_loss:0.1 - timing_s/step:25\n")
        collector, summary = self.collect()
        self.assertEqual(summary["setups"]["random"]["observed_formal_attempt_steps"], [30])
        self.assertEqual(len(collector.step_records), 2)
        self.assertEqual(summary["roles"]["random_continuation"]["logged_step_seconds_all_attempts"]["sum"], 50)

    def test_malformed_control_does_not_discard_training_metrics(self):
        folder, _ = self.continuation_job("random_continuation", "candidate_a", "2.pbs1", "candidate")
        self.write(folder / "committed.json", '{"identity":"candidate_a","job":[],"authority":[],"checkpoint":null}')
        self.write(folder.parent.parent / "control/state.json", '[]')
        self.write(folder.parent.parent / "control/handoff_commit.json", '{"checkpoint":null}')
        self.step(folder, "E", 10, 16)
        collector, summary = self.collect()
        self.assertFalse(collector.step_records[0]["handoff_committed"])
        self.assertIsNone(summary["setups"]["random"]["max_completed_step_on_formal_attempts"])
        self.assertTrue(any("control record" in w.get("error", "") for w in collector.warnings))


class ValidationHistoryTests(unittest.TestCase):
    setUp = HistoryTests.setUp
    write = HistoryTests.write
    collect = HistoryTests.collect

    def evaluation(self, kind="random", step=125, tag="first", protocol="v1", source_record=None):
        run = self.root / "outputs/multidomain" / (m.BASE + m.RUNS[kind + "_continuation"])
        folder = run / "jobs" / ("evaluation_" + tag)
        cp = source_record or dict(step=step, world_size=16 if kind == "random" else 8,
            lineage=kind + "_lineage", files={"actor/model.distcp": {"bytes": 123, "mtime_ns": 456}},
            path=str(self.root / "outputs/multidomain/validation_backfill_sources" / kind / ("global_step_" + str(step))))
        contract = dict(source={"record": cp}, role="evaluation", lineage=cp["lineage"],
                        kind=kind, validation_semantics={"kind": kind, "verifier": protocol})
        req = dict(contract, contract=contract, identity=m.identity(contract), stage="train",
            expected_nodes=1, job_dir=str(folder), control_dir=str(run / "control"),
            run_dir=str(folder / "evaluation"))
        self.write(folder / "request.json", json.dumps(req))
        output = folder / "evaluation/generations/train" / (str(step) + ".jsonl")
        self.write(output, "".join(json.dumps(dict(step=step, input=f"q{i}", output="answer", score=i % 2)) + "\n"
                                 for i in range(1006)))
        key, proto = m.checkpoint_identity(cp), m.validation_protocol(req)
        journal = run / "control/validation_batches" / (key + "_" + proto)
        manifest = dict(version="2026-10-23.validation-batches.1", checkpoint=key, protocol=proto,
                        runtime={"rows": 1006, "batches": 126})
        self.write(journal / "manifest.json", json.dumps(manifest))
        self.write(journal / "progress.json", json.dumps(dict(completed_batches=126, completed_rows=1006,
                                                             cached_batches_this_process=107)))
        self.write(journal / "batch_000001.json", '"DO NOT ARCHIVE LARGE CACHE OUTPUTS"')
        receipt = dict(version="checkpoint-validation-v1", status="PASS", step=step, lineage=cp["lineage"],
            checkpoint_key=key, protocol=proto, metrics={k: 0.5 for k in m.VAL_CORE}, at_utc="2026-10-24T02:00:00+00:00",
            outputs=dict(path=str(output), rows=1006, sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
            origin=dict(job_dir=str(folder), phase="train", journal=dict(path=str(journal),
                manifest_sha256=m.identity(manifest), reused_batches=107, batches=126)))
        receipt_path = run / "control/evaluations" / (key + "_" + proto + ".json")
        self.write(receipt_path, json.dumps(receipt))
        self.reference(folder, receipt_path, step)
        self.parent(tag, {kind: folder})
        return folder, receipt_path, output

    def reference(self, folder, receipt, step):
        self.write(folder / "train" / f"validation_{step}.json", json.dumps(dict(status="PASS", step=step,
            reused=True, receipt=str(receipt), receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())))

    def parent(self, tag, children):
        area = self.root / "outputs/multidomain/validation_backfills/test-operation"
        folder = area / "jobs" / tag
        jid = "9" + str(len(tag)) + ".pbs1"
        child_records = {kind: dict(folder=str(child), request_sha256=hashlib.sha256((child / "request.json").read_bytes()).hexdigest())
                         for kind, child in children.items()}
        contract = dict(expected_nodes=1, children=child_records)
        self.write(folder / "request.json", json.dumps(dict(contract, contract=contract, identity=m.identity(contract))))
        self.write(folder / "job_id.txt", jid)
        self.write(folder / "result.json", json.dumps(dict(status="PASS", evaluations={k: dict(status="PASS", evaluated_step=125 if k == "random" else 100,
            optimizer_updates=0, checkpoints_written=0) for k in children})))
        self.write(area / "state.json", json.dumps(dict(operation="test-operation", phase="PASS", attempts=2,
                                                       job=dict(job_id=jid, folder=str(folder)))))
        return folder, jid

    def test_completed_backfills_are_evaluation_points_not_training_progress(self):
        self.evaluation()
        self.evaluation("domain", 100, "second")
        collector, summary = self.collect()
        self.assertEqual(summary["setups"]["random"]["backfilled_validation_steps"], [125])
        self.assertEqual(summary["setups"]["domain"]["backfilled_validation_steps"], [100])
        self.assertIsNone(summary["setups"]["random"]["max_completed_step_on_formal_attempts"])
        self.assertIsNone(summary["roles"]["domain_continuation"]["max_logged_update_step"])
        self.assertEqual(len(collector.validations), 2)
        self.assertTrue(all(r["verified_complete"] for r in collector.validations))
        self.assertEqual(collector.validations[0]["reused_batches"], 107)
        with (collector.output / "training_dynamics.csv").open() as f:
            self.assertEqual(list(csv.DictReader(f)), [])
        with (collector.output / "validation_dynamics.csv").open() as f:
            self.assertEqual({int(r["step"]) for r in csv.DictReader(f)}, {100, 125})

    def test_reused_receipt_across_jobs_is_one_point_and_parent_time_is_once(self):
        folder, receipt, _ = self.evaluation()
        second = folder.parent / "evaluation_reuse"
        request = json.loads((folder / "request.json").read_text())
        request.update(job_dir=str(second), run_dir=str(second / "evaluation"))
        self.write(second / "request.json", json.dumps(request))
        self.reference(second, receipt, 125)
        self.write(second / "pbs.log", "step:125 - val-core/math/acc/mean@1:0.5\n")
        self.parent("reuse_job", {"random": second})
        collector = m.Collector(self.root, self.root / "analysis", include_pbs=True)
        def pbs(root, jid):
            return dict(job_id=jid, job_state="F", Exit_status=0,
                        resources_used={"walltime": "01:00:00"}, Resource_List={"ngpus": 8})
        with patch.object(m, "pbs_job", side_effect=pbs), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = collector.run()
        self.assertEqual(len(collector.validations), 1)
        self.assertEqual(len(collector.validations[0]["consumer_jobs"]), 2)
        self.assertEqual(summary["shared_backfill_accounting"]["allocated_gpu_hours_known"], 16)
        self.assertEqual(summary["roles"]["random_continuation"]["allocated_gpu_hours_known"], 0)
        with (collector.output / "job_timeline.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2)

    def test_partial_or_tampered_outputs_are_not_verified(self):
        _, _, output = self.evaluation()
        output.write_text(output.read_text().replace('"answer"', '"tampered"', 1))
        collector, summary = self.collect()
        self.assertFalse(collector.validations[0]["verified_complete"])
        self.assertEqual(summary["setups"]["random"]["backfilled_validation_steps"], [])
        self.assertIn("output", collector.validations[0]["error"])

    def test_1005_rows_even_with_matching_sha_are_incomplete(self):
        folder, receipt_path, output = self.evaluation()
        output.write_text("\n".join(output.read_text().splitlines()[:1005]) + "\n")
        receipt = json.loads(receipt_path.read_text())
        receipt["outputs"].update(rows=1005, sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        receipt_path.write_text(json.dumps(receipt))
        self.reference(folder, receipt_path, 125)
        collector, _ = self.collect()
        self.assertFalse(collector.validations[0]["verified_complete"])

    def test_different_protocol_at_same_step_is_not_averaged_or_deduplicated(self):
        self.evaluation(tag="one", protocol="v1")
        self.evaluation(tag="two", protocol="v2")
        collector, _ = self.collect()
        self.assertEqual(len(collector.validations), 2)
        self.assertTrue(all(r["verified_complete"] for r in collector.validations))
        self.assertEqual(len({r["protocol"] for r in collector.validations}), 2)

    def test_receipt_wrong_checkpoint_is_rejected(self):
        folder, path, _ = self.evaluation()
        receipt = json.loads(path.read_text())
        receipt["checkpoint_key"] = "wrong"
        path.write_text(json.dumps(receipt))
        self.reference(folder, path, 125)
        collector, _ = self.collect()
        self.assertIn("checkpoint digest", collector.validations[0]["error"])

    def test_journal_corruption_is_rejected_and_cache_text_not_archived(self):
        _, path, _ = self.evaluation()
        receipt = json.loads(path.read_text())
        manifest = Path(receipt["origin"]["journal"]["path"]) / "manifest.json"
        manifest.write_text('{}')
        collector, _ = self.collect()
        self.assertFalse(collector.validations[0]["verified_complete"])
        with tarfile.open(collector.output.with_suffix(".tar.gz")) as archive:
            self.assertFalse(any(name.endswith("batch_000001.json") for name in archive.getnames()))
            self.assertTrue(any(name.endswith("progress.json") for name in archive.getnames()))

    def test_nonfinite_receipt_is_reported_without_breaking_collection(self):
        folder, path, _ = self.evaluation()
        receipt = json.loads(path.read_text())
        receipt["metrics"][m.VAL_CORE[0]] = float("nan")
        path.write_text(json.dumps(receipt))
        self.reference(folder, path, 125)
        collector, _ = self.collect()
        self.assertFalse(collector.validations[0]["verified_complete"])
        self.assertIn("nonfinite", collector.validations[0]["error"])

    def test_console_only_validation_remains_distinct_from_complete_receipt(self):
        HistoryTests.job(self, "random_single", "legacy_train", "1.pbs1",
                        "step:25 - val-core/math/acc/mean@1:0.4\n")
        collector, summary = self.collect()
        self.assertEqual(collector.validations[0]["verification"], "console_reported")
        self.assertEqual(summary["setups"]["random"]["validation_steps"], [25])
        self.assertEqual(summary["setups"]["random"]["verified_validation_steps"], [])

    def test_top_level_protocol_cannot_diverge_from_signed_request(self):
        folder, _, _ = self.evaluation()
        req = json.loads((folder / "request.json").read_text())
        req["validation_semantics"]["verifier"] = "changed"
        (folder / "request.json").write_text(json.dumps(req))
        collector, _ = self.collect()
        self.assertFalse(collector.validations[0]["verified_complete"])

    def test_formal_validation_survives_pruned_checkpoint_with_phase_proof(self):
        folder, path, output = self.evaluation()
        req = json.loads((folder / "request.json").read_text())
        req["role"] = req["contract"]["role"] = "formal"
        req["source"]["record"]["step"] = req["contract"]["source"]["record"]["step"] = 120
        req["identity"] = m.identity(req["contract"])
        (folder / "request.json").write_text(json.dumps(req))
        # Formal jobs own their PBS allocation, without a backfill envelope.
        shutil.rmtree(self.root / "outputs/multidomain/validation_backfills")
        self.write(folder / "job_id.txt", "123.pbs1")
        collector, summary = self.collect()
        self.assertTrue(collector.validations[0]["verified_complete"])
        self.assertFalse(collector.validations[0]["is_backfill"])
        self.assertEqual(collector.validations[0]["checkpoint_binding"], "phase_receipt_sha256")
        self.assertEqual(summary["setups"]["random"]["verified_validation_steps"], [125])

    def test_same_identity_conflicting_receipts_are_not_selected_by_filename(self):
        folder, path, _ = self.evaluation()
        changed = json.loads(path.read_text())
        changed["metrics"][m.VAL_CORE[0]] = 0.9
        self.write(path.parent / "duplicate.json", json.dumps(changed))
        collector, _ = self.collect()
        self.assertEqual(len(collector.validations), 2)
        self.assertTrue(all(r["verification"] == "conflicting_receipts" for r in collector.validations))
        self.assertTrue(all(not r["verified_complete"] for r in collector.validations))

    def test_relocated_archive_resolves_only_its_output_tree(self):
        self.evaluation()
        with tempfile.TemporaryDirectory() as destination:
            moved = Path(destination)
            shutil.copytree(self.root / "outputs", moved / "outputs")
            collector = m.Collector(moved, moved / "analysis", include_pbs=False)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                collector.run()
            self.assertTrue(collector.validations[0]["verified_complete"])
            self.assertTrue(all(p.is_relative_to(moved) for p in collector.generation_scan))


class OperationsStorageTests(unittest.TestCase):
    setUp = HistoryTests.setUp
    write = HistoryTests.write
    collect = HistoryTests.collect

    def checkpoint(self, kind="random", step=150):
        run = self.root / "outputs/multidomain" / (m.BASE + m.RUNS[kind + "_continuation"])
        cp = run / "lineages/lineage/checkpoints" / ("global_step_" + str(step))
        contents = {k: dict(path=k + "/dist_ckpt", format="megatron_dist_checkpoint")
                    for k in ("model", "optimizer", "lr_scheduler", "rng_state")}
        for item in contents.values():
            self.write(cp / "actor" / item["path"] / "__0_0.distcp", "tensor payload never opened")
            self.write(cp / "actor" / item["path"] / ".metadata", "binary metadata")
        self.write(cp / "actor/ckpt_contents.json", json.dumps(dict(schema_version=2, global_step=step,
                   world_size=8, contents=contents)))
        self.write(cp / "actor/transformer_config.json", '{}')
        self.write(cp / "data.pt", 'dataloader binary')
        if kind == "domain":
            self.write(cp / "sampling.json", '{}')
        inventory = {}
        for path in cp.rglob("*"):
            if path.is_file():
                stat = path.stat()
                inventory[str(path.relative_to(cp))] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        record = dict(path=str(cp), step=step, world_size=8, lineage="lineage", files=inventory)
        self.write(cp / "COMPLETE_CONTINUATION.json", json.dumps(record))
        return run, cp, record

    def test_full_checkpoint_stat_checks_never_open_tensor_payloads(self):
        _, _, record = self.checkpoint()
        real_open = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn(path.suffix, (".pt", ".distcp"))
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            collector, _ = self.collect()
        row = collector.checkpoint_storage[0]
        self.assertEqual(row["status"], "FULL_STATE_INVENTORY_MATCH")
        self.assertTrue(row["model_inventory_matches"])
        self.assertTrue(row["optimizer_inventory_matches"])
        with tarfile.open(collector.output.with_suffix(".tar.gz")) as bundle:
            self.assertFalse(any(name.endswith((".pt", ".distcp")) for name in bundle.getnames()))

    def test_missing_optimizer_is_distinct_from_missing_model_shard(self):
        _, cp, _ = self.checkpoint()
        (cp / "actor/optimizer/dist_ckpt/__0_0.distcp").unlink()
        (cp / "data.pt").unlink()
        collector, _ = self.collect()
        row = collector.checkpoint_storage[0]
        self.assertEqual(row["status"], "MODEL_INVENTORY_MATCH_FULL_STATE_UNAVAILABLE")
        self.assertFalse(row["optimizer_inventory_matches"])
        (cp / "actor/model/dist_ckpt/__0_0.distcp").unlink()
        collector = m.Collector(self.root, self.root / "again", include_pbs=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            collector.run()
        self.assertFalse(collector.checkpoint_storage[0]["model_inventory_matches"])
        self.assertEqual(collector.checkpoint_storage[0]["status"], "INCOMPLETE_OR_CHANGED")

    def test_legacy_weights_without_recorded_inventory_are_unverified(self):
        _, cp, _ = self.checkpoint()
        (cp / "COMPLETE_CONTINUATION.json").unlink()
        collector, _ = self.collect()
        row = collector.checkpoint_storage[0]
        self.assertEqual(row["status"], "PRESENT_UNVERIFIED")
        self.assertGreater(row["components"]["model"]["present_payload_files"], 0)
        self.assertFalse(row["model_inventory_matches"])

    def test_symlinked_model_shard_and_stale_inventory_do_not_pass(self):
        _, cp, _ = self.checkpoint()
        shard = cp / "actor/model/dist_ckpt/__0_0.distcp"
        external = self.write(self.root / "elsewhere.distcp", shard.read_text())
        shard.unlink()
        shard.symlink_to(external)
        collector, _ = self.collect()
        self.assertFalse(collector.checkpoint_storage[0]["model_inventory_matches"])
        self.assertIn("Symlink", str(collector.checkpoint_storage[0]["file_failures"]))

    def test_archive_payload_survives_deleted_origin_and_is_collected(self):
        run, cp, record = self.checkpoint("domain", 100)
        key = m.checkpoint_identity(record)
        directory = run / "checkpoint_archive/objects" / key
        payload = directory / "payload/global_step_100"
        shutil.copytree(cp, payload, copy_function=os.link)
        obj = dict(kind="domain", key=key, original=record, payload=str(payload), mode="full",
                   step=100, lineage="lineage", inventory=record["files"], trusted_inventory=True, holds=["rescue"])
        self.write(directory / "manifest.json", json.dumps(obj))
        self.write(run / "checkpoint_archive/index.json", json.dumps(dict(cohorts={"v": dict(overall={"key": key, "step": 100})})))
        shutil.rmtree(cp)
        collector, _ = self.collect()
        row = collector.checkpoint_storage[0]
        self.assertEqual(row["status"], "FULL_STATE_INVENTORY_MATCH")
        self.assertEqual(row["archive_holds"], ["rescue"])
        self.assertEqual(collector.best_archives["domain"]["objects"][0]["key"], key)
        with tarfile.open(collector.output.with_suffix(".tar.gz")) as bundle:
            names = bundle.getnames()
            self.assertTrue(any(x.endswith("manifest.json") and "/checkpoint_archive/" in x for x in names))
            self.assertFalse(any("/payload/" in x and x.endswith(".distcp") for x in names))

    def test_reduced_archive_reports_weights_without_claiming_full_state(self):
        run, cp, record = self.checkpoint()
        key = m.checkpoint_identity(record)
        directory = run / "checkpoint_archive/objects" / key
        payload = directory / "payload/global_step_150"
        shutil.copytree(cp, payload, copy_function=os.link)
        for name in ("optimizer", "lr_scheduler", "rng_state"):
            shutil.rmtree(payload / "actor" / name)
        (payload / "data.pt").unlink()
        self.write(directory / "manifest.json", json.dumps(dict(kind="random", key=key, original=record,
            payload=str(payload), mode="weights", step=150, trusted_inventory=True)))
        collector, _ = self.collect()
        row = next(x for x in collector.checkpoint_storage if x["archive_mode"] == "weights")
        self.assertEqual(row["status"], "MODEL_INVENTORY_MATCH_FULL_STATE_UNAVAILABLE")
        self.assertFalse(row["optimizer_inventory_matches"])

    def test_read_only_lock_probe_missing_and_held(self):
        lock = self.root / "does_not_exist/lock"
        self.assertEqual(m.lock_observation(lock), dict(held=False, missing=True))
        self.assertFalse(lock.parent.exists())
        lock = self.write(self.root / "lease.lock", "")
        with lock.open("rb") as handle:
            m.fcntl.flock(handle, m.fcntl.LOCK_EX | m.fcntl.LOCK_NB)
            self.assertTrue(m.lock_observation(lock)["held"])
        self.assertFalse(m.lock_observation(lock)["held"])

    def test_expired_supervisor_and_running_pbs_remain_separate(self):
        run, _, _ = self.checkpoint()
        folder = HistoryTests.job(self, "random_continuation", "formal_a", "123.pbs1", "step:150 - actor/pg_loss:0.1\n")
        self.write(run / "control/state.json", json.dumps(dict(operation="op", phase="ACTIVE", controller_status="RUNNING",
            heartbeat_utc="2020-01-01T00:00:00+00:00", deadline_utc="2020-01-02T00:00:00+00:00",
            active=dict(job_id="123.pbs1", folder=str(folder)))))
        self.write(run / "control/operations/op/plan.json", '{}')
        calls = []
        def query(argv, *args, **kwargs):
            calls.append(argv)
            self.assertEqual(argv[0], "qstat" if argv[0] != "git" else "git")
            return dict(returncode=0, stdout=json.dumps({"Jobs": {"123.pbs1": dict(job_state="R", Exit_status=271,
                Resource_List={"nodect": 2})}}), stderr="")
        collector = m.Collector(self.root, self.root / "analysis", include_pbs=True)
        with patch.object(m, "command", side_effect=query), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            collector.run()
        op = collector.operations["setups"]["random"]
        self.assertTrue(op["deadline"]["passed"])
        self.assertFalse(op["controller_observed_running"])
        self.assertEqual(op["jobs"]["active"]["pbs"]["job_state"], "R")
        self.assertFalse(op["jobs"]["active"]["pbs"]["terminal_state_observed"])
        self.assertEqual(collector.continuation["random"]["state.json"]["phase"], "ACTIVE")
        self.assertTrue(all(c[0] in ("qstat", "git") for c in calls))

    def test_broken_archive_does_not_hide_other_setup_or_training(self):
        for setup in ("random", "domain"):
            run, _, _ = self.checkpoint(setup)
            if setup == "domain":
                self.write(run / "checkpoint_archive/objects/bad/manifest.json", '{"broken":')
        collector, _ = self.collect()
        self.assertEqual(len(collector.checkpoint_storage), 2)
        self.assertTrue(collector.best_archives["domain"]["errors"])
        self.assertEqual(collector.best_archives["random"]["errors"], [])


if __name__ == "__main__":
    unittest.main()
