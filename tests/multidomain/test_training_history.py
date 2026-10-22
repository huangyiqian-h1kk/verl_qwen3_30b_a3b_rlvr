"""Regression tests for scientific accounting across interrupted D4 attempts."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
