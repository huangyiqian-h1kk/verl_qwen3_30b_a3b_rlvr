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


if __name__ == "__main__":
    unittest.main()
