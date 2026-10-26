import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "scripts/multidomain/finish_training.py"
SPEC = importlib.util.spec_from_file_location("finish_training", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class FinishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "root"
        self.base = self.root / "outputs/multidomain"
        self.base.mkdir(parents=True)
        self.output = self.root / "outputs/analysis/report"
        self.output.mkdir(parents=True)

    def file(self, path, text="x"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def json(self, path, value):
        return self.file(path, json.dumps(value))

    def setup_owner(self, kind):
        p = m.paths(self.root, kind)
        operation = p["control"] / "operations/op"
        template = operation / "template"
        cli = self.file(template / "runtime/d4_training_continuation/cli.py",
                        "from pathlib import Path\n"
                        "def spawn(root, kind):\n    raise AssertionError('spawned')\n"
                        "def daemon(root, kind):\n    raise AssertionError('ran')\n")
        self.json(operation / "plan.json", dict(root=str(self.root), kind=kind, template=str(template),
                  template_files={str(cli.relative_to(template)): m.sha(cli)}))
        self.json(p["control"] / "state.json", dict(operation="op", phase="EXPIRED", controller_status="STOPPED",
                  host="another-login", pid=12345, authority=dict(run=str(p["run"] / "lineages/main"), lineage="main")))
        self.json(p["monitor"] / "state.json", dict(status="STOPPED"))
        return p

    def job(self, kind, jid="123.pbs1", state="R"):
        p = m.paths(self.root, kind)
        folder = p["run"] / "jobs" / jid
        script = self.file(folder / "stage.pbs", "#PBS -N 0390_d4r_cont\n")
        self.file(folder / "job_id.txt", jid)
        return dict(Job_Name="0390_d4r_cont", Job_Owner=m.getpass.getuser() + "@login2", queue=m.QUEUE,
                    job_state=state, Submit_arguments=str(script))

    def test_stop_expired_owners_on_any_host_cancels_both_and_guards_old_resume(self):
        for kind in m.KINDS:
            self.setup_owner(kind)
        rows = {"123.pbs1": self.job("random", "123.pbs1", "H"),
                "124.pbs1": self.job("domain", "124.pbs1", "R")}
        commands = []
        def execute(argv, **kwargs):
            commands.append(argv)
            rows.pop(argv[1])
            return type("Result", (), dict(returncode=0, stdout="", stderr=""))()
        with patch.object(m, "pbs_jobs", side_effect=lambda: dict(rows)), patch.object(m, "command", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            result = m.stop(self.root, self.root, self.output)
        self.assertEqual(result["status"], "STOPPED")
        self.assertEqual(commands, [["qdel", "123.pbs1"], ["qdel", "124.pbs1"]])
        for kind in m.KINDS:
            control = m.paths(self.root, kind)["control"]
            plan = m.read(control / "operations/op/plan.json")
            self.assertEqual(plan["retirement_guard"], m.VERSION)
            namespace = {}
            exec((Path(plan["template"]) / "runtime/d4_training_continuation/cli.py").read_text(), namespace)
            namespace["daemon"](self.root, kind)
            namespace["spawn"](self.root, kind)
        # Repeating the command adopts the same persistent guard, no new qdel.
        with patch.object(m, "pbs_jobs", return_value={}), patch.object(m, "command", side_effect=AssertionError("mutation")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.stop(self.root, self.root, self.output)["status"], "STOPPED")

    def test_unknown_pbs_never_becomes_success_or_qdel(self):
        for kind in m.KINDS:
            self.setup_owner(kind)
        with patch.object(m, "pbs_jobs", side_effect=RuntimeError("timeout")), patch.object(m, "command", side_effect=AssertionError("qdel")):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                m.stop(self.root, self.root, self.output)
        self.assertEqual(m.read(self.output / "stop.json")["status"], "INCOMPLETE")

    def test_remote_live_owner_does_not_signal_local_pid(self):
        p = self.setup_owner("random")
        with self.assertRaisesRegex(RuntimeError, "SAME stop command there"):
            m.signal_owner(p["control"], self.root, "random", self.root)

    def test_writer_lock_blocks_success_after_qdel(self):
        for kind in m.KINDS:
            p = self.setup_owner(kind)
        stage = self.file(p["run"] / "stage.lock", "")
        real = m.wait_lease
        def wait(path, *args, **kwargs):
            if path == stage:
                raise RuntimeError("Writer still alive")
            return real(path, *args, **kwargs)
        with patch.object(m, "pbs_jobs", return_value={}), patch.object(m, "wait_lease", side_effect=wait):
            with self.assertRaisesRegex(RuntimeError, "Writer still alive"):
                m.stop(self.root, self.root, self.output)
        self.assertEqual(m.read(self.output / "stop.json")["status"], "INCOMPLETE")

    def test_identity_and_ambiguity_checked_before_cancel(self):
        row = self.job("random")
        self.assertEqual(set(m.owned_jobs(self.root, {"123.pbs1": row})), {"123.pbs1"})
        with self.assertRaisesRegex(RuntimeError, "identity/queue"):
            m.owned_jobs(self.root, {"123.pbs1": dict(row, queue="rt_HF")})
        with self.assertRaisesRegex(RuntimeError, "Unrecorded/ambiguous"):
            m.owned_jobs(self.root, {"999.pbs1": row})
        with self.assertRaisesRegex(RuntimeError, "Unidentified"):
            m.owned_jobs(self.root, {"123.pbs1": dict(row, Submit_arguments="/elsewhere/stage.pbs")})
        self.assertEqual(m.owned_jobs(self.root, {"123.pbs1": dict(row, Job_Owner="someone_else@login1")}), {})
        self.assertEqual(m.owned_jobs(self.root, {"123.pbs1": dict(row, job_state="F")}), {})

    def test_symlink_or_parent_escape_rejected(self):
        elsewhere = self.file(self.root / "outside")
        link = self.root / "link"
        link.symlink_to(elsewhere)
        for path in (link, self.root / "../escape"):
            with self.assertRaises(RuntimeError):
                m.safe(path, self.root)

    def test_modified_template_prevents_cancellation(self):
        for kind in m.KINDS:
            p = self.setup_owner(kind)
        template = p["control"] / "operations/op/template/runtime/d4_training_continuation/cli.py"
        template.write_text("changed")
        with patch.object(m, "pbs_jobs", side_effect=AssertionError("query before fence")):
            with self.assertRaisesRegex(RuntimeError, "Frozen source changed"):
                m.stop(self.root, self.root, self.output)

    def archive(self, kind, step, key, mode="full"):
        p = m.paths(self.root, kind)
        folder = p["run"] / "checkpoint_archive/objects" / key
        cp = folder / "payload" / ("global_step_" + str(step))
        for name in ("actor/model/x.distcp", "actor/model/.metadata", "actor/optimizer/o.distcp", "data.pt"):
            self.file(cp / name, "x" * 8192)
        inventory = {}
        for file in cp.rglob("*"):
            if file.is_file():
                s = file.stat()
                inventory[str(file.relative_to(cp))] = dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns)
        model = {k: v for k, v in inventory.items() if k.startswith("actor/model/")}
        self.json(folder / "manifest.json", dict(key=key, kind=kind, step=step, mode=mode, lineage="main",
                  payload=str(cp), inventory=inventory, model_inventory=model))
        return cp

    def setup_archives(self):
        for kind in m.KINDS:
            p = self.setup_owner(kind)
            a = self.archive(kind, 150, "a" * 64)
            b = self.archive(kind, 125, "b" * 64)
            c = self.archive(kind, 200, "c" * 64)
            self.json(p["run"] / "checkpoint_archive/index.json", dict(cohorts={"protocol": dict(
                overall=dict(key="a" * 64, path=str(a), step=150),
                math=dict(key="b" * 64, path=str(b), step=125))}))

    def test_best_latest_full_and_domain_winner_model_only_proposal(self):
        self.setup_archives()
        protected, selected, errors = m.retention(self.root)
        self.assertEqual(errors, [])
        self.assertTrue(all(s["latest_full_step"] == 200 for s in selected.values()))
        for cp, policy in protected.items():
            cp = Path(cp)
            is_domain_best = cp.name == "global_step_125"
            self.assertEqual(policy["mode"], "model" if is_domain_best else "full")
            self.assertFalse(m.removable_payload(cp / "actor/model/x.distcp", cp, protected))
            self.assertEqual(m.removable_payload(cp / "actor/optimizer/o.distcp", cp, protected), is_domain_best)

    def test_inode_count_and_outside_links_prevent_false_reclaim(self):
        self.setup_archives()
        single = m.paths(self.root, "random")["single"]
        old = self.file(single / "checkpoints/global_step_90/actor/model/old.distcp", "a" * 8192)
        clone = single / "jobs/old/train/source/global_step_90/actor/model/old.distcp"
        clone.parent.mkdir(parents=True)
        os.link(old, clone)
        outside = self.root / "outside-link"
        os.link(old, outside)
        with patch.object(m, "command", side_effect=AssertionError("Not read-only")), contextlib.redirect_stdout(io.StringIO()):
            report = m.inventory(self.root, self.output)
        self.assertEqual(report["errors"], [])
        self.assertGreater(report["totals"]["external_hardlink_inodes"], 0)
        expected = sum((Path(p) / name).stat().st_blocks * 512
                       for p, v in m.retention(self.root)[0].items() if v["mode"] == "model"
                       for name in ("actor/optimizer/o.distcp", "data.pt"))
        self.assertEqual(report["totals"]["proposed_reclaimable_allocated_bytes"], expected)
        outside.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            second = m.inventory(self.root, self.output)
        self.assertEqual(second["totals"]["proposed_reclaimable_allocated_bytes"], expected + old.stat().st_blocks * 512)

    def test_cross_setup_hardlink_is_counted_once_in_union(self):
        self.setup_archives()
        p = self.file(m.paths(self.root, "random")["single"] / "logs/large.log", "x" * 16384)
        target = m.paths(self.root, "domain")["single"] / "logs/link.log"
        target.parent.mkdir(parents=True)
        os.link(p, target)
        with contextlib.redirect_stdout(io.StringIO()):
            report = m.inventory(self.root, self.output)
        self.assertEqual(report["scopes"]["random"]["shared_with_other_scopes_bytes"], p.stat().st_blocks * 512)
        self.assertEqual(report["scopes"]["domain"]["shared_with_other_scopes_bytes"], p.stat().st_blocks * 512)
        summed = sum(s["unique_allocated_bytes"] for s in report["scopes"].values())
        self.assertEqual(summed - report["totals"]["unique_allocated_bytes"], p.stat().st_blocks * 512)

    def test_failed_protection_disables_reclaim_estimate(self):
        self.setup_archives()
        cp = m.paths(self.root, "domain")["run"] / "checkpoint_archive/objects" / ("a" * 64) / "payload/global_step_150"
        (cp / "actor/model/x.distcp").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            report = m.inventory(self.root, self.output)
        self.assertTrue(report["errors"])
        self.assertIsNone(report["totals"]["proposed_reclaimable_allocated_bytes"])

    def test_inventory_does_not_follow_external_symlinks(self):
        self.setup_archives()
        outside = self.root / "external-dir"
        self.file(outside / "huge.distcp", "x" * 8192)
        link = m.paths(self.root, "random")["run"] / "external"
        link.symlink_to(outside, target_is_directory=True)
        with contextlib.redirect_stdout(io.StringIO()):
            report = m.inventory(self.root, self.output)
        self.assertIn(str(link), report["symlinks_not_followed"])
        self.assertTrue((outside / "huge.distcp").exists())


if __name__ == "__main__":
    unittest.main()
