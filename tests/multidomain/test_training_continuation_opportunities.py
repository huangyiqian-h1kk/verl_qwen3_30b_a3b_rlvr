"""Fault tests for running-single admission and repeated transactional handoffs."""
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
from unittest.mock import patch

from test_training_continuation import Harness, FakePBS, checkpoint, FUTURE, PAST
from d4_training_continuation import cli, runtime, opportunity_upgrade as upgrade
from d4_training_continuation.common import digest, files, now, paths, read, sha, write
from d4_training_continuation.controller import Controller
from d4_training_continuation import checkpoints as ck
from d4_training_continuation.opportunities import Opportunities, POLICY, authorized, external_interruption
from d4_training_continuation.scheduler import Unknown


class OpportunityHarness(Harness):
    def factory(self, plan, role, nodes, authority, deadline):
        made = super().factory(plan, role, nodes, authority, deadline)
        path = Path(made["folder"]) / "request.json"
        req = read(path)
        req["control_dir"] = str(paths(self.root, plan["kind"])["control"])
        if role == "candidate" and plan.get("opportunity_id"):
            req["opportunity_id"] = plan["opportunity_id"]
            req["contract"]["opportunity_id"] = plan["opportunity_id"]
            req["identity"] = plan["opportunity_id"]
        write(path, req)
        return made

    def enabled(self, kind="random", promoted=False):
        c = self.controller(kind)
        c.plan["opportunistic"] = dict(POLICY)
        write(c.control / "operations/synthetic/plan.json", c.plan)
        c.save(phase="ACTIVE" if promoted else "SINGLE_ONLY", promoted=promoted,
               migration_complete=True, heartbeat_utc=now())
        self.pbs.rows[c.state["active"]["job_id"]]["Resource_List"] = {"nodect": 1}
        return c

    def due(self, c):
        poll = c.state.get("opportunity_poll")
        if poll and poll.get("at_utc"):
            poll = dict(poll, at_utc=(datetime.now(timezone.utc) - timedelta(seconds=301)).isoformat())
            c.save(opportunity_poll=poll)

    def admit(self, c):
        self.pbs.free = 2
        c.tick()
        self.assertEqual(c.state["opportunity_poll"]["reason"], "waiting_second_capacity_observation")
        self.due(c); c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "SUBMITTING")
        c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "QUALIFYING")
        return c.state["candidate"]

    def ready(self, c):
        # Existing fixture writes all five actual-shaped audit phases. Candidate
        # already exists, so its leading tick does not submit a second one.
        return self.qualify(c)

    def finish_handoff(self, c):
        self.ready(c)
        c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "STOPPING_SINGLE")
        c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "CATCH_UP")
        a = c.state["opportunity"]
        req = read(Path(a["job"]["folder"]) / "request.json")
        cp = checkpoint(Path(req["run_dir"]) / "checkpoints/global_step_30", 30, 16, req["lineage"], c.kind)
        write(Path(a["job"]["folder"]) / "matched.json", dict(identity=req["identity"],
              execution_id="exec1", job_id=a["job"]["job_id"], checkpoint=cp, evidence={}))
        c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "COMMITTED")
        return cp


class AdmissionTests(OpportunityHarness):
    def test_running_single_two_fresh_observations_submits_without_qdel(self):
        c = self.enabled(); single = copy.deepcopy(c.state["active"])
        self.admit(c)
        self.assertEqual(c.state["active"], single)
        self.assertEqual(len(self.pbs.submissions), 1)
        self.assertEqual(self.pbs.submissions[0]["expected_nodes"], 2)
        self.assertEqual(self.pbs.cancellations, [])
        for _ in range(5): c.tick()
        self.assertEqual(len(self.pbs.submissions), 1)

    def test_one_free_slot_does_not_upgrade_but_does_not_stop_training(self):
        c = self.enabled(); self.pbs.free = 1
        c.tick(); self.due(c); c.tick()
        self.assertFalse(c.state.get("opportunity"))
        self.assertEqual(self.pbs.cancellations, [])

    def test_streak_resets_on_unknown_or_busy_and_query_is_throttled(self):
        c = self.enabled(); self.pbs.free = 2; c.tick()
        count = self.pbs.capacity_calls
        c.tick(); c.tick()
        self.assertEqual(self.pbs.capacity_calls, count)
        self.pbs.free = None; self.due(c); c.tick()
        self.assertEqual(c.state["opportunity_poll"]["consecutive"], 0)
        self.pbs.free = 2; self.due(c); c.tick()
        self.assertEqual(c.state["opportunity_poll"]["consecutive"], 1)

    def test_queue_or_dual_does_not_trigger_upgrade(self):
        c = self.enabled(); self.pbs.free = 20
        for state, node in [("Q", 1), ("H", 1), ("R", 2)]:
            self.pbs.rows[c.state["active"]["job_id"]].update(job_state=state, Resource_List={"nodect": node})
            self.due(c)
            c.tick(); self.due(c); c.tick()
            self.assertFalse(c.state.get("opportunity"))

    def test_old_root_271_latch_rearms_only_after_confirmed_external_interruption(self):
        c = self.enabled("domain")
        job = dict(job_id="10.pbs1", folder=str(self.root / "oldfailed"), job_name="old")
        self.pbs.rows[job["job_id"]] = dict(job_state="F", Exit_status=271, comment="terminated by root@login4")
        c.save(candidate=job, handoff_failed=True, failure_reason="Candidate ended before handoff commit: 271")
        self.admit(c)
        self.assertFalse(c.state["handoff_failed"])

    def test_user_cancel_or_recorded_validation_failure_stays_single_only(self):
        c = self.enabled("domain")
        job = dict(job_id="10.pbs1", folder=str(self.root / "oldfailed"), job_name="old")
        self.pbs.rows[job["job_id"]] = dict(job_state="F", Exit_status=271, comment="terminated by user@login4")
        c.save(candidate=job, handoff_failed=True, failure_reason="Candidate ended before handoff commit: 271")
        self.pbs.free = 10; c.tick(); self.due(c); c.tick()
        self.assertFalse(c.state.get("opportunity"))
        self.pbs.rows[job["job_id"]]["comment"] = "terminated by root@login4"
        write(Path(job["folder"]) / "result.json", {"status": "FAIL", "error": "restore mismatch"})
        self.due(c); c.tick()
        self.assertFalse(c.state.get("opportunity"))

    def test_domain_priority_and_queued_peer_reservation(self):
        d, r = self.enabled("domain"), self.enabled("random")
        self.pbs.free = 2
        with patch("d4_training_continuation.opportunities.active", return_value=True):
            r.tick(); self.due(r); r.tick()
            self.assertEqual(r.state["opportunity_poll"]["reason"], "domain_first")
            self.admit(d)
            self.due(r); r.tick()
            self.assertEqual(r.state["opportunity_poll"]["peer_reserved_slots"], 2)
            self.assertFalse(r.state.get("opportunity"))
            self.pbs.rows[d.state["candidate"]["job_id"]]["job_state"] = "R"
            self.due(r); r.tick()
            self.assertEqual(r.state["opportunity_poll"]["peer_reserved_slots"], 0)

    def test_ambiguous_qsub_is_not_duplicated(self):
        c = self.enabled(); self.pbs.free = 2
        c.tick(); self.due(c); c.tick()
        self.pbs.ambiguous = True
        with self.assertRaises(Unknown): c.tick()
        c = self.reload(c)
        with self.assertRaises(Unknown): c.tick()
        self.assertEqual(len(self.pbs.submissions), 1)
        self.assertEqual(self.pbs.cancellations, [])

    def test_expired_admission_never_calls_qsub(self):
        c = self.enabled(); self.pbs.free = 2
        c.tick(); self.due(c); c.tick(); c.save(deadline_utc=PAST)
        c.tick()
        self.assertEqual(self.pbs.submissions, [])
        self.assertEqual(c.state["opportunity"]["failure_class"], "deadline")

    def test_regular_retry_during_qualification_remains_single(self):
        c = self.enabled(promoted=True); self.admit(c)
        self.pbs.rows[c.state["active"]["job_id"]].update(job_state="F", Exit_status=271)
        self.pbs.free = 20; c.tick()
        self.assertEqual(self.pbs.submissions[-1]["role"], "formal")
        self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], 1)


class HandoffTests(OpportunityHarness):
    def test_protection_survives_crash_and_single_progress_before_fence(self):
        c = self.enabled(); self.admit(c); self.ready(c)
        save = c.save
        def interrupted(**changes):
            if (changes.get("opportunity") or {}).get("phase") == "STOPPING_SINGLE":
                raise OSError("injected state-write interruption")
            return save(**changes)
        with patch.object(c, "save", side_effect=interrupted):
            with self.assertRaises(OSError): c.tick()
        checkpoint(c.p["single"] / "checkpoints/global_step_31", 31)
        (c.p["single"] / "checkpoints/latest_checkpointed_iteration.txt").write_text("31")
        c = self.reload(c); c.tick(); c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "CATCH_UP")
        self.assertEqual(c.state["opportunity"]["fallback"]["step"], 31)
        self.assertEqual(c.state["opportunity"]["release"]["target"], 31)

    def test_pin_rename_crash_recovers_even_after_original_is_pruned(self):
        cp = checkpoint(self.root / "original/global_step_30", 30, 8, "formal")
        destination = self.root / "pin"
        def interrupted(path, data):
            if Path(path).name == "source.json":
                raise OSError("injected manifest interruption")
            return write(path, data)
        with patch.object(ck, "write", side_effect=interrupted):
            with self.assertRaises(OSError): ck.pin(cp, destination)
        shutil.rmtree(cp["path"])
        recovered = ck.pin(cp, destination)
        self.assertEqual(ck.validate(recovered)["files"], cp["files"])
        self.assertEqual(ck.pin(cp, destination), recovered)
        with self.assertRaisesRegex(RuntimeError, "another checkpoint"):
            ck.pin(dict(cp, lineage="foreign"), destination)

    def test_intentional_fence_without_exit_code_can_roll_back(self):
        c = self.enabled(); self.admit(c); self.ready(c); c.tick()
        original = c.state["active"]
        self.pbs.rows[original["job_id"]] = dict(job_state="F", Exit_status=None)
        self.pbs.rows[c.state["candidate"]["job_id"]].update(job_state="F", Exit_status=271,
                                                          comment="terminated by root@login4")
        c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "FAILED")
        self.assertNotEqual(c.state["active"], original)
        self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], 1)

    def test_interrupted_rollback_before_qsub_is_resumed(self):
        c = self.enabled(); self.admit(c); self.ready(c); c.tick()
        self.pbs.rows[c.state["candidate"]["job_id"]].update(job_state="F", Exit_status=271,
                                                          comment="terminated by root@login4")
        with patch.object(c, "monitor_formal", side_effect=OSError("temporary pin failure")):
            with self.assertRaises(OSError): c.tick()
        self.assertIsNone(c.state["active"])
        self.assertTrue(c.state["rollback_required"])
        c = self.reload(c); c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "FAILED")
        self.assertIsNotNone(c.state["active"])
        self.assertEqual(len(self.pbs.submissions), 2)
        self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], 1)

    def test_external_kill_cools_down_then_creates_a_new_lineage(self):
        c = self.enabled("domain"); job = self.admit(c)
        first = c.state["opportunity"]["id"]
        self.pbs.rows[job["job_id"]].update(job_state="F", Exit_status=271, comment="terminated by root@login4")
        c.tick()
        self.assertEqual(c.state["opportunity"]["failure_class"], "external")
        self.due(c); c.tick()
        self.assertEqual(len(self.pbs.submissions), 1)
        c.state["opportunity"]["retry_after_utc"] = PAST; c.save(opportunity=c.state["opportunity"])
        self.due(c); self.admit(c)
        self.assertNotEqual(c.state["opportunity"]["id"], first)
        self.assertEqual(self.pbs.cancellations, [])

    def test_changed_qualification_evidence_cannot_cancel_single(self):
        c = self.enabled(); self.admit(c); self.ready(c)
        single = c.state["active"]["job_id"]
        path = Path(c.state["candidate"]["folder"]) / "C/qualification.json"
        row = read(path); row["status"] = "FAIL"; write(path, row)
        c.tick(); c.tick()
        self.assertEqual(c.state["opportunity"]["failure_class"], "validation")
        self.assertNotIn(single, self.pbs.cancellations)
        self.due(c); c.tick()
        self.assertEqual(len(self.pbs.submissions), 1)

    def test_kill_after_fence_rolls_back_protected_single_once(self):
        c = self.enabled(); self.admit(c); self.ready(c); c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "STOPPING_SINGLE")
        self.pbs.rows[c.state["candidate"]["job_id"]].update(job_state="F", Exit_status=271, comment="terminated by root@login4")
        c.tick(); c = self.reload(c); c.tick()
        self.assertEqual(len(self.pbs.submissions), 2)
        req = self.pbs.submissions[-1]
        self.assertEqual(req["role"], "formal")
        self.assertEqual(req["expected_nodes"], 1)
        self.assertEqual(req["source"]["record"]["step"], 30)

    def test_success_changes_authority_and_survives_commit_state_crash(self):
        c = self.enabled(promoted=True); self.admit(c); cp = self.finish_handoff(c)
        a = copy.deepcopy(c.state["opportunity"])
        commit_path = c.control / "opportunities" / a["id"] / "commit.json"
        commit = read(commit_path)
        c.save(opportunity=dict(a, phase="CATCH_UP"), active=commit["previous_job"], authority=commit["previous_authority"])
        c = self.reload(c); c.tick()
        self.assertEqual(c.state["active"], a["job"])
        self.assertEqual(c.state["authority"]["lineage"], cp["lineage"])
        self.assertEqual(len(self.pbs.submissions), 1)

    def test_promoted_then_single_then_new_upgrade_has_separate_commit(self):
        c = self.enabled(promoted=True); self.admit(c); self.finish_handoff(c)
        first = copy.deepcopy(c.state["opportunity"])
        old_commit = c.control / "opportunities" / first["id"] / "commit.json"
        original_bytes = old_commit.read_bytes()
        self.pbs.rows[c.state["active"]["job_id"]].update(job_state="F", Exit_status=271)
        self.pbs.free = 0; c.tick()
        formal = c.state["active"]
        req = read(Path(formal["folder"]) / "request.json")
        checkpoint(Path(req["run_dir"]) / "checkpoints/global_step_31", 31, 8, req["lineage"])
        self.pbs.rows[formal["job_id"]].update(job_state="R", Resource_List={"nodect": 1})
        self.due(c); self.admit(c)
        self.assertNotEqual(c.state["opportunity"]["lineage"], first["lineage"])
        self.assertEqual(c.state["active"], formal)
        self.assertEqual(old_commit.read_bytes(), original_bytes)

    def test_new_attempt_authorization_requires_current_id_and_folder(self):
        c = self.enabled(promoted=True); self.admit(c)
        req = read(Path(c.state["candidate"]["folder"]) / "request.json")
        self.assertTrue(authorized(c.state, req))
        self.assertFalse(authorized(c.state, dict(req, opportunity_id="retired")))
        req["control_dir"] = str(c.control)
        cp = runtime.select_source(req, "A")
        self.assertEqual(cp["step"], 30)
        self.assertEqual(cp["world_size"], 8)

    def test_unknown_candidate_keeps_single_and_never_marks_failed(self):
        c = self.enabled(); self.admit(c)
        self.pbs.unknown.add(c.state["candidate"]["job_id"])
        with self.assertRaises(Unknown): c.tick()
        self.assertEqual(c.state["opportunity"]["phase"], "QUALIFYING")
        self.assertEqual(self.pbs.cancellations, [])

    def test_abort_query_error_does_not_disable_single_restart(self):
        c = self.enabled(); self.admit(c)
        Opportunities(c).failure("test validation failure")
        self.pbs.unknown.add(c.state["candidate"]["job_id"])
        self.pbs.rows[c.state["active"]["job_id"]].update(job_state="F", Exit_status=271)
        with self.assertRaises(Unknown): c.tick()
        self.assertEqual(self.pbs.submissions[-1]["role"], "formal")
        self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], 1)


class UpgradeTests(OpportunityHarness):
    def staged(self, kind="random"):
        from test_training_continuation import IntegrationTests
        c = self.enabled(kind)
        plan = IntegrationTests.template_plan(self, kind)
        target = c.control / "operations/synthetic/template"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(plan["template"], target)
        plan.update(template=str(target), template_files=files(target))
        write(c.control / "operations/synthetic/plan.json", plan)
        return c, plan

    def test_upgrade_stages_new_snapshot_keeps_original_and_deadline(self):
        c, plan = self.staged()
        before = files(plan["template"])
        state = (c.control / "state.json").read_bytes()
        path, old, new = upgrade.prepare(self.root, "random")
        self.assertNotEqual(new["template"], old["template"])
        self.assertEqual(new["opportunistic"], POLICY)
        self.assertEqual(files(plan["template"]), before)
        self.assertEqual((c.control / "state.json").read_bytes(), state)
        self.assertEqual(read(path), plan)
        write(path, new)
        _, same, again = upgrade.prepare(self.root, "random")
        self.assertEqual(same, again)

    def test_unknown_modified_runtime_blocks_before_publish(self):
        c, plan = self.staged()
        f = Path(plan["template"]) / "runtime/d4_training_continuation/controller.py"
        f.write_text(f.read_text() + "\n# unknown source edit\n")
        plan["template_files"] = files(plan["template"])
        path = c.control / "operations/synthetic/plan.json"; write(path, plan)
        with self.assertRaisesRegex(RuntimeError, "Unsupported frozen"):
            upgrade.prepare(self.root, "random")
        self.assertEqual(read(path), plan)

    def test_foreign_host_is_rejected_before_any_stop(self):
        d, _ = self.staged("domain")
        r, _ = self.staged("random")
        for c in [d, r]: c.save(host="foreign-login")
        with patch.object(upgrade, "active", return_value=True), patch.object(upgrade, "stop_and_acquire") as stop:
            with self.assertRaisesRegex(RuntimeError, "controller host"):
                upgrade.enable(self.root)
            stop.assert_not_called()


class UpgradeProcessTests(Harness):
    def test_real_frozen_owner_upgrade_binds_both_controllers_without_training_changes(self):
        import getpass
        import json
        import os
        import subprocess
        import sys
        import test_training_continuation_launch as launch_fixtures
        from test_training_continuation import IntegrationTests
        from d4_training_continuation import cancellation_repair
        self.children = []
        self.addCleanup(lambda: launch_fixtures.LaunchTests.stop_children(self))
        c, planfile, original, entry = launch_fixtures.LaunchTests.fixture(self)
        c.save(phase="SINGLE_ONLY", old_candidate_retired=True)
        d = self.controller("domain")
        plan = IntegrationTests.template_plan(self, "domain")
        target = d.control / "operations/synthetic/template"
        shutil.move(plan["template"], target)
        plan.update(template=str(target), template_files=files(target))
        write(d.control / "operations/synthetic/plan.json", plan)
        d.save(phase="SINGLE_ONLY", migration_complete=True)
        d.p["old_control"].mkdir(parents=True, exist_ok=True)
        write(d.p["monitor"] / "state.json", {"status": "RUNNING"})
        rows = read(self.root / "pbs_rows.json")
        rows[c.state["active"]["job_id"]]["Resource_List"] = {"nodect": 1}
        job = d.state["active"]
        rows[job["job_id"]] = dict(Job_Owner=getpass.getuser()+"@login", queue="R9920261000",
             Job_Name=job["job_name"], Submit_arguments=str(Path(job["folder"])/"stage.pbs"),
             job_state="R", Resource_List={"nodect": 2})
        write(self.root / "pbs_rows.json", rows)
        original_files = files(original)
        before = {k: read(paths(self.root, k)["control"] / "state.json") for k in ("domain", "random")}
        with (c.control / "controller.log").open("a") as out:
            old = subprocess.Popen([sys.executable, "-I", str(entry), "_run", "--root", str(self.root),
                                    "--kind", "random"], stdout=out, stderr=subprocess.STDOUT)
        self.children.append(old)
        launch_fixtures.LaunchTests.wait_for(self, lambda: (c.control / "legacy_identity.json").exists(), c.control)
        identity = read(c.control / "legacy_identity.json")
        real_spawn = cli.spawn
        def spawn(root, kind):
            process = real_spawn(root, kind)
            if process is not None: self.children.append(process)
            return process
        legacy = copy.deepcopy(upgrade.LEGACY)
        legacy["cli.py"].add(sha(original / "runtime/d4_training_continuation/cli.py"))
        with patch.object(upgrade, "LEGACY", legacy), patch.object(cli, "spawn", side_effect=spawn), \
             patch.object(cancellation_repair, "_process_identity", return_value=(identity["uid"], identity["argv"])):
            result = upgrade.enable(self.root)
            self.assertEqual(result["status"], "OPPORTUNISTIC_MONITORS_RUNNING",
                             repr(result) + "\n" + "\n".join((paths(self.root,k)["control"] / "controller.log").read_text()
                                                            for k in ("domain", "random")))
            self.assertEqual(old.wait(timeout=3), 0)
            for kind in ("domain", "random"):
                state = read(paths(self.root, kind)["control"] / "state.json")
                self.assertEqual(state["deadline_utc"], before[kind]["deadline_utc"])
                self.assertEqual(state["active"], before[kind]["active"])
                self.assertIn("d4_training_continuation.opportunities", state["runtime_binding"]["modules"])
            repeated = upgrade.enable(self.root)
            self.assertTrue(all(not r["controller_restart_requested"] for r in repeated["setups"].values()))
        self.assertEqual(files(original), original_files)
        calls = [json.loads(x) for x in (self.root / "pbs_calls.jsonl").read_text().splitlines()]
        self.assertFalse(any(row[0] in ("qsub", "qdel") for row in calls))
