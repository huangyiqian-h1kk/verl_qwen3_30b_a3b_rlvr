"""CPU fault-injection tests. Synthetic files/PBS; no cluster or model access."""
import copy
from datetime import datetime, timezone
import getpass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from d4_training_continuation import PHASES
from d4_training_continuation import checkpoints as ck, cli, evidence, optimizer_repair, runtime, source
from d4_training_continuation.common import digest, extended_deadline, files, now, paths, read, sha, write
from d4_training_continuation.controller import Controller, authoritative_checkpoint, choose_nodes
from d4_training_continuation.scheduler import PBS, Unknown, render_pbs, retryable, validate_job
from d4_training_continuation.capacity import attributes, free_hosts
from d4_training_continuation import trainer as continuation_trainer

POLICY = "fused-adam-empty-local-group-step-v1"
FUTURE = "2099-01-01T00:00:00+00:00"
PAST = "2000-01-01T00:00:00+00:00"


def checkpoint(path, step, world=8, lineage=None, kind="random", parent=None):
    path = Path(path)
    path.mkdir(parents=True)
    (path / "data.pt").write_bytes(b"synthetic dataloader")
    (path / "actor/dist").mkdir(parents=True)
    (path / "actor/dist/payload.distcp").write_bytes(b"synthetic model and moments")
    spec = dict(schema_version=2, global_step=step, world_size=world,
                contents={k: dict(path="dist", format="megatron_dist_checkpoint")
                          for k in ("model", "optimizer", "lr_scheduler", "rng_state")})
    write(path / "actor/ckpt_contents.json", spec)
    if kind == "domain":
        write(path / "sampling.json", {"plan_sha256": "synthetic-plan"})
    for rank in range(world):
        write(path / "actor" / f"continuation_state_rank_{rank}.json",
              dict(optimizer_audit_policy=POLICY, optimizer="moments", model="weights", lr_scheduler=step))
    record = dict(step=step, world_size=world, path=str(path), lineage=lineage, files=ck.inventory(path))
    if lineage:
        write(path / ck.MARKER, {k: v for k, v in record.items() if k != "path"})
    return record


def topology(world):
    return [dict(rank=i, host="hnode001" if i < 8 else "hnode002", world_size=world,
                 tp=2, pp=2, ep=2, cp=1, dp=world // 4, optimizer_repair="continuation-optimizer-v1") for i in range(world)]


class FakePBS:
    def __init__(self):
        self.rows, self.submissions, self.cancellations = {}, [], []
        self.free = 0
        self.capacity_calls = 0
        self.unknown = set()
        self.ambiguous = False

    def observe(self, job):
        if job["job_id"] in self.unknown:
            raise Unknown("temporary PBS query failure")
        return copy.deepcopy(self.rows[job["job_id"]])

    def capacity(self):
        self.capacity_calls += 1
        return dict(free_slots=self.free)

    def submit(self, folder, name):
        folder = Path(folder)
        if (folder / "job_id.txt").exists():
            jid = (folder / "job_id.txt").read_text()
            return dict(job_id=jid, folder=str(folder), job_name=name)
        if (folder / "qsub_attempted.json").exists():
            raise Unknown("ambiguous previous submission")
        write(folder / "qsub_attempted.json", {"at_utc": now()})
        self.submissions.append(read(folder / "request.json"))
        if self.ambiguous:
            raise Unknown("qsub timeout")
        jid = str(900000 + len(self.submissions)) + ".pbs1"
        (folder / "job_id.txt").write_text(jid)
        self.rows[jid] = dict(job_state="Q", run_count=1, stime="synthetic start")
        return dict(job_id=jid, folder=str(folder), job_name=name)

    def cancel(self, job, journal):
        if self.observe(job)["job_state"] == "F":
            return True
        self.cancellations.append(job["job_id"])
        write(journal, dict(job=job, status="RETURNED"))
        self.rows[job["job_id"]] = dict(job_state="F", Exit_status=143)
        return False


class Harness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pbs = FakePBS()
        self.created = 0

    def factory(self, plan, role, nodes, authority, deadline):
        self.created += 1
        p = paths(self.root, plan["kind"])
        folder = p["run"] / "jobs" / (role + str(self.created))
        folder.mkdir(parents=True)
        lineage = plan["candidate_lineage"] if role == "candidate" else authority["lineage"]
        request = dict(root=str(self.root), kind=plan["kind"], role=role, lineage=lineage,
                       source=authority, expected_nodes=nodes, config={"training": {"train_nodes": nodes}},
                       run_dir=str(p["run"] / "lineages" / lineage), job_dir=str(folder),
                       identity=plan["kind"] + "_candidate", contract={"operation": "synthetic"})
        write(folder / "request.json", request)
        return dict(folder=str(folder), job_name="0390_synthetic")

    def controller(self, kind="random"):
        p = paths(self.root, kind)
        legacy = checkpoint(p["single"] / "checkpoints/global_step_30", 30, kind=kind)
        (p["single"] / "checkpoints/latest_checkpointed_iteration.txt").write_text("30")
        jid = "100.pbs1" if kind == "random" else "101.pbs1"
        old = dict(single=dict(job_id=jid, folder=str(p["single"] / "jobs/old"), job_name="0390_old"),
                   dual=None, deadline_utc=FUTURE)
        self.pbs.rows[jid] = dict(job_state="R")
        plan = dict(kind=kind, root=str(self.root), operation="synthetic", old_state=old,
                    single_lineage=kind + "single", candidate_lineage=kind + "dual")
        write(p["control"] / "operations/synthetic/plan.json", plan)
        state = cli.initial_state(plan, 24)
        state.update(candidate_admitted=True, old_candidate_retired=True)
        write(p["control"] / "state.json", state)
        return Controller(self.root, kind, self.pbs, self.factory)

    def reload(self, c):
        return Controller(self.root, c.kind, self.pbs, self.factory)

    def qualify(self, c, source_step=20):
        c.tick()  # directly submits Q with singles retained
        job = c.state["candidate"]
        folder = Path(job["folder"])
        req = read(folder / "request.json")
        self.pbs.rows[job["job_id"]] = dict(job_state="R", run_count=1, stime="synthetic start")
        write(folder / "allocation.json", dict(execution_id="exec1", run_count=1, stime="synthetic start"))
        previous = checkpoint(folder / "input/global_step_20", source_step, kind=c.kind)
        for name, source_world, target in PHASES:
            phase = folder / name
            phase.mkdir()
            execution = dict(phase=name, identity=req["identity"], execution_id="exec1", token="token" + name,
                             source_world=source_world, target_world=target, driver_pid=100 + ord(name))
            write(phase / "phase.json", execution)
            write(phase / "resume_source.json", previous)
            write(phase / "resume.json", dict(status="PASS", step=previous["step"]))
            write(phase / "topology.json", dict(actor_ranks=topology(target)))
            step = previous["step"] + 1
            cp = checkpoint(Path(req["run_dir"]) / f"checkpoints/global_step_{step}", step,
                            target, req["lineage"], c.kind)
            write(phase / f"step_{step}.json", dict(step=step, weights_synchronized=True,
                  actor_update_completed=True, prompts=256, responses=2048, saved_checkpoint_step=step))
            for rank in range(target):
                write(phase / f"restore_rank_{rank}.json", dict(status="PASS", rank=rank, token=execution["token"],
                      target_world=target, source_world=source_world, full_model_optimizer_load_returned=True,
                      lr_state_equal=True, rng_restored=True))
                write(phase / f"optimizer_load_returned_rank_{rank}.json", dict(status="LOAD_RETURNED", rank=rank,
                      token=execution["token"], world_size=target))
                write(phase / f"audit_binding_rank_{rank}.json", dict(policy=POLICY, rank=rank))
                if source_world == target:
                    write(phase / f"restore_comparison_rank_{rank}.json", dict(status="PASS", changed_sections=[]))
            evidence.phase_pass(req, phase, previous, cp, step)
            previous = cp
        ready = dict(identity=req["identity"], job_id=job["job_id"], execution_id="exec1", checkpoint=cp,
                     evidence={n + "/qualification.json": sha(folder / n / "qualification.json") for n, _, _ in PHASES})
        write(folder / "ready.json", ready)
        return ready

    def handoff(self, c):
        ready = self.qualify(c)
        c.tick()
        self.assertEqual(c.state["phase"], "STOPPING_SINGLE")
        c.tick()
        self.assertEqual(c.state["phase"], "CATCH_UP")
        job = c.state["candidate"]
        req = read(Path(job["folder"]) / "request.json")
        cp = checkpoint(Path(req["run_dir"]) / "checkpoints/global_step_30", 30, 16, req["lineage"], c.kind)
        write(Path(job["folder"]) / "matched.json", dict(identity=req["identity"], job_id=job["job_id"],
              execution_id="exec1", checkpoint=cp, evidence={}))
        c.tick()
        self.assertTrue(c.state["promoted"])
        return cp


class ControllerTests(Harness):
    def test_candidate_queues_without_free_node_gate_or_single_cancellation(self):
        c = self.controller(); self.pbs.free = 0; c.tick()
        self.assertEqual(len(self.pbs.submissions), 1)
        self.assertEqual(self.pbs.submissions[0]["expected_nodes"], 2)
        self.assertEqual(self.pbs.capacity_calls, 0)
        self.assertEqual(self.pbs.cancellations, [])

    def test_pre_handoff_271_latches_single_only_and_never_retries_candidate(self):
        c = self.controller(); c.tick()
        self.pbs.rows[c.state["candidate"]["job_id"]] = dict(job_state="F", Exit_status=271)
        c.tick(); c.tick(); c = self.reload(c); c.tick()
        self.assertTrue(c.state["handoff_failed"])
        self.assertEqual(len(self.pbs.submissions), 1)
        self.pbs.rows[c.state["active"]["job_id"]] = dict(job_state="F", Exit_status=271)
        self.pbs.free = 20
        c.tick()
        self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], 1)
        self.assertEqual(self.pbs.capacity_calls, 0)

    def test_setups_have_independent_outcomes(self):
        domain, random = self.controller("domain"), self.controller("random")
        domain.tick(); random.tick()
        self.pbs.rows[domain.state["candidate"]["job_id"]] = dict(job_state="F", Exit_status=1)
        domain.tick(); random.tick()
        self.assertEqual(domain.state["phase"], "SINGLE_ONLY")
        self.assertEqual(random.state["phase"], "QUALIFYING")

    def test_failed_after_single_qdel_rolls_back_latest_single_not_candidate(self):
        c = self.controller(); self.qualify(c); c.tick()
        self.assertTrue(c.state["single_fenced"])
        self.pbs.rows[c.state["candidate"]["job_id"]] = dict(job_state="F", Exit_status=271)
        c.tick(); c.tick()
        self.assertEqual(c.state["phase"], "SINGLE_ONLY")
        self.assertEqual(self.pbs.submissions[-1]["source"]["record"]["step"], 30)
        self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], 1)
        self.assertFalse(c.state.get("promoted"))

    def test_live_candidate_and_five_phases_required_before_qdel(self):
        c = self.controller(); self.qualify(c)
        p = Path(c.state["candidate"]["folder"]) / "C/qualification.json"
        row = read(p); row["target_world"] = 16; write(p, row)
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            c.tick()
        self.assertEqual(self.pbs.cancellations, [])

    def test_pass_ready_is_not_handoff_commit(self):
        c = self.controller(); self.qualify(c); c.tick(); c.tick()
        self.assertEqual(c.state["phase"], "CATCH_UP")
        self.assertFalse(c.state["promoted"])
        self.assertTrue(Path(c.state["fallback"]["path"]).exists())

    def test_success_continues_same_job_then_adapts_on_each_retry(self):
        c = self.controller(); final = self.handoff(c)
        self.assertEqual(len(self.pbs.submissions), 1)  # no follow-on train qsub
        self.assertEqual(c.state["active"], c.state["candidate"])
        for free, nodes, reason in ((1, 2, "at_least_one_free_node"), (0, 1, "no_free_node"),
                                    (None, 1, "capacity_unknown"), (3, 2, "at_least_one_free_node")):
            self.pbs.free = free
            self.pbs.rows[c.state["active"]["job_id"]] = dict(job_state="F", Exit_status=271)
            c.tick()
            self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], nodes)
            self.assertEqual(c.state["retry_selection"]["reason"], reason)
            self.assertEqual(self.pbs.submissions[-1]["source"]["record"]["lineage"], c.plan["candidate_lineage"])

    def test_commit_crash_recovery_does_not_cancel_or_revert(self):
        c = self.controller(); self.handoff(c)
        committed = read(c.control / "handoff_commit.json")
        c.save(promoted=False, active=c.plan["old_state"]["single"], phase="CATCH_UP")
        before = len(self.pbs.cancellations)
        recovered = self.reload(c); recovered.tick()
        self.assertTrue(recovered.state["promoted"])
        self.assertEqual(recovered.state["active"], committed["job"])
        self.assertEqual(len(self.pbs.cancellations), before)

    def test_unknown_job_does_not_submit_or_call_capacity(self):
        c = self.controller(); self.pbs.unknown.add(c.state["active"]["job_id"])
        with self.assertRaises(Unknown): c.tick()
        self.assertEqual(self.pbs.submissions, [])
        self.assertEqual(self.pbs.capacity_calls, 0)

    def test_application_error_does_not_blindly_restart(self):
        c = self.controller(); self.handoff(c)
        self.pbs.rows[c.state["active"]["job_id"]] = dict(job_state="F", Exit_status=1)
        c.tick(); c.tick()
        self.assertEqual(len(self.pbs.submissions), 1)
        self.assertIn("Application", c.state["training_blocked"])

    def test_qsub_ambiguity_survives_restart_without_duplicate(self):
        c = self.controller(); self.pbs.ambiguous = True
        with self.assertRaises(Unknown): c.tick()
        c = self.reload(c)
        with self.assertRaises(Unknown): c.tick()
        self.assertEqual(len(self.pbs.submissions), 1)

    def test_ambiguous_candidate_does_not_stop_single_monitor(self):
        c = self.controller(); self.pbs.ambiguous = True
        with self.assertRaises(Unknown): c.tick()
        self.pbs.ambiguous = False
        self.pbs.rows[c.state["active"]["job_id"]] = dict(job_state="F", Exit_status=271)
        with self.assertRaises(Unknown): c.tick()
        self.assertEqual(self.pbs.submissions[-1]["role"], "formal")
        self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], 1)

    def test_pending_adopted_submission_is_cleared_after_crash(self):
        c = self.controller(); c.tick()
        job = c.state["candidate"]
        c.save(pending={"candidate": dict(job, nodes=2, source=c.state["authority"])})
        c = self.reload(c); c.tick()
        self.assertEqual(c.state["pending"], {})
        self.assertEqual(len(self.pbs.submissions), 1)

    def test_deadline_keeps_formal_job_and_retires_candidate(self):
        c = self.controller(); c.tick(); single = c.state["active"]["job_id"]
        c.save(deadline_utc=PAST); c.tick(); c.tick()
        self.assertTrue(c.state["monitoring_finished"])
        self.assertNotIn(single, self.pbs.cancellations)

    def test_deadline_after_qdel_still_restores_single_once(self):
        c = self.controller(); self.qualify(c); c.tick(); c.save(deadline_utc=PAST)
        c.tick(); c.tick(); c.tick()
        self.assertEqual(self.pbs.submissions[-1]["role"], "formal")
        self.assertEqual(self.pbs.submissions[-1]["expected_nodes"], 1)
        self.assertTrue(c.state["monitoring_finished"])

    def test_new_formal_progress_is_authority_after_16_to_8_fallback(self):
        c = self.controller(); self.handoff(c)
        run, lineage = c.state["authority"]["run"], c.state["authority"]["lineage"]
        checkpoint(Path(run) / "checkpoints/global_step_35", 35, 8, lineage)
        self.pbs.rows[c.state["active"]["job_id"]] = dict(job_state="F", Exit_status=271)
        self.pbs.free = 1; c.tick()
        record = self.pbs.submissions[-1]["source"]["record"]
        self.assertEqual((record["step"], record["world_size"], record["lineage"]), (35, 8, lineage))

    def test_cleanup_query_failure_does_not_block_post_handoff_retry(self):
        c = self.controller(); self.handoff(c)
        c.save(fallback_retired=False)
        self.pbs.unknown.add(c.plan["old_state"]["single"]["job_id"])
        self.pbs.rows[c.state["active"]["job_id"]] = dict(job_state="F", Exit_status=271)
        c.tick()
        self.assertIn("temporary PBS", c.state["cleanup_error"])
        self.assertEqual(self.pbs.submissions[-1]["role"], "formal")


class CheckpointTests(Harness):
    def test_hardlink_pin_survives_original_retention(self):
        record = checkpoint(self.root / "legacy/global_step_10", 10)
        pinned = ck.pin(record, self.root / "pin")
        self.assertEqual(os.stat(Path(record["path"]) / "data.pt").st_ino,
                         os.stat(Path(pinned["path"]) / "data.pt").st_ino)
        shutil.rmtree(record["path"])
        ck.validate(pinned)

    def test_missing_optimizer_dataloader_world_and_mutated_metadata_rejected(self):
        for case in ("data", "optimizer", "world", "changed"):
            with self.subTest(case=case):
                r = checkpoint(self.root / case / "global_step_1", 1)
                p = Path(r["path"])
                if case == "data": (p / "data.pt").unlink()
                elif case == "optimizer": (p / "actor/dist/payload.distcp").unlink()
                elif case == "world":
                    spec = read(p / "actor/ckpt_contents.json"); spec["world_size"] = 16; write(p / "actor/ckpt_contents.json", spec)
                else: (p / "data.pt").write_bytes(b"changed")
                with self.assertRaises((RuntimeError, FileNotFoundError)): ck.validate(r)

    def test_latest_recovers_complete_rename_before_tracker_and_ignores_staging(self):
        run = self.root / "run"
        checkpoint(run / "checkpoints/global_step_3", 3, 8, "lineage")
        checkpoint(run / "checkpoints/.staging/global_step_4", 4, 8, "lineage")
        self.assertEqual(ck.latest(run, "lineage")["step"], 3)
        with self.assertRaisesRegex(RuntimeError, "Mixed checkpoint"):
            ck.latest(run, "other")

    def test_five_phase_evidence_survives_pruning_large_intermediates(self):
        c = self.controller(); ready = self.qualify(c)
        req = read(Path(c.state["candidate"]["folder"]) / "request.json")
        ck.retain(req["run_dir"], req["lineage"])
        self.assertEqual(len(list((Path(req["run_dir"]) / "checkpoints").glob("global_step_*"))), 2)
        evidence.verify_qualification(c.state["candidate"]["folder"], ready)


class PolicyTests(unittest.TestCase):
    def test_deadline_from_previous_or_now(self):
        at = "2026-10-21T10:00:00+00:00"
        self.assertEqual(extended_deadline("2026-10-22T00:27:00+00:00", 24, at), "2026-10-23T00:27:00+00:00")
        self.assertEqual(extended_deadline("2026-10-20T00:27:00+00:00", 24, at), "2026-10-22T10:00:00+00:00")

    def test_retry_exit_requires_confirmed_terminal_and_explicit_walltime(self):
        self.assertTrue(retryable(dict(job_state="F", Exit_status=271)))
        self.assertTrue(retryable(dict(job_state="F", Exit_status=-29, comment="job exceeded resource walltime")))
        for row in (dict(job_state="F", Exit_status=-29), dict(job_state="R", Exit_status=271),
                    dict(job_state="F", Exit_status=1), dict(job_state="F")):
            self.assertFalse(retryable(row))

    def test_eight_ranks_must_share_one_host(self):
        good = topology(8); evidence.topology(good, 8)
        good[7]["host"] = "hnode002"
        with self.assertRaisesRegex(RuntimeError, "physical host"): evidence.topology(good, 8)

    def test_pbs_scripts_reserved_direct_q_24h_no_hold(self):
        for nodes in (1, 2):
            text = render_pbs("/experiment", "/experiment/job", nodes, "0390_synthetic")
            self.assertIn("#PBS -q R9920261000\n", text)
            self.assertIn("#PBS -l walltime=24:00:00", text)
            self.assertIn("#PBS -l select=" + str(nodes) + "\n", text)
            self.assertNotIn("#PBS -h", text)
            self.assertNotIn("host=", text)
            self.assertIn("#PBS -r n", text)

    def test_job_mutation_rejects_wrong_owner_queue_or_script(self):
        job = dict(job_id="123.pbs1", folder="/expected", job_name="0390_test")
        row = dict(Job_Owner=getpass.getuser() + "@login", queue="R9920261000", Job_Name="0390_test", Submit_arguments="/expected/stage.pbs")
        validate_job(job, row)
        for key in ("Job_Owner", "queue", "Job_Name", "Submit_arguments"):
            with self.assertRaises(RuntimeError): validate_job(job, dict(row, **{key: "wrong"}))

    def test_unknown_capacity_is_bounded_and_not_a_fake_zero(self):
        pbs = PBS()
        with patch.object(pbs, "command", side_effect=Unknown("timeout")) as command:
            result = pbs.capacity()
        self.assertEqual(command.call_count, 2)
        self.assertIsNone(result["free_slots"])
        self.assertEqual(choose_nodes(True, result), (1, "capacity_unknown"))

    def test_whole_host_count_no_blacklist_and_no_four_node_gate(self):
        q = dict(queue_type="Execution", acl_group_enable="True", acl_groups="gcg51557", enabled="True", started="True")
        q.update({"default_chunk.node_group": "nii", "resources_available.ncpus": "192", "resources_assigned.ncpus": "0"})
        nodes = {"hnode552": dict(state="free", resources_available=dict(host="hnode552", ncpus=0, ngpus=0, node_group="nii"))}
        for i in (0, 1):
            nodes[f"hnode552[{i}]"] = dict(state="free", resources_available=dict(host="@hnode552", ncpus=96, ngpus=4, node_group="@hnode552"))
        cap = free_hosts({"nodes": nodes}, None, q)
        self.assertEqual(cap["free_slots"], 1)
        self.assertEqual(choose_nodes(True, cap)[0], 2)

    def test_builder_keeps_live_parameter_identity_and_save_metadata(self):
        live = object(); metadata = object(); observed = []
        manager = types.SimpleNamespace(use_megatron_fsdp=False,
                                        _build_model_sharded_state_dict=lambda m: {"param": live})
        def original(self, state, meta, is_loading=False):
            observed.append((state, meta)); return state
        builder = optimizer_repair._wrap_builder(original)
        restored = builder(manager, {"param": "detached"}, metadata, is_loading=True)
        self.assertIs(restored["param"], live)
        saved = {"param": "save-layout"}
        self.assertIs(builder(manager, saved, metadata), saved)
        self.assertIs(observed[-1][1], metadata)


class BindingTests(Harness):
    def test_private_audit_eight_and_sixteen_worlds_and_no_legacy_relabel(self):
        job = self.root / "job"
        audit = job / "audit_source"; audit.mkdir(parents=True)
        (audit / "__init__.py").write_text("")
        (audit / "core.py").write_text("")
        (audit / "state_audit.py").write_text("OPTIMIZER_AUDIT_POLICY=" + repr(POLICY) + "\ndef state_digest(x,sampled=False): return x\ndef engine_state(x): return {'optimizer_audit_policy': OPTIMIZER_AUDIT_POLICY, 'optimizer':x}\n")
        (audit / "audit_offload.py").write_text("from .state_audit import engine_state\ndef checkpoint_engine_state(x): return engine_state(x)\n")
        binder = job / "runtime/d4_training_continuation/pinned_audit.py"
        binder.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "d4_training_continuation/pinned_audit.py", binder)
        write(job / "request.json", dict(job_dir=str(job), expected_nodes=2, contract={"snapshot_files": files(job)}))
        spec = importlib.util.spec_from_file_location("continuation_test_binder", binder)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for world, name in ((8, "C"), (16, "B")):
            phase = job / name; phase.mkdir(); write(phase / "phase.json", {"target_world": world})
            fake = types.SimpleNamespace(distributed=types.SimpleNamespace(is_initialized=lambda: True,
                                        get_world_size=lambda: world, get_rank=lambda: 6))
            poisoned = types.SimpleNamespace(__file__="/wrong/audit.py", OPTIMIZER_AUDIT_POLICY="wrong")
            with patch.dict(os.environ, {"CONT_JOB": str(job), "CONT_PHASE_DIR": str(phase)}, clear=True), \
                 patch.dict(sys.modules, {"torch": fake, "d4_two_node_resumefix.state_audit": poisoned}):
                actual = module.checkpoint_engine_state("moments")
                self.assertEqual(actual["optimizer_audit_policy"], POLICY)
                write(job / "saved.json", {"optimizer": "moments"})
                with self.assertRaisesRegex(RuntimeError, "optimizer_audit_policy"):
                    module.compare_restored_state(job / "saved.json", actual)
                self.assertEqual(read(phase / "restore_comparison_rank_6.json")["status"], "FAIL")
                self.assertNotIn("optimizer_audit_policy", read(job / "saved.json"))
                fake.distributed.get_world_size = lambda: 32
                with self.assertRaisesRegex(RuntimeError, "world differs"): module.require_audit_binding()


class IntegrationTests(Harness):
    def template_plan(self, kind="domain"):
        p = paths(self.root, kind)
        template = self.root / "template"
        template.mkdir()
        shutil.copytree(ROOT / "d4_training_continuation", template / "runtime/d4_training_continuation",
                        ignore=shutil.ignore_patterns("__pycache__"))
        reward = template / "synthetic_reward/__init__.py"
        reward.parent.mkdir()
        old_folder = self.root / "old_job"
        reward.write_text("diagnostics_dir=" + repr(str(old_folder / "math_diagnostics")) + "\n")
        write(template / "math_repair.json", {"diagnostics": str(old_folder / "math_diagnostics"), "policy": "unchanged"})
        (template / "node.template.sh").write_text('#!/bin/bash\n"$MD_PYTHON" -m multidomain.stage --request "$MD_REQUEST"\n')
        cfg = dict(training=dict(train_nodes=2, train_batch_size=256, rollout_n=8, test_freq=25,
                                 save_freq=5, total_epochs=1, total_training_steps=468), seed=42)
        base = copy.deepcopy(cfg); base["training"]["train_nodes"] = 1
        old = dict(job_dir=str(old_folder), config=cfg, data_dir=str(self.root / "data"),
                   reward={"runtime_path": str(old_folder / "synthetic_reward/__init__.py")},
                   sampling={"plan_sha256": "same"}, info={"base_config": base})
        return dict(root=str(self.root), kind=kind, template=str(template), template_files=files(template),
                    operation="synthetic", acceptance={"identity": "common-pass"}, old_request=old,
                    root_files={}, candidate_lineage="dual", single_lineage="single")

    def test_job_snapshot_binds_config_reward_sampler_and_original_manifest_config(self):
        plan = self.template_plan()
        before = copy.deepcopy(plan["old_request"])
        made = source.make_job(plan, "formal", 1, {"lineage": "single", "record": {}}, FUTURE)
        r = source.validate_request(Path(made["folder"]) / "request.json")
        self.assertEqual(r["contract"]["base_config"], before["info"]["base_config"])
        self.assertEqual(r["sampling"], before["sampling"])
        self.assertEqual(r["config"]["training"]["train_batch_size"], 256)
        self.assertEqual(r["config"]["training"]["train_nodes"], 1)
        self.assertEqual(r["config"]["resolved"]["train_gpus"], 8)
        self.assertEqual(r["config"]["resolved"]["training_rollout_replicas"], 2)
        self.assertEqual(r["config"]["scheduler"]["walltime"], "24:00:00")
        self.assertEqual(plan["old_request"], before)
        text = (Path(made["folder"]) / "synthetic_reward/__init__.py").read_text()
        self.assertIn(str(Path(made["folder"]) / "math_diagnostics"), text)
        changed = copy.deepcopy(r); changed["sampling"] = {"plan_sha256": "wrong"}
        write(Path(made["folder"]) / "request.json", changed)
        with self.assertRaisesRegex(RuntimeError, "sampling contract changed"):
            source.validate_request(Path(made["folder"]) / "request.json")

    def test_first_source_captured_at_startup_uses_current_formal_lineage(self):
        c = self.controller(); c.tick()
        job = c.state["candidate"]
        r = read(Path(job["folder"]) / "request.json")
        r["control_dir"] = str(c.control)
        source35 = checkpoint(c.p["single"] / "checkpoints/global_step_35", 35)
        (c.p["single"] / "checkpoints/latest_checkpointed_iteration.txt").write_text("35")
        selected = runtime.select_source(r, "A")
        self.assertEqual(selected["step"], 35)
        self.assertNotEqual(selected["path"], source35["path"])

    def test_apply_repetition_never_extends_twice_and_admits_after_both_owners(self):
        controllers = {k: self.controller(k) for k in cli.KINDS}
        before = {k: c.state["deadline_utc"] for k, c in controllers.items()}
        def spawn(root, kind):
            # Both admission files must still be absent during owner startup.
            self.assertFalse(any((c.control / "admit.json").exists() for c in controllers.values()))
        fake_status = {k: dict(controller_active=True, migration_complete=True, deadline_utc=before[k],
                              candidate={"job_id": "synthetic"}) for k in cli.KINDS}
        with patch.object(cli, "preflight"), patch.object(cli, "spawn", side_effect=spawn), patch.object(cli, "status", return_value=fake_status):
            cli.apply(self.root, "synthetic", 24)
        with patch.object(cli, "preflight"), patch.object(cli, "spawn"), patch.object(cli, "status", return_value=fake_status):
            cli.apply(self.root, "synthetic", 24)
        self.assertEqual({k: read(c.control / "state.json")["deadline_utc"] for k, c in controllers.items()}, before)
        self.assertTrue(all((c.control / "admit.json").is_file() for c in controllers.values()))

    def test_extend_repetition_keeps_exact_targets(self):
        controllers = {k: self.controller(k) for k in cli.KINDS}
        with patch.object(cli, "spawn"), patch.object(cli, "status", return_value={}):
            cli.extend(self.root, "plus24", 24)
            first = {k: read(c.control / "extensions/plus24.json") for k, c in controllers.items()}
            cli.extend(self.root, "plus24", 24)
            self.assertEqual({k: read(c.control / "extensions/plus24.json") for k, c in controllers.items()}, first)
            with self.assertRaisesRegex(RuntimeError, "another duration"): cli.extend(self.root, "plus24", 12)

    def test_direct_qsub_receipt_is_recovered_after_process_crash(self):
        folder = self.root / "job"; folder.mkdir()
        write(folder / "qsub_attempted.json", {"at_utc": now()})
        (folder / "qsub.stdout.txt").write_text("123.pbs1\n")
        pbs = PBS()
        with patch.object(pbs, "observe", return_value={}) as observe, patch.object(pbs, "command") as command:
            job = pbs.submit(folder, "0390_test")
        self.assertEqual(job["job_id"], "123.pbs1")
        observe.assert_called_once(); command.assert_not_called()

    def test_unknown_qsub_without_record_never_issues_another_qsub(self):
        folder = self.root / "job"; folder.mkdir()
        write(folder / "qsub_attempted.json", {"at_utc": now()})
        pbs = PBS()
        with patch.object(pbs, "jobs", return_value={}), patch.object(pbs, "command") as command:
            with self.assertRaisesRegex(Unknown, "Ambiguous qsub"): pbs.submit(folder, "0390_test")
        command.assert_not_called()

    def test_source_or_audit_change_rejects_job_before_gpu_load(self):
        plan = self.template_plan()
        made = source.make_job(plan, "candidate", 2, {"legacy": True}, FUTURE)
        folder = Path(made["folder"])
        changed = folder / "runtime/d4_training_continuation/pinned_audit.py"
        changed.write_text(changed.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(RuntimeError, "Source changed"): source.validate_request(folder / "request.json")

    def test_rank_missing_load_evidence_cannot_pass_even_with_saved_checkpoint(self):
        c = self.controller(); ready = self.qualify(c)
        folder = Path(c.state["candidate"]["folder"])
        phase = folder / "D"
        (phase / "optimizer_load_returned_rank_6.json").unlink()
        with self.assertRaises(FileNotFoundError): evidence.verify_qualification(folder, ready)


class TrainerOrderTests(Harness):
    def build(self, name):
        folder = self.root / "job"; phase = folder / name; phase.mkdir(parents=True)
        run = self.root / "lineage"
        resume = checkpoint(self.root / "source/global_step_20", 20, 8)
        req = dict(root=str(self.root), job_dir=str(folder), run_dir=str(run), lineage="new",
                   kind="random", role="candidate", identity="synthetic", control_dir=str(self.root / "control"))
        write(phase / "phase.json", {"target_world": 16})
        calls = []
        class Base:
            def __init__(self):
                self.total_training_steps = 468
                self.global_steps = 21
                self.config = types.SimpleNamespace(data=types.SimpleNamespace(dataloader_num_workers=0),
                    trainer=types.SimpleNamespace(default_local_dir=str(run / "checkpoints"), max_actor_ckpt_to_keep=None))
                self.checkpoint_manager = types.SimpleNamespace(update_weights=lambda step: calls.append(("sync", step)))
            def _update_actor(self, batch):
                calls.append(("update", self.global_steps))
                return types.SimpleNamespace(meta_info={"metrics": {"actor/entropy": 0.3, "actor/loss": 0.2}})
            def _save_checkpoint(self):
                calls.append(("save", self.global_steps))
                checkpoint(Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}", self.global_steps, 16)
            def _get_gen_batch(self, batch):
                calls.append(("generate", self.global_steps)); return batch
            def _validate(self, merged=False):
                return self._get_gen_batch("validation")
        obj = continuation_trainer.trainer_class(Base)()
        obj.configure_continuation(req, phase, resume)
        summary = types.SimpleNamespace(summarize_batch=lambda _: dict(prompts=256, responses=2048, prompt_ids=["synthetic"]))
        module_patch = patch.dict(sys.modules, {"d4_two_node_resumefix.trainer": summary})
        module_patch.start(); self.addCleanup(module_patch.stop)
        def passed(request, phase, source, cp, step):
            ck.validate(cp)
            self.assertEqual(cp["step"], step)
            write(Path(phase) / "qualification.json", {"status": "PASS"})
        p = patch.object(continuation_trainer, "phase_pass", side_effect=passed)
        p.start(); self.addCleanup(p.stop)
        return obj, req, calls

    def test_qualification_save_before_sync_and_validation_before_next_process(self):
        obj, req, calls = self.build("A")
        obj._update_actor(None); obj.checkpoint_manager.update_weights(21)
        self.assertEqual(calls[:3], [("update", 21), ("save", 21), ("sync", 21)])
        self.assertEqual(obj._validate(), "validation")
        obj.global_steps = 22
        with self.assertRaises(continuation_trainer.PhaseComplete): obj._get_gen_batch("train")
        self.assertNotIn(("generate", 22), calls)
        record = read(Path(req["job_dir"]) / "A/step_21.json")
        self.assertEqual(record["actor_metrics"]["actor/entropy"], 0.3)

    def test_catchup_forces_exact_checkpoint_before_rollout_wakes(self):
        obj, req, calls = self.build("E")
        folder = Path(req["job_dir"])
        def ready(cp):
            write(folder / "release.json", dict(target=23, execution_id="x", identity="synthetic"))
        obj.publish_ready = ready
        obj.wait_for = lambda _: {"identity": "synthetic"}
        with patch.dict(os.environ, {"PBS_JOBID": "123.pbs1"}):
            for step in (21, 22, 23):
                obj.global_steps = step
                obj._update_actor(None); obj.checkpoint_manager.update_weights(step)
        self.assertNotIn(("save", 22), calls)
        self.assertLess(calls.index(("save", 23)), calls.index(("sync", 23)))
        self.assertEqual(read(folder / "matched.json")["checkpoint"]["step"], 23)


if __name__ == "__main__":
    unittest.main()
