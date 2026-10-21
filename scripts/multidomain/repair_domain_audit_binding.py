#!/usr/bin/env python3
"""Pin the domain candidate audit to its validated snapshot; preserve handoff gates.

Default: read-only preflight. --apply pauses only the domain owner, preserves its
deadline, archives the failed candidate's unpromoted checkpoint directory by
rename, and rearms the original controller. No qsub/qdel/qalter is called here.
"""
import argparse
from contextlib import ExitStack
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
UTILITY_SHA = "8e0b37101ae4544dff3cddfb5f14f9e3db39a186fb0bd8bfac924ab4dce296a1"
utility_path = HERE / "repair_switch_worker.py"
if hashlib.sha256(utility_path.read_bytes()).hexdigest() != UTILITY_SHA:
    raise RuntimeError("Repair utilities differ from the reviewed source")
spec = importlib.util.spec_from_file_location("domain_audit_repair_utilities", utility_path)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
VERSION = "domain-pinned-audit-v1"
POLICY = "fused-adam-empty-local-group-step-v1"
PACKAGE = "d4_db_switch"
RUN = u.BASE + "_db_n2_once_v1"
WORKER_SHA = "b95242744644f8f4786e958f57b954e65d05e9ca53f988097161aa6d1ac26bdd"
AUDIT_SHA = "59fddccb464acba7b2f58ed6113ca44fc74ea6af9e3a5ec65a54706b41d3fe8a"
OFFLOAD_SHA = "46506747b51c26bb07e400d7a6c3498dc7a4ed0d425e328f8d4998bd9a197bbf"
ERROR = "Same-topology sampled model/optimizer or full LR state mismatch"
AREA = "audit_binding_repair"


def patched_worker(text):
    u.need(u.text_sha(text) == WORKER_SHA, "Unknown domain worker implementation")
    replacements = {
        "from d4_two_node_resumefix.state_audit import state_digest":
            "from .pinned_audit import state_digest",
        "from d4_two_node_resumefix.audit_offload import checkpoint_engine_state":
            "from .pinned_audit import checkpoint_engine_state, require_audit_binding, compare_restored_state",
        "        from .optimizer_repair import worker_check\n":
            "        from .optimizer_repair import worker_check\n        require_audit_binding()\n",
        '            need(saved.exists() and read(saved) == actual, "' + ERROR + '")':
            "            compare_restored_state(saved, actual)",
    }
    for before, after in replacements.items():
        u.need(text.count(before) == 1, "Unexpected worker structure")
        text = text.replace(before, after, 1)
    compile(text, "repaired_domain_worker.py", "exec")
    return text


def inspect(root, job_id, observe=u.query):
    root = Path(root).resolve()
    control = u.owned(root / "outputs/multidomain" / RUN / "handoff", root)
    plan, state = u.read(control / "plan.json"), u.read(control / "state.json")
    u.need(u.digest(plan["contract"]) == plan["identity"] == state["identity"], "Controller identity differs")
    hashes = plan["contract"]["packages"]
    for name, expected in hashes.items():
        u.need(u.packages(control / "runtime" / name) == expected, "Controller source changed: " + name)
    u.need(hashes["d4_two_node_resumefix"] == plan["info"]["identity_inputs"]["package"],
           "Common check package differs")
    u.need(hashes["d4_two_node_resumefix"]["state_audit.py"] == AUDIT_SHA
           and hashes["d4_two_node_resumefix"]["audit_offload.py"] == OFFLOAD_SHA,
           "This repair requires the previously checked empty-group audit implementation")
    cfg = u.read(control / "runtime_limits_30h_24h/config.json")
    u.need(cfg["root"] == str(root) and cfg["kind"] == "domain"
           and cfg["contract_identity"] == plan["identity"] and cfg["guarded_packages"] == hashes
           and cfg["policy_sha256"] == u.LIMITS_SHA
           and u.sha(control / "runtime_limits_30h_24h/limits_policy.py") == u.LIMITS_SHA,
           "Existing limits adapter differs")
    # Entry/capacity adapters are retained byte-for-byte, including their policies.
    u.need((control / "entry.py").is_file(), "Missing controller entry")
    u.check_acceptance(root, control, plan)
    u.need(u.sha(control / "runtime/d4_db_mathfix/verifier_backend.zip")
           == plan["contract"]["verifier_backend_sha256"], "Verifier snapshot changed")
    worker = control / "runtime/d4_db_switch/worker.py"
    receipt_path = control / AREA / "receipt.json"
    context = dict(root=root, control=control, plan=plan, state=state, limits=cfg,
                   worker=worker, fixed=receipt_path.exists())
    if context["fixed"]:
        receipt = u.read(receipt_path)
        u.need(receipt["version"] == VERSION and receipt["candidate"]["job_id"] == job_id
               and receipt["new_identity"] == plan["identity"]
               and receipt["worker_sha256"] == u.sha(worker)
               and receipt["binder_sha256"] == u.sha(control / "runtime/d4_db_switch/pinned_audit.py")
               == u.sha(HERE / "domain_pinned_audit.py"), "Installed repair receipt differs")
        return context
    patched_worker(worker.read_text())
    u.need(state["phase"] == "WAIT_READY" and state.get("handoff_blocked"), "Domain owner is not blocked before handoff")
    u.need(not any(state.get(k) for k in ("pending_job", "pending_check", "old_monitor_stop_requested", "dual_ready_step"))
           and not any((control / n).exists() for n in u.FENCES), "Submission or handoff has already progressed")
    candidate = state["dual"]
    u.need(candidate["job_id"] == job_id and candidate["job_name"] == "0390_d4dbn2_train", "Candidate changed")
    folder = u.owned(candidate["folder"], control.parent / "jobs")
    u.need(folder.parent == control.parent / "jobs"
           and (folder / "job_id.txt").read_text().strip() == job_id, "Candidate folder differs")
    row = observe(job_id)
    u.need(row.get("job_state") == "F" and str(row.get("Exit_status")) == "1"
           and row.get("queue") == u.QUEUE and row.get("Job_Name") == candidate["job_name"]
           and str(folder / "stage.pbs") in row.get("Submit_arguments", ""), "PBS has not confirmed this reserved candidate finished with exit 1")
    req, result = u.read(folder / "request.json"), u.read(folder / "result.json")
    u.need(req["identity"] == plan["identity"] and req["contract"] == plan["contract"]
           and result["identity"] == plan["identity"] and result["job_id"] == job_id
           and result["status"] == "FAIL", "Candidate result/contract differs")
    for name, expected in req["snapshot_files"].items():
        u.need(u.sha(u.owned(folder / name, folder)) == expected, "Candidate snapshot changed: " + name)
    for name, expected in hashes.items():
        u.need(u.packages(folder / "runtime" / name) == expected, "Candidate package changed: " + name)
    with (folder / "pbs.log").open("rb") as f:
        f.seek(max(0, f.seek(0, 2) - 4 * 1024 * 1024))
        log = f.read().decode(errors="replace")
    u.need(ERROR in log, "Different candidate error; do not rearm")
    first = u.read(folder / "first_update/qualification.json")
    selected = u.read(folder / "restored_update/resume_source.json")
    u.need(first["status"] == "PASS" and first["identity"] == plan["identity"]
           and first["checkpoint"] == selected, "Missing bound first-phase checkpoint")
    cp = u.owned(selected["path"], control.parent / "checkpoints")
    u.need(cp.parent == control.parent / "checkpoints", "Unexpected checkpoint location")
    marker = u.read(cp / "COMPLETE_0390.json")
    u.need(u.sha(cp / "COMPLETE_0390.json") == selected["marker_sha256"]
           and marker["identity"] == plan["identity"] and marker["step"] == selected["step"],
           "Candidate checkpoint marker changed")
    provenance = u.read(cp / "domain_switch.json")
    u.need(provenance["producer_job_id"] == job_id and provenance["identity"] == plan["identity"],
           "Checkpoint belongs to another candidate")
    names = {p.name for p in cp.parent.iterdir()}
    u.need(names <= {cp.name, "latest_complete_0390.json", "latest_checkpointed_iteration.txt"},
           "Other candidate checkpoints/staging entries exist; no archive performed")
    summaries = []
    for rank in range(16):
        path = cp / "actor" / f"db_dual_state_rank_{rank}.json"
        u.need(u.sha(path) == marker["files"][str(path.relative_to(cp))]["sha256"], "Saved audit summary changed")
        summaries.append(u.read(path))
    u.need(all("optimizer_audit_policy" not in x for x in summaries),
           "Candidate did not emit the diagnosed legacy audit schema")
    u.need(not list((folder / "restored_update").glob("optimizer_audit_rank_*.json")),
           "Detailed audits now exist; inspect those before applying this targeted repair")
    u.need(not (folder / "restored_update/qualification.json").exists(), "Second qualification already exists")
    for p in (control.parent / "jobs").glob("*/qsub_attempted.json"):
        u.need((p.parent / "job_id.txt").exists(), "An unresolved candidate submission exists")
    deadline = dt.datetime.fromisoformat(state["deadline_utc"])
    u.need(deadline.tzinfo is not None and deadline > dt.datetime.now(dt.timezone.utc),
           "The monitor deadline has expired; extend the existing owner before applying")
    context.update(candidate_observation=row, checkpoint=cp, candidate_folder=folder)
    return context


def changes(c):
    plan, state, cfg = (copy.deepcopy(c[k]) for k in ("plan", "state", "limits"))
    worker = patched_worker(c["worker"].read_text())
    binder = (HERE / "domain_pinned_audit.py").read_text()
    old_identity = plan["identity"]
    files = plan["contract"]["packages"][PACKAGE]
    files.update({"worker.py": u.text_sha(worker), "pinned_audit.py": u.text_sha(binder)})
    plan["identity"] = u.digest(plan["contract"])
    cfg["guarded_packages"] = copy.deepcopy(plan["contract"]["packages"])
    cfg["contract_identity"] = plan["identity"]
    candidate = copy.deepcopy(state["dual"])
    state.update(identity=plan["identity"], phase="WAIT_CHECKPOINT", dual=None, previous_candidate=candidate,
                 handoff_blocked=None, wait_reason=None, last_error=None, stop_reason=None,
                 controller_status="STOPPED", old_monitor_stop_requested=False, dual_ready_step=None,
                 audit_binding_repair=VERSION)
    archive = c["control"] / AREA / "old_checkpoints"
    receipt = dict(version=VERSION, old_identity=old_identity, new_identity=plan["identity"],
                   candidate=candidate, worker_sha256=u.text_sha(worker), binder_sha256=u.text_sha(binder),
                   deadline_utc=state["deadline_utc"], checkpoint_archive=str(archive),
                   installed_utc=u.now(), candidate_observation=c["candidate_observation"],
                   note="Legacy summaries are preserved; the replacement must rerun both qualification phases.")
    return {"runtime/d4_db_switch/pinned_audit.py": binder,
            "runtime/d4_db_switch/worker.py": worker,
            "plan.json": u.json_text(plan), "runtime_limits_30h_24h/config.json": u.json_text(cfg),
            "state.json": u.json_text(state), AREA + "/receipt.json": u.json_text(receipt)}


def recover(control):
    journal = control / AREA / "transaction.json"
    if not journal.exists():
        return
    tx = u.read(journal)
    u.need(tx["version"] == VERSION, "Unknown audit repair transaction")
    if tx["status"] != "PREPARED":
        return
    for name, item in tx["files"].items():
        path = u.owned(control / name, control)
        value = path.read_text() if path.exists() else None
        u.need(value in (item["before"], item["after"]), "Independent edit during interrupted repair: " + name)
    original, archive = control.parent / "checkpoints", control / AREA / "old_checkpoints"
    if archive.exists():
        u.need(not original.exists() or not any(original.iterdir()), "New checkpoint appeared; preserve both directories")
        if original.exists():
            original.rmdir()
        archive.rename(original)
    for name, item in tx["files"].items():
        path = control / name
        if item["before"] is None:
            path.unlink(missing_ok=True)
        else:
            u.atomic_text(path, item["before"])
    tx["status"] = "ROLLED_BACK"
    u.write(journal, tx)


def publish(c):
    control = c["control"]
    u.need(not c["fixed"], "Already fixed")
    archive = control / AREA / "old_checkpoints"
    u.need(not archive.exists(), "Checkpoint archive already exists without a completed receipt")
    files = {n: dict(before=(control / n).read_text() if (control / n).exists() else None, after=s)
             for n, s in changes(c).items()}
    tx = dict(version=VERSION, status="PREPARED", files=files)
    journal = control / AREA / "transaction.json"
    u.write(journal, tx)
    try:
        (control.parent / "checkpoints").rename(archive)
        (control.parent / "checkpoints").mkdir()
        for name, item in files.items():
            u.atomic_text(control / name, item["after"])
        tx["status"] = "INSTALLED"
        u.write(journal, tx)
    except BaseException:
        recover(control)
        raise


def launch_remaining(root, control):
    if u.active(control) or u.read(control / "state.json")["phase"] == "DONE":
        return
    deadline = dt.datetime.fromisoformat(u.read(control / "state.json")["deadline_utc"])
    hours = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600
    u.need(hours > 0, "Repair is installed but the existing monitor window has expired")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    result = subprocess.run([sys.executable, "-I", str(control / "entry.py"), "resume",
                             "--root", str(root), "--hours", str(hours)], cwd=root, env=env, timeout=45)
    u.need(result.returncode == 0 and u.active(control), "Controller startup unconfirmed; inspect its log")


def run(root, job_id, apply=False):
    root = Path(root).resolve()
    control = root / "outputs/multidomain" / RUN / "handoff"
    journal = control / AREA / "transaction.json"
    pending = journal.exists() and u.read(journal).get("status") == "PREPARED"
    if not apply:
        u.need(not pending, "Interrupted repair; use --apply to recover the journal")
        c = inspect(root, job_id)
        print(u.json_text(dict(status="ALREADY_FIXED" if c["fixed"] else "PREFLIGHT_PASS",
                              candidate=job_id, deadline_utc=c["state"]["deadline_utc"],
                              action="Pin snapshot audit; archive unpromoted dual checkpoint; rerun full qualification")))
        return
    with u.lock(control / AREA / "apply.lock", create=True):
        if pending:
            u.stop_controls([control])
            with ExitStack() as stack:
                for path in (control / "controller.lock", control.parent / "submission.lock", control.parent / "stage.lock"):
                    stack.enter_context(u.lock(path, create=True))
                recover(control)
        c = inspect(root, job_id)
        if not c["fixed"]:
            u.stop_controls([control])
            try:
                with ExitStack() as stack:
                    for path in (root / "outputs/multidomain" / (u.BASE + "_capacity") / "policy.lock",
                                 control / "controller.lock", control.parent / "submission.lock", control.parent / "stage.lock"):
                        stack.enter_context(u.lock(path, create=True))
                    c = inspect(root, job_id)
                    if not c["fixed"]:
                        publish(c)
            except BaseException:
                # A fully rolled-back owner remains blocked on the old candidate.
                if not journal.exists() or u.read(journal).get("status") != "PREPARED":
                    launch_remaining(root, control)
                raise
        inspect(root, job_id)
        launch_remaining(root, control)
        print("[0390] Audit binding repair installed. Domain controller resumed within its existing window; candidate submission is asynchronous.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--domain-job", required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    try:
        run(args.root, args.domain_job, args.apply)
    except Exception as exc:
        print("[0390] STOP: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
