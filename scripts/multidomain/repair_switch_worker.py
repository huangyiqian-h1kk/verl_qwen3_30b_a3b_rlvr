#!/usr/bin/env python3
"""Repair the candidate Worker name collision, preserving existing handoff gates.

Standalone stdlib entry: safe to execute with `git show COMMIT:path | python -I -`.
Default is read-only preflight. --apply updates the two owned controller snapshots
and resumes their original submission code. It never calls qsub, qdel or qalter.
"""
import argparse
import ast
from contextlib import ExitStack
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

VERSION = "switch-worker-name-v1"
BASE = "qwen30b_d4_10k_uniform_ctx64k_seed42_v1"
QUEUE = "R9920261000"
ROLES = {
    "random": (BASE + "_n2_once_v1", "d4_managed_switch", "0390_d4n2_once", "RandomSwitchWorker"),
    "domain": (BASE + "_db_n2_once_v1", "d4_db_switch", "0390_d4dbn2_train", "DomainSwitchWorker"),
}
OLD_WORKERS = {
    "random": "18ceb9696458f02b615935cfd1fa93a246631b190e715900633cb480d11c4cb1",
    "domain": "84f650577a29270ea8bef2adf383186b8800a96eeb9b25bc0be68502b6a8f5a8",
}
# Hashes of canonical {filename: sha256} maps from the deployed controller releases.
KNOWN_PACKAGES = {
    "random": {"c1c7523549770d792ad2ed0a072887f8cd25326d96d5ba71c848caaceb491096"},
    "domain": {"add67211491ee59c2a3e620fc89356cf54ea28f4f855e0ed47fa4104cc074085",
               "d165a0379ea9ec71e596b329c58a793896fcd87ac3e8c434c0c5f75a3ab5e21a"},
}
VERL_FUNCTIONS = {
    "_determine_fsdp_megatron_base_class": "72f13884b8d674748ae732442c6f26eb3840b9cc655a4af7851c4310994a9b51",
    "create_colocated_worker_cls": "22123f31b05067c57a504570033bc8fba09c199df6f3f175cee272a9200dc32e",
}
LIMITS_SHA = "6c4dba02aac87ee35488e58261b391a27049b43bb3acf0dcf1bddb2e42cf33ed"
ERROR = "ActorRolloutRefWorker.__init__() missing 2 required positional arguments: 'config' and 'role'"
FENCES = ("qdel.json", "decision.json", "readiness.json", "matched.json", "superseded.json")


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, obj):
    atomic_text(path, json_text(obj))


def lock(path, create=False):
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a" if create else "r+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        f.close()
        raise
    return f


def active(control):
    try:
        with lock(control / "controller.lock"):
            return False
    except BlockingIOError:
        return True


def packages(path):
    return {p.name: sha(p) for p in sorted(Path(path).glob("*.py"))}


def owned(path, root):
    path, root = Path(path), Path(root).resolve()
    need(not path.is_symlink() and path.resolve().is_relative_to(root), "Path leaves owned directory: " + str(path))
    return path


def patched_worker(kind, source):
    """Rename the class definition, preserving methods and their __class__ cells.

    A module alias retains `from .worker import Worker`; an alias does not change
    the class's __name__, which is what pinned verl examines along the MRO.
    """
    need(text_sha(source) == OLD_WORKERS[kind], "Unrecognized candidate worker source: " + kind)
    name = ROLES[kind][3]
    old = "class Worker(TwoNodeWorker):"
    need(source.count(old) == 1, "Expected exactly one candidate class")
    updated = source.replace(old, "class " + name + "(TwoNodeWorker):", 1)
    updated += ("\n\n# Pinned verl selects its container base by __name__; avoid its reserved Worker name.\n"
                "# Keep the existing import API without adding another inheritance layer.\n"
                "Worker = " + name + "\n")
    before, after = ast.parse(source), ast.parse(updated)
    after.body.pop()  # Compatibility alias.
    classes = [x for x in after.body if isinstance(x, ast.ClassDef)]
    need(len(classes) == 1 and classes[0].name == name, "Unexpected worker class structure")
    classes[0].name = "Worker"
    need(ast.dump(before) == ast.dump(after), "Worker methods changed during rename")
    return updated


def check_verl(path):
    functions = {x.name: x for x in ast.parse(Path(path).read_text()).body if isinstance(x, ast.FunctionDef)}
    for name, expected in VERL_FUNCTIONS.items():
        need(name in functions and text_sha(ast.dump(functions[name], include_attributes=False)) == expected,
             "Installed verl worker factory differs from the diagnosed version: " + name)


def query(job_id):
    need(re.fullmatch(r"\d+\.[A-Za-z0-9_.-]+", job_id), "Invalid PBS job id")
    result = subprocess.run(["qstat", "-xf", "-F", "json", job_id], capture_output=True,
                            text=True, timeout=45, env={**os.environ, "LC_ALL": "C"})
    need(result.returncode == 0, "PBS observation unavailable: " + result.stderr.strip())
    rows = json.loads(result.stdout, strict=False).get("Jobs", {})
    need(job_id in rows, "PBS did not return the exact candidate id")
    return rows[job_id]


def package_contract(kind, plan):
    name = ROLES[kind][1]
    hashes = ({name: copy.deepcopy(plan["contract"]["package"])} if kind == "random"
              else copy.deepcopy(plan["contract"]["packages"]))
    n2 = plan["info"]["identity_inputs"]["package"]
    if "d4_two_node_resumefix" in hashes:
        need(hashes["d4_two_node_resumefix"] == n2, "Common check package differs within contract")
    hashes["d4_two_node_resumefix"] = n2
    return hashes


def check_limits(root, kind, control, plan, hashes):
    policy = control / "runtime_limits_30h_24h"
    cfg = read(policy / "config.json")
    need(sha(policy / "limits_policy.py") == LIMITS_SHA == cfg["policy_sha256"], "Unknown limits adapter")
    need(cfg["root"] == str(root) and cfg["kind"] == kind and cfg["contract_identity"] == plan["identity"]
         and cfg["guarded_packages"] == hashes and cfg["monitor_default_hours"] == 30
         and cfg["future_random_training_walltime"] == "24:00:00", "Limits adapter contract differs")
    expected = ('from pathlib import Path\nimport hashlib, json, sys\n'
        'here=Path(__file__).resolve().parent\npolicy=here/"runtime_limits_30h_24h"\n'
        'if hashlib.sha256((policy/"limits_policy.py").read_bytes()).hexdigest()!=' + repr(LIMITS_SHA)
        + ': raise RuntimeError("Limits policy changed")\n'
        'sys.path.insert(0,str(policy))\nfrom limits_policy import boot\nboot(here,sys.argv[1:])\n')
    need((control / "entry.py").read_text() == expected, "Unknown controller entry adapter")
    return cfg


def check_acceptance(root, control, plan):
    accepted = read(control / "acceptance.json")
    need(accepted["status"] == "PASS" and accepted["grpo_steps_completed"] == 2
         and accepted["identity"] == plan["info"]["identity"] == plan["contract"]["check_identity"],
         "Missing bound common check PASS")
    check_jobs = root / "outputs/multidomain" / (BASE + "_n2_v1") / "jobs"
    evidence = accepted.get("evidence", {})
    need(bool(evidence), "Common check lacks evidence manifest")
    folders = set()
    for name, expected in evidence.items():
        path = owned(name, check_jobs)
        need(sha(path) == expected, "Common check evidence changed: " + str(path))
        folders.add(check_jobs / path.relative_to(check_jobs).parts[0])
    need(len(folders) == 1, "Mixed common check evidence")
    folder = folders.pop()
    need((folder / "job_id.txt").read_text().strip() == accepted["job_id"], "Common check job differs")
    need(read(folder / "result.json") == accepted, "Accepted check differs from original result")
    return accepted


def inspect(root, kind, expected_job, observe=query):
    run, package, job_name, _ = ROLES[kind]
    control = owned(root / "outputs/multidomain" / run / "handoff", root)
    plan, state = read(control / "plan.json"), read(control / "state.json")
    need(digest(plan["contract"]) == plan["identity"] == state["identity"], "Controller identity mismatch")
    hashes = package_contract(kind, plan)
    for name, values in hashes.items():
        need(packages(control / "runtime" / name) == values, "Frozen package changed: " + name)
    original_hashes = dict(hashes[package], **{"worker.py": OLD_WORKERS[kind]})
    need(digest(original_hashes) in KNOWN_PACKAGES[kind], "Unknown controller implementation: " + kind)
    worker = control / "runtime" / package / "worker.py"
    receipt_path = control / "worker_name_repair/receipt.json"
    fixed = receipt_path.exists()
    if fixed:
        receipt = read(receipt_path)
        need(receipt["version"] == VERSION and receipt["new_identity"] == plan["identity"]
             and receipt["candidate"]["job_id"] == expected_job and receipt["worker_sha256"] == sha(worker),
             "Repair receipt differs; do not reset this controller")
        # Only the already recorded rename can be recognized as installed.
        restored = worker.read_text().split("\n\n# Pinned verl selects", 1)[0]
        restored = restored.replace("class " + ROLES[kind][3] + "(TwoNodeWorker):", "class Worker(TwoNodeWorker):", 1)
        need(patched_worker(kind, restored) == worker.read_text(), "Installed repair source differs")
    else:
        patched_worker(kind, worker.read_text())
    cfg = check_limits(root, kind, control, plan, hashes)
    check_acceptance(root, control, plan)
    if kind == "random":
        for name, expected in plan["single_monitor_files"].items():
            need(sha(owned(control / "support/random_queue" / name, control)) == expected, "Single monitor snapshot changed")
    elif plan["contract"].get("verifier_backend_sha256"):
        need(sha(control / "runtime/d4_db_mathfix/verifier_backend.zip") == plan["contract"]["verifier_backend_sha256"],
             "Domain verifier archive changed")
    check_verl(Path(plan["info"]["base_config"]["verl_src"]) / "verl/single_controller/ray/base.py")
    context = dict(kind=kind, control=control, worker=worker, plan=plan, state=state,
                   limits=cfg, hashes=hashes, fixed=fixed)
    if fixed:
        return context  # Never rearm a new/active candidate or a completed handoff.
    need(state["phase"] == "WAIT_READY" and bool(state.get("handoff_blocked")), "Controller is not blocked before handoff")
    need(not any(state.get(k) for k in ("pending_job", "pending_check", "old_monitor_stop_requested", "dual_ready_step")),
         "Submission or handoff is already in progress")
    need(not any((control / n).exists() for n in FENCES), "Handoff evidence already exists")
    dual = state["dual"]
    need(dual["job_id"] == expected_job and dual["job_name"] == job_name, "Candidate changed; inspect current status")
    folder = owned(dual["folder"], control.parent / "jobs")
    need(folder.parent == control.parent / "jobs", "Wrong candidate directory")
    row = observe(expected_job)
    need(row.get("job_state") == "F" and str(row.get("Exit_status")) == "1", "Candidate is not confirmed finished with exit 1")
    need(row.get("queue") == QUEUE and row.get("Job_Name") == job_name
         and str(folder / "stage.pbs") in row.get("Submit_arguments", ""), "PBS job does not match the recorded reserved candidate")
    need((folder / "job_id.txt").read_text().strip() == expected_job, "Candidate id record differs")
    request = read(folder / "request.json")
    need(request["identity"] == plan["identity"] and request["contract"] == plan["contract"]
         and request["info"] == plan["info"] and request["expected_nodes"] == 2, "Candidate contract differs")
    for name, expected in request["snapshot_files"].items():
        need(sha(owned(folder / name, folder)) == expected, "Candidate snapshot changed: " + name)
    for name, values in hashes.items():
        need(packages(folder / "runtime" / name) == values, "Candidate package differs: " + name)
    result = read(folder / "result.json")
    need(result["status"] == "FAIL" and result["job_id"] == expected_job and result["identity"] == plan["identity"],
         "Missing candidate-bound FAIL")
    with (folder / "pbs.log").open("rb") as f:
        f.seek(max(0, f.seek(0, 2) - 4 * 1024 * 1024))
        log = f.read().decode(errors="replace")
    need(ERROR in log and "verl/single_controller/ray/base.py" in log and "super().__init__()" in log,
         "Candidate did not fail with the diagnosed worker constructor error")
    need(not any((folder / name).exists() for name in ("ready.json", "readiness.json", "ready_to_handoff.json",
                                                      "first_update/qualification.json", "restored_update/result.json")),
         "Candidate has progressed beyond worker construction")
    need(not list(folder.glob("**/restore_rank_*.json")) and not list(folder.glob("**/step_*.json")),
         "Candidate has restoration/update evidence; not a constructor-only failure")
    need(not list((control.parent / "checkpoints").glob("global_step_*"))
         and not list((control.parent / "checkpoints").glob(".staging_*")), "Dual checkpoint state exists")
    for marker in (control.parent / "jobs").glob("*/qsub_attempted.json"):
        need((marker.parent / "job_id.txt").exists(), "Unresolved candidate submission: " + str(marker.parent))
    context["candidate_observation"] = row
    return context


def changes(context):
    kind, control = context["kind"], context["control"]
    package = ROLES[kind][1]
    updated = patched_worker(kind, context["worker"].read_text())
    plan, state, cfg = copy.deepcopy(context["plan"]), copy.deepcopy(context["state"]), copy.deepcopy(context["limits"])
    old_identity = plan["identity"]
    target = plan["contract"]["package"] if kind == "random" else plan["contract"]["packages"][package]
    target["worker.py"] = text_sha(updated)
    plan["identity"] = digest(plan["contract"])
    plan["hours"] = 30
    cfg["contract_identity"] = plan["identity"]
    cfg["guarded_packages"][package]["worker.py"] = text_sha(updated)
    candidate = copy.deepcopy(state["dual"])
    state.update(identity=plan["identity"], phase="WAIT_CHECKPOINT", dual=None, previous_candidate=candidate,
                 handoff_blocked=None, wait_reason=None, last_error=None, stop_reason=None,
                 controller_status="STOPPED", old_monitor_stop_requested=False, dual_ready_step=None,
                 worker_name_repair=VERSION)
    receipt = dict(version=VERSION, installed_utc=now(), old_identity=old_identity, new_identity=plan["identity"],
                   candidate=candidate, worker_sha256=text_sha(updated), common_check_identity=plan["info"]["identity"],
                   candidate_observation=context["candidate_observation"], old_candidate_preserved=True)
    return {
        str(context["worker"].relative_to(control)): updated,
        "plan.json": json_text(plan),
        "runtime_limits_30h_24h/config.json": json_text(cfg),
        "state.json": json_text(state),
        "worker_name_repair/receipt.json": json_text(receipt),
    }


def recover_interrupted(control):
    """Under controller lock, undo only our own incomplete publication before retrying."""
    journal = control / "worker_name_repair/transaction.json"
    if not journal.exists():
        return
    record = read(journal)
    need(record["version"] == VERSION, "Unknown repair transaction")
    if record["status"] != "PREPARED":
        return
    for name, item in record["files"].items():
        path = owned(control / name, control)
        current = path.read_text() if path.exists() else None
        need(current in (item["before"], item["after"]), "Interrupted repair file changed independently: " + name)
    for name, item in record["files"].items():
        path = owned(control / name, control)
        if item["before"] is None:
            path.unlink(missing_ok=True)
        else:
            atomic_text(path, item["before"])
    record["status"] = "ROLLED_BACK"
    write(journal, record)


def publish(context):
    control = context["control"]
    need(not context["fixed"], "Already repaired")
    files = {}
    for name, after in changes(context).items():
        path = owned(control / name, control)
        files[name] = dict(before=path.read_text() if path.exists() else None, after=after)
    record = dict(version=VERSION, status="PREPARED", at_utc=now(), files=files)
    area = control / "worker_name_repair"
    if (area / "transaction.json").exists():
        atomic_text(area / ("transaction_" + uuid.uuid4().hex + ".json"), (area / "transaction.json").read_text())
    write(area / "transaction.json", record)
    try:
        for name, item in files.items():
            atomic_text(control / name, item["after"])
        record["status"] = "INSTALLED"
        write(area / "transaction.json", record)
    except BaseException:
        recover_interrupted(control)
        raise


def stop_controls(controls):
    for c in controls:
        if active(c):
            (c / "stop.request").touch()
    until = time.monotonic() + 45
    while any(active(c) for c in controls) and time.monotonic() < until:
        time.sleep(.25)
    need(not any(active(c) for c in controls), "Controller is still finishing a tick; rerun --apply after it stops")


def launch(root, control):
    if active(control) or read(control / "state.json").get("phase") == "DONE":
        return
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["LC_ALL"] = "C"
    result = subprocess.run([sys.executable, "-I", str(control / "entry.py"), "resume",
                             "--root", str(root), "--hours", "30"], cwd=root, env=env, timeout=45)
    need(result.returncode == 0, "Controller did not resume; see " + str(control / "controller.log"))
    state = read(control / "state.json")
    start, end = (dt.datetime.fromisoformat(state[k]) for k in ("window_started_utc", "deadline_utc"))
    need(active(control) and state["controller_status"] == "RUNNING" and abs((end-start).total_seconds()-108000) < 2,
         "30-hour monitor startup was not confirmed: " + str(control))


def run(root, candidates, apply=False):
    root = Path(root).resolve()
    controls = {kind: root / "outputs/multidomain" / ROLES[kind][0] / "handoff" for kind in candidates}
    if not apply:
        for kind, c in controls.items():
            path = c / "worker_name_repair/transaction.json"
            need(not path.exists() or read(path)["status"] != "PREPARED", "Interrupted repair; use --apply to recover")
            item = inspect(root, kind, candidates[kind])
            print(json_text(dict(role=kind, status="ALREADY_FIXED" if item["fixed"] else "PREFLIGHT_PASS",
                                 candidate=candidates[kind], changes="class name and bound controller contract only",
                                 monitor_hours=30, future_random_walltime="24:00:00")), flush=True)
        return
    area = root / "outputs/multidomain" / (BASE + "_n2_once_v1") / "worker_name_repair"
    with lock(area / "apply.lock", create=True):
        # Recover interrupted metadata before normal preflight. No jobs are touched.
        for c in controls.values():
            path = c / "worker_name_repair/transaction.json"
            if path.exists() and read(path)["status"] == "PREPARED":
                stop_controls([c])
                with lock(c / "controller.lock"):
                    recover_interrupted(c)
        before = {k: inspect(root, k, candidates[k]) for k in controls}
        affected = [controls[k] for k, v in before.items() if not v["fixed"]]
        stop_controls(affected)
        try:
            with ExitStack() as stack:
                for c in affected:
                    stack.enter_context(lock(c / "controller.lock"))
                    stack.enter_context(lock(c.parent / "submission.lock", create=True))
                    stack.enter_context(lock(c.parent / "stage.lock", create=True))
                contexts = {k: inspect(root, k, candidates[k]) for k in controls}
                for context in contexts.values():
                    if not context["fixed"]:
                        publish(context)
        finally:
            # A failure before publication can resume the unchanged blocked owner;
            # an interrupted publication is rolled back before it can be launched.
            errors = {}
            for kind, c in controls.items():
                try:
                    inspect(root, kind, candidates[kind])
                    launch(root, c)
                except Exception as exc:
                    errors[kind] = str(exc)
            if errors:
                print(json_text(dict(controller_resume_errors=errors)), flush=True)
        need(not errors, "Some controllers were not resumed; inspect the messages above")
        print("[0390] Worker name repair installed. Existing controllers resumed for 30h; new candidate submission is asynchronous.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--random-job", required=True, help="Failed random candidate to repair")
    parser.add_argument("--domain-job", required=True, help="Failed domain candidate to repair")
    parser.add_argument("--apply", action="store_true", help="Apply after full preflight and resume original controllers")
    args = parser.parse_args()
    try:
        run(args.root, {"random": args.random_job, "domain": args.domain_job}, args.apply)
    except Exception as exc:
        print("[0390] STOP: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
