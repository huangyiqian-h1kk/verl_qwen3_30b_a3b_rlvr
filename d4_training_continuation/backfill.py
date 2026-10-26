"""One reserved-queue allocation, two fresh evaluation-only processes, then exit."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid
from . import checkpoints as ck, evaluation
from .common import digest, expired, files, locked, need, now, paths, read, sha, verify_files, write
from .scheduler import PBS, Unknown, finished, retryable

TARGETS = {
    "random": dict(step=125, world=16, job="formal_12f4da3e0925419ab5bece3c4c0315de", job_id="2744818.pbs1"),
    "domain": dict(step=100, world=8, job="formal_3c8e514ec0f543e799546adc6b5da415", job_id="2744561.pbs1"),
}


def protect(root):
    """Run BEFORE any owner can resubmit and retire temporary recovery pins."""
    area = Path(root) / "outputs/multidomain/validation_backfill_sources"
    result = {}
    with locked(area / "pin.lock"):
        for kind, target in TARGETS.items():
            destination = area / f"{kind}_step_{target['step']}"
            folder = paths(root, kind)["run"] / "jobs" / target["job"]
            need((folder / "job_id.txt").read_text().strip() == target["job_id"], "Historical source job changed")
            request = read(folder / "request.json")
            need(request["identity"] == digest(request["contract"])
                 and request["source"] == request["contract"]["source"], "Historical request changed")
            original = request["source"]["record"]
            if (destination / "source.json").exists():
                record = ck.validate(read(destination / "source.json"))
            else:
                candidates = [original]
                phase_pin = folder / "train/source/source.json"
                if phase_pin.exists():
                    candidates.append(read(phase_pin))
                errors = []
                for candidate in candidates:
                    try:
                        need(evaluation.checkpoint_key(candidate) == evaluation.checkpoint_key(original),
                             "Historical recovery copies disagree")
                        ck.validate(candidate)
                        record = ck.pin(candidate, destination)
                        break
                    except FileNotFoundError as error:
                        errors.append(str(error))
                else:
                    raise RuntimeError("Historical checkpoint no longer available: " + repr(errors))
            need(record["step"] == target["step"] and record["world_size"] == target["world"],
                 "Wrong historical checkpoint was protected")
            need(evaluation.checkpoint_key(record) == evaluation.checkpoint_key(original),
                 "Protected history is not the requested checkpoint")
            result[kind] = record
    return result


def prepare(root, operation, plans, sources, deadline):
    area = Path(root).resolve() / "outputs/multidomain/validation_backfills" / operation
    path = area / "state.json"
    if path.exists():
        state = read(path)
        need(state["operation"] == operation and state["sources"] == sources, "Backfill operation changed")
        return area
    # Bind the already-upgraded setup-specific snapshots; never use one setup's
    # verifier/sampler configuration for the other setup.
    write(area / "plans.json", plans)
    write(path, dict(operation=operation, root=str(Path(root).resolve()), sources=sources,
          deadline_utc=deadline, phase="PREPARED", pending=None, job=None, attempts=0, created_utc=now()))
    return area


def make_attempt(area, state):
    from .source import make_job
    from .scheduler import render_pbs
    plans = read(area / "plans.json")
    folder = area / "jobs" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    children = {}
    for kind in ("random", "domain"):
        record = ck.validate(state["sources"][kind])
        children[kind] = make_job(plans[kind], "evaluation", 1,
            dict(record=record, lineage=record["lineage"]), state["deadline_utc"])
    shutil.copytree(Path(__file__).parent, folder / "runtime/d4_training_continuation",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    root = Path(state["root"])
    (folder / "entry.py").write_text("from pathlib import Path\nimport sys\nhere=Path(__file__).resolve().parent\n"
        f"sys.path.insert(0, {str(root)!r})\nsys.path.insert(0, str(here/'runtime'))\n"
        "from d4_training_continuation.backfill import main\nmain()\n")
    node = (Path(plans["random"]["template"]) / "node.template.sh").read_text()
    old = '"$MD_PYTHON" -m multidomain.stage --request "$MD_REQUEST"'
    need(node.count(old) == 1, "Unknown Ray node entry point")
    (folder / "node.sh").write_text(node.replace(old,
        '"$MD_PYTHON" -I ' + shlex.quote(str(folder / "entry.py")) + ' execute --request "$MD_REQUEST"'))
    name = "0390_d4_valfix"
    (folder / "stage.pbs").write_text(render_pbs(root, folder, 1, name))
    # node.sh's infrastructure probes require the ordinary resource config.
    config = read(Path(children["random"]["folder"]) / "request.json")["config"]
    contract = dict(root=str(root), job_dir=str(folder), job_name=name, expected_nodes=1, stage="train",
                    config=config, children={k: dict(v, request_sha256=sha(Path(v["folder"]) / "request.json"))
                                             for k, v in children.items()}, snapshot_files=files(folder))
    write(folder / "request.json", dict(contract, contract=contract, identity=digest(contract)))
    return dict(folder=str(folder), job_name=name)


def validate_request(path):
    r = read(path)
    need(r["identity"] == digest(r["contract"]), "Backfill envelope changed")
    need(Path(path).resolve() == Path(r["job_dir"]) / "request.json", "Wrong backfill request path")
    need(all(r[k] == v for k, v in r["contract"].items()), "Backfill envelope field changed")
    verify_files(r["job_dir"], r["snapshot_files"])
    from .source import validate_request as child_request
    for kind, child in r["children"].items():
        child_path = Path(child["folder"]) / "request.json"
        need(sha(child_path) == child["request_sha256"], "Backfill child request changed")
        c = child_request(child_path)
        need(c["role"] == "evaluation" and c["kind"] == kind and c["expected_nodes"] == 1,
             "Backfill child is not a single-node evaluation-only request")
    return r


def allocation(path):
    r = validate_request(path)
    folder = Path(r["job_dir"])
    need(not (folder / "allocation.json").exists(), "PBS rerun of backfill envelope is forbidden")
    hosts = [x.strip() for x in (folder / "hosts").read_text().splitlines() if x.strip()]
    need(len(hosts) == 1, "Backfill requires one node")
    job = dict(folder=str(folder), job_name=r["job_name"], job_id=os.environ["PBS_JOBID"])
    row = PBS().observe(job)
    need(row["job_state"] == "R" and row.get("run_count") and row.get("stime"), "Unbound backfill allocation")
    record = dict(job_id=job["job_id"], hosts=hosts, execution_id=uuid.uuid4().hex,
                  run_count=row["run_count"], stime=row["stime"], at_utc=now())
    write(folder / "allocation.json", record)
    for child in r["children"].values():
        write(Path(child["folder"]) / "allocation.json", record)


def execute(path):
    from .runtime import wait_gpus
    r = validate_request(path)
    folder = Path(r["job_dir"])
    result = dict(status="RUNNING", identity=r["identity"], job_id=os.environ["PBS_JOBID"], evaluations={})
    write(folder / "result.json", result)
    try:
        for kind in ("random", "domain"):
            child = Path(r["children"][kind]["folder"])
            request = read(child / "request.json")
            receipt = evaluation.completed(request, request["source"]["record"])
            if receipt:
                proof = receipt.get("origin", {}).get("journal")
                if proof:
                    from .validation_journal import dependency_versions
                    manifest = read(Path(proof["path"]) / "manifest.json")
                    need(manifest["runtime"]["dependencies"] == dependency_versions(),
                         "Completed backfill dependency versions changed")
                result["evaluations"][kind] = dict(status="PASS", evaluated_step=receipt["step"],
                    optimizer_updates=0, checkpoints_written=0, reused=True,
                    receipt=str(evaluation.receipt_path(request, request["source"]["record"])))
                write(folder / "result.json", result)
                continue
            # execute() owns its isolated stage.lock and runs a NEW phase
            # process. Ray must release all eight GPUs between the two models.
            wait_gpus(1)
            with (child / "pbs.log").open("a") as out:
                subprocess.run([sys.executable, "-I", str(child / "entry.py"), "execute", "--request",
                    str(child / "request.json")], cwd=r["root"], stdout=out, stderr=subprocess.STDOUT, check=True)
            proof = read(child / "train/result.json")
            need(proof["status"] == "PASS" and proof["optimizer_updates"] == 0, "Backfill child failed")
            result["evaluations"][kind] = proof
            write(folder / "result.json", result)
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        result["finished_utc"] = now()
        write(folder / "result.json", result)


def tick(area, pbs=None):
    """At most one allocation; retries only confirmed 271/PBS walltime exits."""
    area, pbs = Path(area), pbs or PBS()
    with locked(area / "controller.lock"):
        state = read(area / "state.json")
        def save(**changes):
            state.update(changes, updated_utc=now())
            write(area / "state.json", state)
        if state["phase"] in ("PASS", "BLOCKED", "EXPIRED"):
            return state
        if state.get("job"):
            row = pbs.observe(state["job"])
            save(observation=row)
            if not finished(row):
                return state
            folder = Path(state["job"]["folder"])
            result = read(folder / "result.json") if (folder / "result.json").exists() else {}
            if str(row.get("Exit_status")) == "0" and result.get("status") == "PASS":
                r = validate_request(folder / "request.json")
                for kind, child in r["children"].items():
                    request = read(Path(child["folder"]) / "request.json")
                    need(evaluation.completed(request, state["sources"][kind]), "Missing complete backfill receipt")
                save(phase="PASS", result=result)
                return state
            if row.get("Exit_status") is None:
                raise Unknown("Backfill ended without a usable PBS exit status")
            if not retryable(row):
                save(phase="BLOCKED", error="Evaluation ended without retryable PBS exit", result=result)
                return state
            save(job=None, phase="RETRY")
        if expired(state["deadline_utc"]) and not state.get("pending"):
            save(phase="EXPIRED")
            return state
        if not state.get("pending"):
            save(pending=make_attempt(area, state), phase="SUBMITTING")
        item = state["pending"]
        job = pbs.submit(item["folder"], item["job_name"])
        save(job=job, pending=None, phase="QUEUED", attempts=state["attempts"] + 1, error=None)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("allocation", "execute"))
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    globals()[args.action](args.request)
