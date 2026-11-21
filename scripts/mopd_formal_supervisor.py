"""Supervise a qualified formal execution with a separately pinned guardian policy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid


def load(record_path):
    """Only the two Git-pinned controller modules override the qualified snapshot."""
    record = json.loads(record_path.read_text())
    base = record_path.parents[2]
    manifest = base / "manifest.json"
    os.environ["D4_MOPD_EXECUTION_RECORD"] = str(record_path)
    sys.path[:0] = [record["runtime"], json.loads(manifest.read_text())["verl_src"]]
    import d4_mopd
    from d4_mopd.assets import verify
    m = verify(manifest)
    patch = Path(__file__).resolve().parents[1] / "d4_mopd"
    # This directory is an operator archive containing exactly these two files.
    if {p.name for p in patch.glob("*.py")} != {"control.py", "walltime_retry.py"}:
        raise ValueError("Run from a pinned guardian-only archive")
    d4_mopd.__path__.insert(0, str(patch))
    from d4_mopd import control
    if Path(control.__file__).resolve() != patch / "control.py":
        raise ValueError("Guardian override did not load")
    return m


def rearm(m, expected_jobs, step):
    from d4_mopd import ARMS, control
    from d4_mopd.execution import control_root, inspect_resume
    from d4_teachers.common import lock, need, now, read, write
    from d4_training_continuation.common import locked
    from d4_training_continuation.scheduler import PBS
    root = control_root(m)
    pbs = PBS()
    original = pbs.command
    def observed(args, timeout=20):
        print(now(), "START", args, flush=True)
        try:
            return original(args, timeout=max(90, timeout))
        finally:
            print(now(), "END", args, flush=True)
    pbs.command = observed
    with lock(root / "launch.lock", wait=60), locked(root / "guardian.lock"), lock(root / "states.lock", wait=120):
        need(not (root / "STOP").exists(), "Execution is stopped")
        states = read(root / "states.json")
        need(not any(s.get("pending") for s in states.values()), "A submission is pending")
        evidence = {}
        for arm, jid in zip(ARMS, expected_jobs, strict=True):
            task = states["train:" + arm]
            job = task["job"]
            run = control.task_root(m, "train", arm)
            folder = Path(job["folder"])
            need(task["status"] == "BLOCKED" and job["job_id"] == jid and folder.parent == run / "attempts",
                 "Expected blocked training job changed")
            request = read(folder / "request.json")
            need(request.get("arm") == arm and request.get("phase") == "train"
                 and request.get("folder") == str(folder) and request.get("identity") == m["identity"]
                 and request.get("execution_identity") == m["execution"]["execution_identity"]
                 and request.get("runtime_identity") == m["execution"]["runtime_identity"]
                 and request.get("execution_profile") == m["execution"]["profile"]["name"], "Job identity changed")
            control.qualification_passed(m, arm)
            row = pbs.observe(job)
            decision = control.retry_assessment(row, control.failure_result(job), job, m["identity"])
            need(decision == dict(allowed=True, reason="pbs_walltime_with_captured_ray_sigterm"),
                 "Real job failed walltime replay: " + repr(decision))
            with locked(run / "writer.lock"):
                proof = read(run / "checkpoints" / f"global_step_{step}" / "mopd_complete.json")
                resume = inspect_resume(run, m, arm, proof["sampler_identity"])
                need(resume["step"] == step and not (run / "receipt.json").exists(), "Resume point changed")
            evidence[arm] = dict(job=job, old_state=task.copy(), pbs=row, assessment=decision, resume=resume)
        # Check other controllers and their allocations before releasing either task.
        with control.formal_submission(m, "train", pbs):
            pass
        journal = root / "guardian_recoveries" / (uuid.uuid4().hex + ".json")
        write(journal, dict(at_utc=now(), execution_identity=m["execution"]["execution_identity"],
                            operator_source=str(Path(__file__).resolve()), evidence=evidence))
        for arm in ARMS:
            task = states["train:" + arm]
            # Keep the old job bound: the next real tick observes it again and
            # performs the ordinary journalled READY -> SUBMITTED transition.
            task.update(status="RECHECK", previous_error=task.pop("error", None), recovery=str(journal))
        write(root / "states.json", states)
        print(now(), "Rearmed", str(journal), flush=True)


def snapshot(m):
    from d4_mopd import ARMS, control
    from d4_mopd.execution import control_root
    from d4_teachers.common import active, now, read
    root = control_root(m)
    states = read(root / "states.json")
    rows, complete, attention = {}, True, []
    for arm in ARMS:
        for phase in ("train", "test"):
            key = phase + ":" + arm
            state = states.get(key, {})
            job = state.get("job") or {}
            run = control.task_root(m, phase, arm)
            receipt = read(run / "receipt.json") if (run / "receipt.json").is_file() else {}
            bound = (receipt.get("status") == "PASS" and receipt.get("arm") == arm
                     and receipt.get("scope") == phase and receipt.get("identity") == m["identity"]
                     and receipt.get("execution_identity") == m["execution"]["execution_identity"]
                     and receipt.get("runtime_identity") == m["execution"]["runtime_identity"]
                     and receipt.get("execution_profile") == m["execution"]["profile"]["name"])
            complete &= state.get("status") == "DONE" and bound
            info = dict(status=state.get("status"), job=job.get("job_id"), receipt_bound=bound,
                        error=state.get("error"), observation_error=state.get("observation_error"))
            if state.get("status") == "BLOCKED" or (state.get("status") == "DONE" and not bound):
                attention.append(key)
            if phase == "train":
                info["checkpoint"] = max((int(p.parent.name.split("_")[-1]) for p in
                    (run / "checkpoints").glob("global_step_*/mopd_complete.json")), default=None)
                if job:
                    folder = Path(job["folder"])
                    resume = folder / "resume.json"
                    info["resumed_step"] = read(resume).get("step") if resume.is_file() else None
                    info["latest_update"] = max((int(p.stem[7:]) for p in folder.glob("update_*.json")
                                                 if p.stem[7:].isdigit()), default=None)
            rows[key] = info
    return dict(at_utc=now(), pid=os.getpid(), host=socket.gethostname(), states=rows,
                stopped=(root / "STOP").exists(), guardian_held=active(root / "guardian.lock"),
                heartbeat=read(root / "heartbeat.json") if (root / "heartbeat.json").exists() else None,
                attention=attention, complete=complete)


def supervise(m, record):
    from d4_mopd.execution import control_root
    from d4_teachers.common import active, lock, now, read, write
    root = control_root(m)
    logdir = root / "formal_supervision"
    logdir.mkdir(exist_ok=True)
    with lock(logdir / "observer.lock"):
        while True:
            started = time.monotonic()
            try:
                data = snapshot(m)
                if not data["complete"] and not data["stopped"]:
                    with lock(root / "launch.lock", wait=60):
                        states = read(root / "states.json")
                        runnable = any(s["status"] not in ("DONE", "BLOCKED") for s in states.values())
                        if runnable and not (root / "STOP").exists() and not active(root / "guardian.lock"):
                            args = [sys.executable, "-I", str(Path(__file__).resolve()), "guardian", str(record)]
                            with (root / "guardian.log").open("ab") as output:
                                child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=output,
                                                         stderr=subprocess.STDOUT, start_new_session=True)
                            print(now(), "guardian launch", args, child.pid, flush=True)
                            deadline = time.monotonic() + 120
                            while time.monotonic() < deadline:
                                if child.poll() is not None:
                                    raise RuntimeError("Guardian exited at startup: " + str(child.returncode))
                                ack = root / "guardian_startup.json"
                                receipt = read(ack) if ack.is_file() else {}
                                if receipt.get("pid") == child.pid and receipt.get("host") == socket.gethostname() and receipt.get("at_utc", "") >= data["at_utc"]:
                                    break
                                time.sleep(1)
                            else:
                                raise RuntimeError("Guardian startup not yet confirmed")
                    data = snapshot(m)
            except Exception as exc:
                data = dict(at_utc=now(), pid=os.getpid(), host=socket.gethostname(), observation_error=repr(exc))
            write(logdir / "latest.json", data)
            print(json.dumps(data), flush=True)
            if data.get("complete") or data.get("stopped"):
                return
            time.sleep(max(0, 1800 - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("rearm", "supervise", "guardian"))
    parser.add_argument("record", type=Path)
    parser.add_argument("--jobs", nargs=2, metavar=("MIX", "ONLY"))
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    def log_subprocess(event, values):
        if event == "subprocess.Popen":
            print(json.dumps(dict(event=event, at_utc=datetime.now(timezone.utc).isoformat(),
                                  argv=values[1]), default=str), file=sys.stderr, flush=True)
    sys.addaudithook(log_subprocess)
    record = args.record.resolve()
    m = load(record)
    if args.action == "rearm":
        if not args.jobs or not args.step:
            parser.error("rearm requires both job IDs and the expected checkpoint step")
        rearm(m, args.jobs, args.step)
    elif args.action == "supervise":
        supervise(m, record)
    else:
        from d4_mopd.control import guardian
        guardian(m["manifest_path"])


if __name__ == "__main__":
    main()
