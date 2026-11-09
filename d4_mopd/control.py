"""Shared-filesystem supervisor and identity-bound PBS submission for two arms."""
import os
from contextlib import ExitStack
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import uuid

from . import ARMS
from .assets import verify
from d4_teachers.common import active, lock, need, now, read, write
from d4_training_continuation.common import atomic_text, locked
from d4_training_continuation.scheduler import PBS, Unknown, finished
from d4_teachers.controller import retry_reason


def task_root(m, phase, arm):
    need(phase in ("qualification", "train") and arm in ARMS, "Unknown MOPD task")
    return Path(m["manifest_path"]).parent / ("qualification" if phase == "qualification" else "runs") / arm


def entry_command(m):
    root = Path(m["manifest_path"]).parent
    bootstrap = ("import sys; sys.path[:0]=" + repr([str(root / "runtime"), m["verl_src"]])
                 + "; from d4_mopd.cli import main; main()")
    return [m["python"], "-I", "-c", bootstrap]


def render_pbs(m, folder, job_name):
    q = shlex.quote
    runtime = Path(m["manifest_path"]).parent / "runtime"
    return "\n".join([
        "#!/bin/bash", "#PBS -P gcg51557", "#PBS -q R9920261000", "#PBS -v RTYPE=rt_HF",
        "#PBS -l select=2", "#PBS -l walltime=24:00:00", "#PBS -N " + job_name,
        "#PBS -r n", "#PBS -c n", "#PBS -j oe", "#PBS -k oe", "set -euo pipefail",
        "exec > >(tee -a " + q(str(folder / "pbs.log")) + ") 2>&1",
        "if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi",
        "module load gcc/13.2.0 hpcx/2.20",
        "awk '!seen[$0]++' \"$PBS_NODEFILE\" > " + q(str(folder / "hosts")),
        "[[ $(wc -l < " + q(str(folder / "hosts")) + ") -eq 2 ]]",
        "mpirun -np 2 --map-by ppr:1:node --bind-to none --hostfile " + q(str(folder / "hosts"))
        + " bash " + q(str(runtime / "d4_mopd/node.sh")) + " " + q(str(folder / "request.json")), ""])


def new_attempt(m, phase, arm):
    run = task_root(m, phase, arm)
    folder = run / "attempts" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    request = dict(manifest=m["manifest_path"], identity=m["identity"], phase=phase, arm=arm,
                   folder=str(folder), run_dir=str(run), job_name="0390_mp_" + ("q" if phase == "qualification" else "t")
                   + ("mix" if arm == "domain_mix" else "only"))
    write(folder / "request.json", request)
    # No mutable checkout is placed on PYTHONPATH, including inside Ray workers.
    (folder / "entry.py").write_text("import sys\nsys.path[:0]=" + repr([
        str(Path(m["manifest_path"]).parent / "runtime"), m["verl_src"]])
        + "\nfrom d4_mopd.cli import main\nmain()\n")
    (folder / "python.txt").write_text(m["python"] + "\n")
    (folder / "stage.pbs").write_text(render_pbs(m, folder, request["job_name"]))
    return dict(folder=str(folder), job_name=request["job_name"])


def tick(m, state, pbs):
    for key, s in state.items():
        if s["status"] in ("DONE", "BLOCKED"):
            continue
        run = task_root(m, s["phase"], s["arm"])
        try:
            job = s.get("job")
            if not job:
                # Durable pending directory precedes qsub. An ambiguous qsub is
                # recovered at the same path, never retried with a new directory.
                pending = s.get("pending")
                if pending is None:
                    pending = new_attempt(m, s["phase"], s["arm"])
                    s["pending"] = pending
                    write(Path(m["manifest_path"]).parent / "states.json", state)
                s["job"] = pbs.submit(pending["folder"], pending["job_name"])
                s["pending"] = None
                s["status"] = "SUBMITTED"
                continue
            row = pbs.observe(job)  # validates owner, queue, job name and exact script path
            s["pbs"] = {k: row.get(k) for k in ("job_state", "Exit_status", "comment")}
            s.pop("observation_error", None)
            if not finished(row):
                s["status"] = row["job_state"]
                continue
            result_path = Path(job["folder"]) / "result.json"
            result = read(result_path) if result_path.exists() else {}
            if str(row.get("Exit_status")) == "0" and result.get("status") == "PASS":
                need(result.get("identity") == m["identity"], "Unbound success receipt")
                if s["phase"] == "qualification":
                    need(result.get("resumed_from") == 4 and result["step"] == 8, "Missing fresh-process resume proof")
                proof = dict(result, pbs_job=job, pbs=row, scope=s["phase"], at_utc=now())
                write(run / "receipt.json", proof)
                s["status"] = "DONE"
            elif str(row.get("Exit_status")) == "0" and result.get("status") == "RESUME_REQUIRED":
                need(result.get("identity") == m["identity"] and s["phase"] == "qualification"
                     and result.get("step") == 4, "Unexpected pause")
                s.update(status="READY", previous_job=job, job=None)
            elif str(row.get("Exit_status")) == "0" and result.get("status") == "CONTROL_PAUSED":
                need(result.get("identity") == m["identity"], "Unbound pause receipt")
                s.update(status="READY", previous_job=job, job=None)
            else:
                reason = retry_reason(row, job)
                if reason and result.get("status") != "FAIL":
                    s.update(status="READY", previous_job=job, job=None, retry_reason=reason,
                             external_interruptions=s.get("external_interruptions", 0)+1)
                else:
                    s.update(status="BLOCKED", error=result.get("error") or "Unclassified terminal job; inspect logs")
        except Unknown as exc:
            s["observation_error"] = str(exc)  # Keep job/pending; no duplicate qsub.
        except Exception as exc:
            s.update(status="BLOCKED", error=str(exc))
        finally:
            s["observed_utc"] = now()
            write(Path(m["manifest_path"]).parent / "states.json", state)
    return state


def guardian(manifest_path):
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    with ExitStack() as lease:
        lease.enter_context(locked(root / "guardian.lock"))
        while True:
            if (root / "STOP").exists():
                with lock(root / "launch.lock", wait=60):
                    if (root / "STOP").exists():
                        lease.close()
                        return
            with lock(root / "states.lock", wait=120):
                states = tick(m, read(root / "states.json"), PBS())
                done = all(s["status"] in ("DONE", "BLOCKED") for s in states.values())
                write(root / "heartbeat.json", dict(at_utc=now(), host=socket.gethostname(), pid=os.getpid(),
                                                     states={k: v["status"] for k, v in states.items()}, all_terminal=done))
            if done:
                # Serialize retirement with registration. A start on another login
                # node cannot see an old guardian lock and then lose its guardian.
                with lock(root / "launch.lock", wait=60):
                    with lock(root / "states.lock", wait=120):
                        current = read(root / "states.json")
                        if all(s["status"] in ("DONE", "BLOCKED") for s in current.values()):
                            lease.close()
                            return
            time.sleep(m["spec"]["poll_seconds"])


def start(manifest_path, phase, arms):
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    need(not (root / "STOP").exists(), "Experiment is stopped; use resume-control explicitly")
    proof = read(root / "cpu_preflight.json")
    need(proof["status"] == "PASS" and proof["identity"] == m["identity"], "CPU preflight has not passed")
    if phase == "train":
        for arm in arms:
            q = read(task_root(m, "qualification", arm) / "receipt.json")
            need(q["status"] == "PASS" and q["identity"] == m["identity"] and q["resumed_from"] == 4,
                 "GPU end-to-end qualification/reload is required before formal training: " + arm)
    with lock(root / "launch.lock", wait=60):
        with lock(root / "states.lock", wait=120):
            path = root / "states.json"
            state = read(path) if path.exists() else {}
            for arm in arms:
                key = phase + ":" + arm
                state.setdefault(key, dict(phase=phase, arm=arm, status="READY", job=None, pending=None))
            write(path, state)
        if not active(root / "guardian.lock"):
            with (root / "guardian.log").open("ab") as log:
                child = subprocess.Popen(entry_command(m) + ["guardian", "--manifest", str(manifest_path)],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=str(root))
            deadline = time.monotonic() + 30
            while not active(root / "guardian.lock") and time.monotonic() < deadline:
                need(child.poll() is None, "Guardian exited during startup; inspect guardian.log")
                time.sleep(0.1)
            need(active(root / "guardian.lock"), "Guardian did not acquire its lease; inspect guardian.log")
    return dict(status="REGISTERED", tasks=[phase+":"+a for a in arms], manifest=m["manifest_path"],
                note="Use status to confirm the guardian heartbeat and PBS submission")
