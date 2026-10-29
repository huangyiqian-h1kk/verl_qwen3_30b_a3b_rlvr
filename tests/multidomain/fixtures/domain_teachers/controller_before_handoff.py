"""Campaign-scoped PBS guardian. Unknown observations never grant another submit."""

import os
from pathlib import Path
import socket
import time
from . import DOMAINS
from .common import (
    active,
    age_seconds,
    file_sha,
    is_fenced,
    lock,
    need,
    now,
    read,
    write,
)
from .source import load_campaign, make_job
from .exporting import manifest, validate_export
from d4_training_continuation import checkpoints as ck
from d4_training_continuation.scheduler import PBS, Unknown, finished, retryable

TASKS = (*DOMAINS, "base")
RETRY_POLICY = "four-domain-external-interruption-v1"
OBSERVATION_POLICY = "persistent-pbs-observation-v1"
# Unknown observations and held jobs still need their existing owner. They must
# keep this loop alive even when every other teacher has already finished/failed.
TERMINAL = {
    "DONE",
    "STOPPED",
    "BLOCKED_APPLICATION",
    "BLOCKED_RETRIES",
    "BLOCKED_ARTIFACT",
}


def state_path(c, domain):
    return Path(c["path"]) / "runs" / domain / "state.json"


def progress(c, domain, phase):
    run = Path(c["path"]) / "runs" / domain
    if phase == "train":
        cp = ck.latest(run, read(run / "run.json")["lineage"])
        step = cp["step"] if cp else 0
        # Completed eval batches count as progress across a walltime interruption.
        batches = sum(
            read(p).get("completed_rows", 0)
            for p in (run / "evaluations").glob("*/progress.json")
        )
        return step * 100000 + batches
    if phase == "test":
        return sum(
            read(p).get("completed_rows", 0)
            for p in (run / "evaluations").glob("test_*/progress.json")
        )
    return int((run / "export.json").exists())


def artifact(c, domain, phase):
    run = Path(c["path"]) / "runs" / domain
    if phase == "train":
        done = read(run / "training_complete.json")
        cp = ck.latest(run, read(run / "run.json")["lineage"])
        need(
            done["status"] == "PASS" and cp and done["completed_step"] == cp["step"],
            "Training completion is unproven",
        )
    elif phase == "export":
        validate_export(read(run / "export.json"))
    else:
        from .evaluation import read_receipt

        receipt = read_receipt(run / "test.json")
        need(
            receipt["status"] == "PASS"
            and receipt["split"] == "test"
            and all(d in receipt["scores"] for d in DOMAINS),
            "Cross-domain test is incomplete",
        )


def next_phase(c, phase):
    if phase == "train" and c["policy"]["auto_export"]:
        return "export"
    if phase == "export" and c["policy"]["auto_test"]:
        return "test"
    return None


def reconcile(c, domain, state, pbs):
    """Resolve intents/jobs first, including after stop; never discard an unknown ID."""
    folder = Path(c["path"]) / "runs" / domain
    try:
        if state.get("pending"):
            pending = state["pending"]
            recovered = pbs.recover(pending["folder"], pending["job_name"])
            if recovered:
                state.update(job=recovered, pending=None)
                write(state_path(c, domain), state)
            elif is_fenced(c["path"], domain):
                state.update(pending=None, status="STOPPED")
                return
            else:
                # Submit only a durable, never-attempted intent, after a fresh fence check.
                need(not is_fenced(c["path"], domain), "Submission fenced")
                state["job"] = pbs.submit(pending["folder"], pending["job_name"])
                state["pending"] = None
                write(state_path(c, domain), state)
        job = state.get("job")
        if not job:
            if is_fenced(c["path"], domain):
                state["status"] = "STOPPED"
            return
        row = pbs.observe(job)
        state["pbs"] = row
        if state.get("error"):
            state["previous_error"] = dict(
                message=state.pop("error"), observed_utc=state.get("observed_utc")
            )
        if is_fenced(c["path"], domain):
            if pbs.cancel(job, folder / "cancel.json"):
                state.update(status="STOPPED", previous_job=job, job=None)
            else:
                state["status"] = "STOPPING"
            return
        if row["job_state"] == "H":
            state["status"] = "BLOCKED_HELD"
            return
        if row["job_state"] in ("Q", "R", "E", "B", "S"):
            state["status"] = "QUEUED" if row["job_state"] == "Q" else "RUNNING"
            return
        if not finished(row):
            raise Unknown("Unrecognized PBS state: " + str(row.get("job_state")))
        result = Path(job["folder"]) / "result.json"
        if (
            result.exists()
            and read(result).get("status") == "PASS"
            and str(row.get("Exit_status")) == "0"
        ):
            try:
                artifact(c, domain, state["phase"])
            except Exception as e:
                state.update(status="BLOCKED_ARTIFACT", error=str(e))
                return
            phase = next_phase(c, state["phase"])
            state.update(
                previous_job=job,
                job=None,
                status="READY" if phase else "DONE",
                no_progress_retries=0,
                last_progress=0,
                phase=phase or state["phase"],
            )
            return
        if retryable(row):
            value = progress(c, domain, state["phase"])
            count = (
                0
                if value > state["last_progress"]
                else state["no_progress_retries"] + 1
            )
            state.update(
                last_progress=value,
                no_progress_retries=count,
                previous_job=job,
                job=None,
                # Match the original 4-domain formal-training retry policy.
                # Lost work before a durable save is not an application failure.
                # Keep the counter as telemetry, never as a retry veto.
                status="READY",
            )
        else:
            state.update(
                status="BLOCKED_APPLICATION",
                error="PBS ended without verified success or an authorized retry condition",
            )
    except Unknown as e:
        state.update(status="BLOCKED_UNKNOWN", error=str(e))
    except Exception as e:
        state.update(status="BLOCKED_ARTIFACT", error=type(e).__name__ + ": " + str(e))
    finally:
        state["observed_utc"] = now()
        write(state_path(c, domain), state)


def tick(c, pbs=None):
    pbs = pbs or PBS()
    states = {d: read(state_path(c, d)) for d in TASKS}
    for domain, state in states.items():
        reconcile(c, domain, state, pbs)
    # Count only this campaign's unresolved jobs/intents, including held/unknown.
    occupied = sum(
        bool(
            s.get("pending")
            or (
                s.get("job")
                and (s["status"] == "BLOCKED_UNKNOWN" or not finished(s.get("pbs", {})))
            )
        )
        for s in states.values()
    )
    for domain, state in states.items():
        if (
            state["status"] != "READY"
            or occupied >= c["policy"]["max_concurrent_jobs"]
            or is_fenced(c["path"], domain)
        ):
            continue
        if domain == "base" and (
            not c["policy"]["auto_test"]
            or not all(
                (Path(c["path"]) / "runs" / d / "export.json").exists() for d in DOMAINS
            )
        ):
            if not c["policy"]["auto_test"]:
                state["status"] = "DONE"
                write(state_path(c, domain), state)
            continue
        try:
            pending = make_job(c, domain, state["phase"])
        except Exception as e:
            state.update(
                status="STOPPED"
                if is_fenced(c["path"], domain)
                else "BLOCKED_ARTIFACT",
                error=str(e),
            )
            write(state_path(c, domain), state)
            continue
        state.update(pending=pending, status="SUBMITTING")
        write(state_path(c, domain), state)  # Persist BEFORE qsub.
        occupied += 1
        reconcile(c, domain, state, pbs)
    try:
        manifest(c)
    except Exception as e:
        write(Path(c["path"]) / "manifest_error.json", dict(error=str(e), at_utc=now()))
    write(
        Path(c["path"]) / "heartbeat.json",
        dict(
            at_utc=now(),
            host=socket.gethostname(),
            pid=os.getpid(),
            retry_policy=RETRY_POLICY,
            observation_policy=OBSERVATION_POLICY,
            controller_file=str(Path(__file__).resolve()),
            controller_sha256=file_sha(__file__),
            states={d: s["status"] for d, s in states.items()},
        ),
    )
    # A blocked teacher is independent; the other teachers continue. Unknown and
    # held jobs remain owned and are re-observed on every subsequent poll.
    return states


def guardian(path):
    c = load_campaign(path)
    proof = read(Path(path) / "preflight.json")
    need(
        proof["status"] == "PASS" and proof["campaign_identity"] == c["identity"],
        "Run CPU preflight first",
    )
    with lock(Path(path) / "guardian.lock"):
        write(
            Path(path) / "guardian_runtime.json",
            dict(
                at_utc=now(),
                host=socket.gethostname(),
                pid=os.getpid(),
                retry_policy=RETRY_POLICY,
                observation_policy=OBSERVATION_POLICY,
                controller_file=str(Path(__file__).resolve()),
                controller_sha256=file_sha(__file__),
            ),
        )
        while True:
            with lock(Path(path) / "control.lock", wait=60):
                states = tick(c)
            unfinished = [
                s
                for d, s in states.items()
                if s["status"] not in TERMINAL
                and not (
                    d == "base"
                    and s["status"] == "READY"
                    and all(states[x]["status"] in TERMINAL for x in DOMAINS)
                )
            ]
            if not unfinished:
                return states
            time.sleep(c["policy"]["poll_seconds"])


def status(path):
    c = load_campaign(path, verify=False)
    heartbeat = Path(path) / "heartbeat.json"
    beat = read(heartbeat) if heartbeat.exists() else None
    alive = active(Path(path) / "guardian.lock")
    return dict(
        campaign=c["policy"]["campaign"],
        guardian_lock_held=alive,
        heartbeat=beat,
        heartbeat_stale=not beat
        or age_seconds(beat["at_utc"]) > max(120, c["policy"]["poll_seconds"] * 4),
        states={d: read(state_path(c, d)) for d in TASKS},
        fenced=is_fenced(path),
    )


def stop(path, domain=None):
    c = load_campaign(path, verify=False)
    need(domain is None or domain in TASKS, "Unknown teacher")
    target = Path(path) if domain is None else Path(path) / "runs" / domain
    write(
        target / "STOP.json",
        dict(at_utc=now(), scope=domain or "campaign", reason="explicit user stop"),
    )
    # The durable fence is already visible to an in-flight qsub/start/resume.
    # If no guardian owns the lock, this process performs the same reconciliation.
    if not active(Path(path) / "guardian.lock"):
        with lock(Path(path) / "guardian.lock"):
            with lock(Path(path) / "control.lock", wait=60):
                for d in TASKS if domain is None else (domain,):
                    reconcile(c, d, read(state_path(c, d)), PBS())
    return status(path)


def reenable(path, domain=None):
    c = load_campaign(path, verify=False)
    need(domain is None or domain in TASKS, "Unknown teacher")
    with lock(Path(path) / "control.lock", wait=60):
        target = Path(path) if domain is None else Path(path) / "runs" / domain
        fence = target / "STOP.json"
        need(fence.exists(), "No explicit stop fence to reenable")
        for d in TASKS if domain is None else (domain,):
            state = read(state_path(c, d))
            need(
                not state.get("job") and not state.get("pending"),
                "Confirm all stopped jobs terminal before reenable",
            )
            if state["status"] == "STOPPED":
                state["status"] = "READY"
                write(state_path(c, d), state)
        fence.rename(fence.with_name("STOP_reenabled_" + str(time.time_ns()) + ".json"))
