"""Retry one explicitly identified checkpoint export killed before runtime start.

This command is an operator-requested exception for terminal PBS exit 265. It
does not broaden the guardian's automatic 271/walltime retry policy. The existing
guardian owns submission, subsequent export/test phases, and evaluation caches.
"""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

POLICY = "checkpoint-export-sigkill-once-v1"


def inspect_export(c, task, expected_job, pbs):
    from d4_teachers import checkpoint_evaluation as extra
    from d4_teachers.common import below, is_fenced, need, read, safe_id
    from d4_teachers.source import validate_request
    from d4_training_continuation.scheduler import ID, finished

    root = Path(c["path"])
    need(task in extra.TARGETS, "Select a registered checkpoint export task")
    need(ID.fullmatch(expected_job), "Invalid expected PBS job ID")
    need(not is_fenced(root, task), "Explicit stop fence prevents export retry")
    plan = extra.plan(c)
    need(plan and task in plan["targets"], "Checkpoint evaluation is not registered")
    run = root / "runs" / task
    state = read(run / "state.json")
    token = POLICY + ":" + task + ":" + expected_job
    receipt = root / "repairs" / POLICY / (task + "_" + safe_id(expected_job) + ".json")
    previous = read(receipt) if receipt.exists() else None
    if previous:
        need(
            previous["policy"] == POLICY
            and previous["campaign_identity"] == c["identity"]
            and previous["task"] == task
            and previous["job_id"] == expected_job,
            "Retry receipt belongs to another task/job/campaign",
        )
    if state.get("explicit_export_retry") == token:
        need(previous is not None, "Committed retry is missing its audit receipt")
        return dict(already=True, task=task, job_id=expected_job, receipt=str(receipt))
    job = state.get("job") or {}
    need(
        state["status"] in ("BLOCKED_APPLICATION", "BLOCKED_UNKNOWN")
        and state["phase"] == "export"
        and not state.get("pending")
        and job.get("job_id") == expected_job,
        "Task changed or already owns another attempt; no retry was registered",
    )
    folder = below(job["folder"], run / "jobs")
    need(
        not (folder / "result.json").exists()
        and not (run / "export.json").exists()
        and not (run / "test.json").exists(),
        "Runtime/output evidence exists; this startup-only retry does not apply",
    )
    # PBS.observe checks the current owner, queue, name and exact submission path.
    row = pbs.observe(job)
    need(
        finished(row) and str(row.get("Exit_status")) == "265",
        "Expected job is not freshly confirmed terminal with Exit_status=265",
    )
    request = validate_request(folder / "request.json")
    need(
        request["role"] == "export"
        and request.get("task") == task
        and request["campaign_identity"] == c["identity"]
        and request["run_dir"] == str(run),
        "Job request is not the selected campaign/task export",
    )
    need(
        extra.verifier(c) == plan["verifier_identity"],
        "Final-test verifier changed since checkpoint evaluation registration",
    )
    return dict(
        already=False,
        task=task,
        job_id=expected_job,
        receipt=str(receipt),
        token=token,
        state=state,
        observation=row,
    )


def rearm_export(c, task, expected_job, pbs):
    """Called by retry_repair while control and guardian leases are held."""
    from d4_teachers.common import is_fenced, lock, need, now, read, write

    root = Path(c["path"])
    run = root / "runs" / task
    with lock(run / "writer.lock"):
        checked = inspect_export(c, task, expected_job, pbs)
        if checked["already"]:
            return dict(status="ALREADY_REARMED", **checked)
        state = checked["state"]
        receipt = Path(checked["receipt"])
        # The audit write precedes the state commit. A crash between these writes
        # is replayable; the state token prevents a second rearm after commit.
        if not receipt.exists():
            write(
                receipt,
                dict(
                    policy=POLICY,
                    campaign_identity=c["identity"],
                    task=task,
                    job_id=expected_job,
                    before=state,
                    observation=checked["observation"],
                    at_utc=now(),
                ),
            )
        need(not is_fenced(root, task), "Explicit stop appeared during export retry")
        need(read(run / "state.json") == state, "Task state changed during retry")
        updated = dict(state)
        if updated.get("error"):
            updated["previous_error"] = dict(
                message=updated.pop("error"), observed_utc=state.get("observed_utc")
            )
        updated.update(
            status="READY",
            previous_job=state["job"],
            job=None,
            pending=None,
            pbs=checked["observation"],
            explicit_export_retry=checked["token"],
            recovered_utc=now(),
        )
        write(run / "state.json", updated)
    return dict(
        status="REARMED_EXPORT",
        task=task,
        previous_job=expected_job,
        receipt=str(receipt),
    )


def retry_export(path, task, expected_job, pbs=None):
    from d4_teachers import retry_repair
    from d4_teachers.source import load_campaign
    from d4_training_continuation.scheduler import PBS

    root = Path(path).resolve()
    c = load_campaign(root)
    pbs = pbs or PBS()
    checked = inspect_export(c, task, expected_job, pbs)
    installer = None
    if not checked["already"]:
        installer = partial(rearm_export, task=task, expected_job=expected_job, pbs=pbs)
    guardian = retry_repair.repair(root, pbs=pbs, install_under_lease=installer)
    return dict(
        status="ALREADY_REARMED" if checked["already"] else "EXPORT_RETRY_REGISTERED",
        task=task,
        previous_job=expected_job,
        receipt=checked["receipt"],
        guardian=guardian,
        note="One explicit export retry; automatic 271/walltime recovery is unchanged.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task", choices=("math_step50", "cp_step80"), required=True)
    parser.add_argument("--expected-job", required=True)
    args = parser.parse_args()
    print(
        json.dumps(retry_export(args.campaign, args.task, args.expected_job), indent=2)
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
