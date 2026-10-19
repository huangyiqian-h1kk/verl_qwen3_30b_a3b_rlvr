#!/usr/bin/env python3
"""Read-only D4 job/controller evidence; stdlib only, including when piped to python -I -."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import socket
import subprocess

BASE = "qwen30b_d4_10k_uniform_ctx64k_seed42_v1"
RUNS = {"random_single": "", "domain_single": "_db_v1", "common_check": "_n2_v1",
        "random_dual": "_n2_once_v1", "domain_dual": "_db_n2_once_v1"}
JOB_ID = re.compile(r"[0-9]+(?:\.[A-Za-z0-9_.-]+)?\Z")
STATE_KEYS = ("phase", "status", "controller_status", "host", "pid", "identity",
              "started_utc", "window_started_utc", "deadline_utc", "last_check_utc",
              "last_job_state", "last_exit_status", "stopped_utc", "stop_reason",
              "last_error", "wait_reason", "single_blocked", "check_blocked",
              "handoff_blocked", "single_query_error", "single", "check", "dual",
              "common_check_job", "job_id", "job_dir", "previous_candidate",
              "pending_job", "pending_check", "resubmissions", "submissions",
              "check_submissions", "managed_by", "controller_dir", "monitor_window_hours",
              "operational_limits_version")
RECORD_KEYS = ("status", "action", "stage", "mode", "qualify", "job_id", "identity",
               "check_identity", "source_identity", "error", "reason", "completed_step",
               "grpo_steps_completed", "expected_nodes", "walltime", "started_utc",
               "finished_utc", "at_utc", "step", "path", "checkpoint", "checkpoint_at_submission",
               "single_final_step", "single_step", "dual_step", "returncode", "stderr",
               "version", "monitor_default_hours", "future_random_training_walltime")


def select(obj, keys):
    return {k: obj[k] for k in keys if k in obj}


def read_json(path):
    try:
        with path.open("rb") as handle:
            raw = handle.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("JSON exceeds the 2 MiB diagnostic limit")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("Expected a JSON object")
        return obj
    except (OSError, ValueError) as exc:
        return {"read_error": str(exc)}


def tail(path, lines=40, limit=256 * 1024):
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, 2)
            start = max(0, size - limit)
            handle.seek(start)
            raw = handle.read(limit)
        if start:
            raw = raw.partition(b"\n")[2]  # Discard a possible partial first line.
        text = raw.decode("utf-8", errors="replace").splitlines()
        chosen = text[-lines:]
        return {"lines": [s[:2000] for s in chosen], "file_bytes": size,
                "truncated": bool(start or len(text) > lines or any(len(s) > 2000 for s in chosen))}
    except OSError as exc:
        return {"read_error": str(exc)}


def log_evidence(path, lines):
    result = tail(path, lines)
    if "read_error" in result:
        return result
    # Search only the bounded final 256 KiB, not multi-GB training logs.
    scan = tail(path, 10000)
    if "read_error" in scan:
        result["scan_error"] = scan["read_error"]
        return result
    recent = scan["lines"]
    hits = [i for i, line in enumerate(recent)
            if re.search(r"Traceback|\b(?:\w*Error|\w*Exception)\b|FAIL|Killed|SIGTERM|SIGKILL", line)]
    indices = set()
    for i in (hits[:1] + hits[-2:]):
        indices.update(range(max(0, i - 3), min(len(recent), i + 8)))
    result["error_context_from_final_256KiB"] = [recent[i] for i in sorted(indices)]
    return result


def lock_status(path):
    """Probe the existing lock without creating files or trusting a remote PID."""
    try:
        with path.open("rb") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"held": True}
            fcntl.flock(handle, fcntl.LOCK_UN)
            return {"held": False}
    except FileNotFoundError:
        return {"held": False, "missing": True}
    except OSError as exc:
        return {"held": None, "error": str(exc)}


def command(args, root, timeout=15):
    try:
        p = subprocess.run(args, cwd=root, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"query_error": str(exc)}


def parse_qstat(text):
    """PBS wrappers may prepend a table/warning; require one actual Jobs object."""
    decoder = json.JSONDecoder()
    found = []
    for m in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("Jobs"), dict):
            found.append(value["Jobs"])
    if len(found) != 1:
        raise ValueError("Expected one PBS Jobs mapping, found " + str(len(found)))
    return found[0]


def query_job(root, jid):
    response = command(["qstat", "-xf", "-F", "json", jid], root)
    try:
        if response.get("returncode") != 0:
            raise ValueError(response.get("query_error") or response.get("stderr") or "qstat failed")
        jobs = parse_qstat(response["stdout"])
        # A numeric ID may be qualified by the server. Never silently switch servers.
        matches = [(key, value) for key, value in jobs.items()
                   if key == jid or ("." not in jid and key.split(".")[0] == jid)]
        if len(matches) != 1 or not isinstance(matches[0][1], dict):
            raise ValueError("PBS did not return exactly the requested job")
        key, row = matches[0]
        keys = ("Job_Name", "Job_Owner", "job_state", "queue", "Exit_status", "comment",
                "qtime", "stime", "obittime", "run_count", "exec_host", "exec_vnode", "Submit_arguments")
        result = {"job_id": key, **select(row, keys)}
        for group in ("Resource_List", "resources_used"):
            fields = ("walltime", "ncpus", "ngpus", "nodect", "mem", "select", "place")
            nested = row.get(group, {})
            result[group] = {k: nested[k] if isinstance(nested, dict) and k in nested else row[group + "." + k]
                             for k in fields if (isinstance(nested, dict) and k in nested) or group + "." + k in row}
        if "stderr" in response and response["stderr"]:
            result["query_warning"] = response["stderr"][-2000:]
        return result
    except ValueError as exc:
        return {"job_id": jid, "query_error": str(exc),
                "raw_output_tail": response.get("stdout", "")[-2000:],
                "note": "Unknown/history unavailable is not success and is not permission to resubmit."}


def job_ids(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if (k == "job_id" or k.endswith("_job")) and isinstance(v, str) and JOB_ID.fullmatch(v):
                yield v
            else:
                yield from job_ids(v)
    elif isinstance(value, list):
        for v in value:
            yield from job_ids(v)


def records(directory, names):
    output = {}
    for name in names:
        path = directory / name
        if path.is_file():
            row = read_json(path)
            output[name] = row if "read_error" in row else select(row, RECORD_KEYS)
    return output


def control_evidence(directory, managed, lines):
    state = read_json(directory / "state.json")
    output = {"directory": str(directory), "state": select(state, (*STATE_KEYS, "read_error")),
              "lock": lock_status(directory / ("controller.lock" if managed else "monitor.lock")),
              "stop_requested": (directory / "stop.request").exists()}
    if managed:
        plan = read_json(directory / "plan.json")
        output["plan"] = {"identity": plan.get("identity"), "hours": plan.get("hours"),
                          "poll_seconds": plan.get("poll_seconds"), "read_error": plan.get("read_error")}
        output["records"] = records(directory, ("acceptance.json", "readiness.json", "qdel.json",
             "decision.json", "matched.json", "superseded.json", "runtime_limits_30h_24h/config.json"))
        events = []
        for line in tail(directory / "events.jsonl", 10000).get("lines", []):
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get("event") != "OBSERVATION":
                    events.append(row)
            except ValueError:
                continue
        output["recent_events"] = events[-12:]
    else:
        pending = directory / "pending_submission.json"
        if pending.is_file():
            output["pending_submission"] = read_json(pending)
    output["log"] = log_evidence(directory / ("controller.log" if managed else "monitor.log"), lines)
    return output


def collect(root, extra_jobs=(), lines=40, recent_jobs=1):
    root = Path(root).resolve()
    parent = root / "outputs/multidomain"
    result = {"collected_at_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
              "root": str(root), "read_only": True,
              "note": "Non-atomic snapshot. Lock state is instantaneous; recorded RUNNING alone is not liveness. "
                      "Checkpoint/acceptance records are displayed, not revalidated. PBS times are unconverted.",
              "controllers": {}, "single_monitors": {}, "jobs": {}}
    ids = set(extra_jobs)
    by_id = {}
    for role, suffix in RUNS.items():
        run = parent / (BASE + suffix)
        if role.endswith("dual") or role.endswith("single"):
            managed = role.endswith("dual")
            directory = run / ("handoff" if managed else "monitor")
            evidence = control_evidence(directory, managed, lines)
            result["controllers" if managed else "single_monitors"][role] = evidence
            ids.update(job_ids(evidence))
        # Only known run directories, never recursively scan checkpoints/model tensors.
        found = []
        for marker in sorted((run / "jobs").glob("*/job_id.txt")):
            try:
                if not marker.resolve().is_relative_to(root):
                    continue
                jid = marker.read_text().strip()
            except OSError:
                continue
            if JOB_ID.fullmatch(jid):
                by_id.setdefault(jid, []).append((role, marker.parent))
                found.append(jid)
        ids.update(found[-recent_jobs:] if recent_jobs else [])
    for jid in sorted(ids, key=lambda s: (int(s.split(".")[0]), s)):
        item = {"records": []}
        for key, folders in by_id.items():
            if key != jid and not ("." not in jid and key.split(".")[0] == jid):
                continue
            for role, folder in folders:
                request = read_json(folder / "request.json")
                details = {"role": role, "folder": str(folder), "request": select(request, (*RECORD_KEYS, "read_error")),
                           "reports": records(folder, ("result.json", "ready_to_handoff.json", "resume_source.json",
                               "first_update/result.json", "first_update/qualification.json", "first_update/resume_source.json",
                               "restored_update/result.json", "restored_update/qualification.json", "restored_update/resume_source.json",
                               "train/result.json", "runtime_limits.json"))}
                contract = request.get("contract")
                details["bound_check_identity"] = contract.get("check_identity") if isinstance(contract, dict) else None
                details["result_exists"] = (folder / "result.json").is_file()
                item["records"].append(details)
        if len(item["records"]) > 1:
            item["record_warning"] = "Multiple records match this job; inspect all, do not silently adopt one."
        result["jobs"][jid] = item
    with ThreadPoolExecutor(max_workers=3) as pool:
        for jid, pbs in zip(result["jobs"], pool.map(lambda j: query_job(root, j), result["jobs"])):
            item = result["jobs"][jid]
            item["pbs"] = pbs
            for record in item["records"]:
                # Successful historical common checks need no long log; still report PASS and exit.
                if record["role"] == "common_check" and str(pbs.get("Exit_status")) == "0":
                    continue
                record["pbs_log"] = log_evidence(Path(record["folder"]) / "pbs.log", lines)
    result["qstat_now"] = command(["qstat"], root)
    result["git"] = {name: command(["git", *args], root) for name, args in (
        ("branch", ["branch", "--show-current"]), ("head", ["rev-parse", "HEAD"]),
        ("tracked_changes", ["--no-optional-locks", "status", "--short", "--untracked-files=no"]))}
    result["collection_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--job", action="append", default=[], help="Also query this historical PBS job (repeatable)")
    parser.add_argument("--tail-lines", type=int, default=40)
    parser.add_argument("--recent-jobs", type=int, default=1, help="Include N newest job records per run")
    args = parser.parse_args(argv)
    if not args.root.is_dir() or not 1 <= args.tail_lines <= 200 or not 0 <= args.recent_jobs <= 10:
        parser.error("Expected an existing root, 1..200 tail lines and 0..10 recent jobs")
    if any(not JOB_ID.fullmatch(j) for j in args.job):
        parser.error("Invalid PBS job ID")
    print(json.dumps(collect(args.root, args.job, args.tail_lines, args.recent_jobs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
