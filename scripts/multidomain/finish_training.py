#!/usr/bin/env python3
"""Retire the two D4 setups, or inventory their storage. No checkpoint deletion.

stop: fence owners, install a persistent frozen-runtime guard, cancel only
identity-checked experiment jobs, and verify leases/PBS. inventory: read-only
inode accounting and a review-only checkpoint retention proposal. Stdlib only
except the existing, tested pidfd helper when a local live owner needs a signal.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
import uuid

BASE = "qwen30b_d4_10k_uniform_ctx64k_seed42_v1"
QUEUE = "R9920261000"
VERSION = "2026-10-26.finish.1"
KINDS = ("domain", "random")
CP = re.compile(r"global_step_(\d+)\Z")
JOB = re.compile(r"\d+\.[A-Za-z0-9_.-]+\Z")
TENSORS = {".pt", ".distcp", ".safetensors", ".bin"}


def now():
    return datetime.now(timezone.utc).isoformat()


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(path, parent):
    path, parent = Path(path).absolute(), Path(parent).resolve()
    need(".." not in path.parts, "Parent traversal in owned path")
    need(path.is_relative_to(parent) and path != parent, "Outside owned tree: " + str(path))
    for part in (path, *path.parents):
        if part == parent:
            break
        need(not part.is_symlink(), "Symlink in owned path: " + str(part))
    return path


def paths(root, kind):
    base = root / "outputs/multidomain"
    single = base / (BASE + ("_db_v1" if kind == "domain" else ""))
    old = base / (BASE + ("_db" if kind == "domain" else "") + "_n2_once_v1")
    run = base / (BASE + ("_db" if kind == "domain" else "") + "_continuation_v1")
    return dict(run=run, control=run / "control", single=single,
                monitor=single / "monitor", old=old, handoff=old / "handoff")


def lease(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.is_symlink(), "Symlinked lease: " + str(path))
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BaseException:
        handle.close()
        raise


def held(path):
    if not path.exists():
        return False
    need(not path.is_symlink(), "Symlinked lease")
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def wait_lease(path, seconds=120):
    end = time.monotonic() + seconds
    while True:
        try:
            return lease(path)
        except BlockingIOError:
            need(time.monotonic() < end, "Still busy; no success claimed: " + str(path))
            time.sleep(.5)


def signal_owner(control, root, kind, ops):
    state = read(control / "state.json")
    host = str(state.get("host", ""))
    need(host.split(".")[0] == socket.gethostname().split(".")[0],
         "Live legacy continuation owner is on " + host +
         "; it does not implement a stop file. Run this SAME stop command there. No PBS jobs cancelled.")
    pid = state.get("pid")
    need(isinstance(pid, int) and pid > 1 and pid != os.getpid(), "Invalid owner PID")
    operation = str(state["operation"])
    need(re.fullmatch(r"[A-Za-z0-9_.-]+", operation), "Unsafe operation")
    entry = control / "operations" / operation / "controller_entry.py"
    sys.path.insert(0, str(ops))
    from d4_training_continuation import process_signal
    try:
        descriptor = process_signal.open_process(pid)
    except ProcessLookupError:
        return
    try:
        proc = Path("/proc") / str(pid)
        try:
            uid = proc.stat().st_uid
            argv = [os.fsdecode(x) for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        except FileNotFoundError:
            return
        expected = ["-I", str(entry), "_run", "--root", str(root), "--kind", kind]
        need(uid == os.getuid() and argv[1:] == expected,
             "Live owner identity differs; no signal sent")
        try:
            process_signal.send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            pass
    finally:
        os.close(descriptor)


def fence_template(root, kind, report_dir):
    """Old CLI resume/extend reads this plan: even it enters the guarded daemon."""
    control = paths(root, kind)["control"]
    state = read(control / "state.json")
    operation = safe(control / "operations" / state["operation"], control)
    planfile = operation / "plan.json"
    plan = read(planfile)
    need(plan.get("kind") == kind and Path(plan["root"]).resolve() == root, "Wrong retirement plan")
    if plan.get("retirement_guard") == VERSION:
        return
    original = safe(Path(plan["template"]), operation)
    for name, checksum in plan["template_files"].items():
        need(sha(safe(original / name, original)) == checksum, "Frozen source changed: " + name)
    relative = "runtime/d4_training_continuation/cli.py"
    text = (original / relative).read_text()
    marker = control / "retired.json"
    for name in ("spawn", "daemon"):
        signature = f"def {name}(root, kind):\n"
        need(text.count(signature) == 1, "Unrecognized frozen CLI function: " + name)
        guard = ("    # Persistent user-requested retirement; no automatic restart.\n"
                 f"    if Path({str(marker)!r}).exists():\n"
                 "        return\n")
        text = text.replace(signature, signature + guard)
    target = operation / "retirement" / VERSION / "template"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.parent / (".staging_" + uuid.uuid4().hex)
        try:
            shutil.copytree(original, temporary, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (temporary / relative).write_text(text)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    need((target / relative).read_text() == text, "Retirement template differs")
    checksums = dict(plan["template_files"], **{relative: sha(target / relative)})
    for name, checksum in checksums.items():
        need(sha(safe(target / name, target)) == checksum, "Staged retirement source differs")
    write(report_dir / (kind + "_plan_before.json"), plan)
    write(planfile, dict(plan, template=str(target), template_files=checksums,
                         retirement_guard=VERSION))


def command(argv, timeout=45):
    env = dict(os.environ, LC_ALL="C")
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def pbs_jobs():
    errors = []
    for _ in range(2):
        try:
            proc = command(["qstat", "-f", "-F", "json"])
            need(proc.returncode == 0, "qstat: " + proc.stderr[:500])
            value = json.loads(proc.stdout)
            need(isinstance(value, dict) and ("Jobs" in value or "pbs_server" in value), "Invalid PBS JSON")
            need(isinstance(value.get("Jobs", {}), dict), "Invalid PBS Jobs")
            return value.get("Jobs", {})
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
    raise RuntimeError("PBS query unknown; not treated as no jobs: " + " | ".join(errors))


def in_scope_script(root, script):
    base = root / "outputs/multidomain"
    try:
        rel = script.relative_to(base)
    except ValueError:
        return False
    return bool(rel.parts and (rel.parts[0] == BASE or rel.parts[0].startswith(BASE + "_")
                               or rel.parts[0] == "validation_backfills"))


def owned_jobs(root, rows):
    result, ambiguous = {}, []
    for jid, row in rows.items():
        if row.get("Job_Owner", "").split("@")[0] != getpass.getuser():
            continue
        args = str(row.get("Submit_arguments", "")).replace("\n", "").replace("\t", "")
        try:
            tokens = shlex.split(args)
        except ValueError:
            tokens = []
        script = Path(tokens[0]) if len(tokens) == 1 else None
        scoped = script is not None and in_scope_script(root, script)
        suspect = str(row.get("Job_Name", "")).startswith("0390_d4")
        if row.get("job_state") in ("F", "C"):
            continue
        if not scoped:
            if suspect:
                ambiguous.append(dict(job_id=jid, reason="Cannot bind D4 job to owned submission path", row=row))
            continue
        need(JOB.fullmatch(jid) and row.get("queue") == QUEUE, "Unexpected owned PBS identity/queue")
        safe(script, root)
        need(script.name == "stage.pbs" and script.is_file(), "Missing owned stage.pbs")
        recorded = script.parent / "job_id.txt"
        need(recorded.exists() and recorded.read_text().strip() == jid,
             "Unrecorded/ambiguous PBS submission; inspect before cancellation: " + jid)
        names = re.findall(r"^#PBS\s+-N\s+(\S+)\s*$", script.read_text(), re.M)
        need(names == [row.get("Job_Name")], "PBS job name differs from recorded script: " + jid)
        need(row.get("job_state") in ("Q", "R", "H", "E", "B", "S", "T", "W", "U"), "Unknown PBS state")
        result[jid] = dict(row, bound_script=str(script))
    need(not ambiguous, "Unidentified D4 jobs; no cancellation: " + json.dumps(ambiguous, ensure_ascii=False))
    return result


def stop(root, ops, output, seconds=180):
    base = root / "outputs/multidomain"
    report = dict(version=VERSION, started_utc=now(), status="STOPPING", root=str(root), qdel={})
    write(output / "stop.json", report)
    try:
        with ExitStack() as stack:
            stack.enter_context(lease(base / "continuation_operations/apply.lock"))
            # An archiver may install a new controller, so stop/fence it first.
            service = base / "checkpoint_archive_service"
            service.mkdir(exist_ok=True)
            (service / "stop.request").touch()
            for kind in KINDS:
                p = paths(root, kind)
                need((p["control"] / "state.json").is_file(), "Missing continuation state for " + kind)
                for name in ("monitor", "handoff"):
                    p[name].mkdir(parents=True, exist_ok=True)
                    (p[name] / "stop.request").touch()
            stack.enter_context(wait_lease(service / "watch.lock"))
            stack.enter_context(lease(service / "install.lock"))
            for kind in KINDS:
                control = paths(root, kind)["control"]
                if held(control / "controller.lock"):
                    signal_owner(control, root, kind, ops)
                stack.enter_context(wait_lease(control / "controller.lock"))
            # All old resubmit and handoff leases are fenced before qdel.
            for kind in KINDS:
                p = paths(root, kind)
                for path in (p["handoff"] / "controller.lock", p["monitor"] / "monitor.lock",
                             p["single"] / "submission.lock", p["old"] / "submission.lock"):
                    stack.enter_context(wait_lease(path))
                archive = p["run"] / "checkpoint_archive"
                stack.enter_context(wait_lease(archive / "archive.lock"))
            for campaign in (base / "validation_backfills").glob("*/state.json"):
                stack.enter_context(wait_lease(campaign.parent / "controller.lock"))
            for kind in KINDS:
                control = paths(root, kind)["control"]
                write(output / (kind + "_state_before.json"), read(control / "state.json"))
                write(control / "retired.json", dict(version=VERSION, at_utc=now(),
                      reason="user_requested_end_of_training", report=str(output)))
                fence_template(root, kind, output)
                state = read(control / "state.json")
                state.update(controller_status="STOPPED", monitoring_finished=True,
                             phase="STOPPED_BY_USER", retired_utc=now(), retirement_report=str(output))
                write(control / "state.json", state)
                monitor = paths(root, kind)["monitor"] / "state.json"
                if monitor.exists():
                    old = read(monitor)
                    write(monitor, dict(old, status="STOPPED", stop_reason="user_requested_end_of_training"))
            report["owners_fenced"] = True
            jobs = owned_jobs(root, pbs_jobs())
            report["jobs_before"] = jobs
            write(output / "stop.json", report)
            for jid, row in jobs.items():
                # E already means PBS is ending it; wait for disappearance.
                if row["job_state"] == "E":
                    continue
                report["qdel"][jid] = dict(intent_utc=now())
                write(output / "stop.json", report)
                try:
                    p = command(["qdel", jid])
                    report["qdel"][jid].update(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
                except (OSError, subprocess.TimeoutExpired) as error:
                    report["qdel"][jid]["error"] = str(error)
                write(output / "stop.json", report)
            end = time.monotonic() + seconds
            while True:
                remaining = owned_jobs(root, pbs_jobs())
                report["remaining_jobs"] = remaining
                write(output / "stop.json", report)
                if not remaining:
                    break
                need(time.monotonic() < end, "Jobs are still present; inspect stop.json and rerun SAME command")
                time.sleep(3)
            # A scheduler disappearance alone does not prove a writer released its lock.
            for kind in KINDS:
                p = paths(root, kind)
                targets = [p["single"], p["old"], p["run"]]
                targets += list((p["run"] / "lineages").glob("*"))
                for target in targets:
                    if (target / "stage.lock").exists():
                        stack.enter_context(wait_lease(target / "stage.lock"))
            report.update(status="STOPPED", finished_utc=now(), owners_fenced=True,
                          archive_service_stopped=True, checkpoints_deleted=0)
            write(output / "stop.json", report)
    except BaseException as error:
        report.update(status="INCOMPLETE", error=type(error).__name__ + ": " + str(error), updated_utc=now())
        write(output / "stop.json", report)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def scan_roots(root):
    base = root / "outputs/multidomain"
    found = []
    for path in sorted(base.iterdir()):
        if path.is_symlink() or not path.is_dir():
            continue
        if path.name == BASE or path.name.startswith(BASE + "_"):
            group = "domain" if path.name.startswith(BASE + "_db_") else "random"
            if path.name == BASE + "_n2_v1":
                group = "shared"
            found.append((path, group))
        elif path.name in ("validation_backfills", "checkpoint_archive_service", "continuation_operations"):
            found.append((path, "shared"))
        elif path.name == "validation_backfill_sources":
            for child in sorted(path.iterdir()):
                if child.is_dir() and not child.is_symlink():
                    group = "random" if child.name.startswith("random_step_") else "domain" if child.name.startswith("domain_step_") else "shared"
                    found.append((child, group))
    return found


def valid_inventory(payload, inventory):
    failures = []
    if not inventory:
        return ["Empty inventory"]
    for name, expected in inventory.items():
        try:
            file = safe(payload / name, payload)
            st = file.stat()
            need(file.is_file() and st.st_size == expected["bytes"] and st.st_mtime_ns == expected["mtime_ns"], "Size/mtime mismatch")
            if expected.get("sha256") and st.st_size <= 8 * 1024 * 1024:
                need(sha(file) == expected["sha256"], "Metadata checksum mismatch")
        except (OSError, RuntimeError, KeyError) as error:
            failures.append(dict(file=name, error=str(error)))
    return failures


def retention(root):
    """Select canonical copies. Only proposals; never remove/convert an archive."""
    protected, result, errors = {}, {}, []
    for kind in KINDS:
        p = paths(root, kind)
        try:
            state = read(p["control"] / "state.json")
            archive = p["run"] / "checkpoint_archive"
            index = read(archive / "index.json")
            winners, full = defaultdict(list), set()
            for protocol, cohort in index["cohorts"].items():
                for metric, row in cohort.items():
                    winners[row["key"]].append(protocol + ":" + metric)
                    if metric in ("overall", "overall_full"):
                        full.add(row["key"])
            objects = {}
            for file in (archive / "objects").glob("*/manifest.json"):
                obj = read(file)
                payload = safe(Path(obj["payload"]), file.parent)
                failures = valid_inventory(payload, obj["inventory"])
                if failures:
                    errors.append(dict(setup=kind, path=str(payload), errors=failures))
                objects[obj["key"]] = dict(obj, inventory_errors=failures)
            authority = state.get("authority") or {}
            live = authority.get("lineage")
            latest = sorted((o for o in objects.values() if o["mode"] == "full" and o.get("lineage") == live
                             and not o["inventory_errors"]), key=lambda o: o["step"], reverse=True)
            need(latest, "No validated full archive for current authority")
            # Also inspect the current rolling tip; a final checkpoint may be newer than the last bridge scan.
            tip = latest[0]
            run = safe(Path(authority["run"]), root)
            for marker in (run / "checkpoints").glob("global_step_*/COMPLETE_CONTINUATION.json"):
                rec = read(marker)
                if rec.get("lineage") == live and rec["step"] > tip["step"] and not valid_inventory(marker.parent, rec["files"]):
                    tip = dict(step=rec["step"], payload=str(marker.parent), key=None)
            if tip.get("key"):
                full.add(tip["key"])
            else:
                protected[tip["payload"]] = dict(mode="full", reasons=["latest_authoritative_full"], model_files=[])
            for key, obj in objects.items():
                reasons = winners.get(key, [])
                if key == tip.get("key"):
                    reasons = reasons + ["latest_authoritative_full"]
                if key in full:
                    need(obj["mode"] == "full" and not obj["inventory_errors"], "Protected full archive is incomplete")
                    mode = "full"
                elif key in winners:
                    need(not obj["inventory_errors"], "Protected model archive is incomplete")
                    mode = "model"
                elif obj["mode"] == "unverified_weights":
                    mode, reasons = "review", ["unverified_historical_weights_do_not_delete_automatically"]
                else:
                    continue
                protected[obj["payload"]] = dict(mode=mode, reasons=reasons, model_files=sorted(obj.get("model_inventory", {})))
            result[kind] = dict(latest_full_step=tip["step"], latest_full_path=tip["payload"],
                                winners=index["cohorts"], inventory_checks="size/mtime and small-file checksums; not tensor-load test")
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            errors.append(dict(setup=kind, error=str(error)))
    return protected, result, errors


def checkpoint_parent(path):
    for parent in path.parents:
        if CP.fullmatch(parent.name):
            return parent
    return None


def removable_payload(file, checkpoint, protected):
    if checkpoint is None:
        return False
    policy = protected.get(str(checkpoint))
    if policy:
        if policy["mode"] in ("full", "review"):
            return False
        if str(file.relative_to(checkpoint)) in policy["model_files"]:
            return False
    # Preserve all small provenance/config/receipt records even for old attempts.
    return file.suffix in TENSORS or file.name == ".metadata"


def component(file, checkpoint):
    if checkpoint is None:
        return "non_checkpoint"
    relative = file.relative_to(checkpoint)
    if relative.parts[:2] == ("actor", "model"):
        return "model"
    if relative.parts[:2] == ("actor", "optimizer"):
        return "optimizer"
    if relative.parts[:2] == ("actor", "extra"):
        return "lr_rng_other_training_state"
    if relative.name == "data.pt":
        return "dataloader"
    return "other_checkpoint_files"


def inventory(root, output):
    roots = scan_roots(root)
    protected, selected, errors = retention(root)
    nodes, checkpoints, symlinks = {}, {}, []
    counts = Counter()
    for directory, group in roots:
        print("[0390] Inventory: " + str(directory), flush=True)
        def onerror(error):
            errors.append(dict(path=error.filename, error=str(error)))
        for parent, dirs, files in os.walk(directory, followlinks=False, onerror=onerror):
            parent = Path(parent)
            links = [name for name in dirs if (parent / name).is_symlink()]
            dirs[:] = sorted(set(dirs) - set(links))
            for file in [parent] + [parent / n for n in sorted(files + links)]:
                try:
                    st = file.lstat()
                    key = (st.st_dev, st.st_ino)
                    node = nodes.setdefault(key, dict(size=st.st_size, allocated=st.st_blocks * 512,
                        nlink=st.st_nlink, count=0, proposed=0, groups=set(), components=set(), regular=stat.S_ISREG(st.st_mode)))
                    need((st.st_size, st.st_blocks * 512, st.st_nlink) == (node["size"], node["allocated"], node["nlink"]),
                         "File changed during inventory")
                    node["count"] += 1
                    node["groups"].add(group)
                    counts["entries"] += 1
                    if stat.S_ISLNK(st.st_mode):
                        symlinks.append(str(file))
                        continue
                    if not node["regular"]:
                        continue
                    cp = checkpoint_parent(file)
                    node["components"].add(component(file, cp))
                    proposed = removable_payload(file, cp, protected)
                    node["proposed"] += int(proposed)
                    if cp:
                        row = checkpoints.setdefault(str(cp), dict(path=str(cp), setup=group,
                            step=int(CP.fullmatch(cp.name).group(1)), logical_bytes=0,
                            linked_allocated_bytes=0, files=0, proposed_payload_bytes=0,
                            retention=protected.get(str(cp), dict(mode="review_delete_payload", reasons=["not_selected_best_or_final"], model_files=[]))))
                        row["logical_bytes"] += st.st_size
                        row["linked_allocated_bytes"] += st.st_blocks * 512
                        row["files"] += 1
                        row["proposed_payload_bytes"] += st.st_size if proposed else 0
                except (OSError, RuntimeError) as error:
                    errors.append(dict(path=str(file), error=str(error)))
    totals = dict(unique_logical_file_bytes=0, unique_allocated_bytes=0, unique_files=0,
                  proposed_reclaimable_allocated_bytes=0, external_hardlink_inodes=0)
    groups = {g: dict(unique_logical_file_bytes=0, unique_allocated_bytes=0, exclusive_allocated_bytes=0,
                     shared_with_other_scopes_bytes=0) for g in ("random", "domain", "shared")}
    components = defaultdict(lambda: dict(unique_logical_file_bytes=0, unique_allocated_bytes=0))
    for node in nodes.values():
        totals["unique_allocated_bytes"] += node["allocated"]
        if node["regular"]:
            label = next(iter(node["components"])) if len(node["components"]) == 1 else "shared_across_components"
            components[label]["unique_logical_file_bytes"] += node["size"]
            components[label]["unique_allocated_bytes"] += node["allocated"]
            totals["unique_files"] += 1
            totals["unique_logical_file_bytes"] += node["size"]
            totals["external_hardlink_inodes"] += int(node["nlink"] > node["count"])
            if node["proposed"] == node["count"] == node["nlink"]:
                totals["proposed_reclaimable_allocated_bytes"] += node["allocated"]
        for group in node["groups"]:
            groups[group]["unique_allocated_bytes"] += node["allocated"]
            groups[group]["unique_logical_file_bytes"] += node["size"] if node["regular"] else 0
            groups[group]["exclusive_allocated_bytes" if len(node["groups"]) == 1 else "shared_with_other_scopes_bytes"] += node["allocated"]
    if errors:
        totals["proposed_reclaimable_allocated_bytes"] = None
    report = dict(version=VERSION, at_utc=now(), root=str(root), totals=totals, scopes=groups, components=dict(components),
        inventory_status="PARTIAL" if errors else "COMPLETE",
        roots=[dict(path=str(p), setup=g) for p, g in roots], selection=selected,
        symlinks_not_followed=symlinks, errors=errors, checkpoints_deleted=0,
        caveats=["Read-only proposal; not authorization to delete or conversion of archive manifests.",
                 "Allocated bytes use st_blocks*512, deduplicated by (device,inode). Not backend snapshot/compression quota.",
                 "Per-setup unique totals can overlap; use totals.unique_allocated_bytes for the union.",
                 "Reclaim estimate requires ALL nlink references to be proposed; outside links are excluded.",
                 "Shared base models, datasets, root-level caches/wandb and previous analysis bundles are outside this run-output scope.",
                 "Do not remove whole job/control/archive directories; keep metadata, evaluations, logs and code."])
    write(output / "storage.json", report)
    write(output / "checkpoint_retention_review.json", sorted(checkpoints.values(), key=lambda r: (r["setup"], r["step"], r["path"])))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    for group, row in groups.items():
        print(f"STORAGE_{group.upper()}_TiB={row['unique_allocated_bytes'] / 2**40:.6f}")
    print(f"STORAGE_UNION_TiB={totals['unique_allocated_bytes'] / 2**40:.6f}")
    if totals["proposed_reclaimable_allocated_bytes"] is not None:
        print(f"REVIEW_ONLY_RECLAIMABLE_TiB={totals['proposed_reclaimable_allocated_bytes'] / 2**40:.6f}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stop", "inventory", "stop-and-inventory"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ops", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    need((root / "outputs/multidomain").is_dir(), "Wrong training root")
    ops = (args.ops or root.parent / "data_mixture_rl_4d_ops_capacity").resolve()
    output = args.output_dir or root / "outputs/analysis" / ("d4_finish_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output = output.absolute()
    need(not output.exists(), "Use a NEW output directory; stop is safe to rerun")
    output.mkdir(parents=True)
    try:
        if args.action in ("stop", "stop-and-inventory"):
            stop(root, ops, output)
        if args.action in ("inventory", "stop-and-inventory"):
            report = inventory(root, output)
            need(not report["errors"], "Storage inventory incomplete; inspect storage.json errors. No checkpoints deleted.")
    finally:
        bundle = output.with_name(output.name + ".tar.gz")
        with tarfile.open(bundle, "w:gz") as archive:
            archive.add(output, arcname=output.name)
        print("RESULT_BUNDLE=" + str(bundle), flush=True)


if __name__ == "__main__":
    main()
