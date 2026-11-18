#!/usr/bin/env python3
"""One-time, journaled adoption of both qualified step-8 states into formal runs.

This operator tool does not edit an immutable runtime, start a guardian, submit a
job, remove STOP, or change qualification proofs. Prepare copies or hardlinks the
immutable files into independent staging directories. Activate requires stopped
controllers and terminal PBS jobs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from d4_mopd import ARMS, DOMAINS, checkpoints
from d4_mopd.execution import control_root, inspect_resume, validate_qualification, verify_record
from d4_mopd.metrics import SELECTION_KEY, select_best, source_metrics
from d4_mopd.spec import PlannedSampler
from d4_teachers.common import digest, file_sha, need, now, read, safe_id, verify_files, write
from d4_training_continuation.common import locked
from d4_training_continuation.scheduler import PBS, finished

SCHEMA = "d4-mopd-qualification-adoption-v1"


def emit(event):
    print(json.dumps(dict(at_utc=now(), **event), ensure_ascii=False), flush=True)


def command(argv, **kwargs):
    started = now()
    emit(dict(event="command_start", argv=argv, started_utc=started))
    try:
        result = subprocess.run(argv, capture_output=True, text=True, **kwargs)
    except BaseException as exc:
        emit(dict(event="command_end", argv=argv, started_utc=started,
                  ended_utc=now(), error=repr(exc)))
        raise
    emit(dict(event="command_end", argv=argv, started_utc=started, ended_utc=now(),
              returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    need(result.returncode == 0, "Command failed: " + repr(argv))
    return result.stdout.strip()


class AuditedPBS(PBS):
    def command(self, args, timeout=20):
        started = now()
        emit(dict(event="command_start", argv=args, started_utc=started))
        try:
            result = super().command(args, timeout)
        except BaseException as exc:
            emit(dict(event="command_end", argv=args, started_utc=started,
                      ended_utc=now(), error=repr(exc)))
            raise
        emit(dict(event="command_end", argv=args, started_utc=started,
                  ended_utc=now(), returncode=0, stdout=result))
        return result


def regular(path):
    path = Path(path)
    need(stat.S_ISREG(path.lstat().st_mode), "Expected a regular file, not a link: " + str(path))
    return path


def no_links(path, root):
    path, root = Path(path), Path(root)
    need(path == root or root in path.parents, "Path escapes adoption directory")
    for candidate in (path, *path.parents):
        need(not candidate.is_symlink(), "Symlink in adoption path: " + str(candidate))
        if candidate == root:
            break


def metadata(path):
    s = regular(path).stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                sha256=file_sha(path) if s.st_size <= 1048576 else None)


def tree(path):
    path = Path(path)
    need(path.is_dir() and not path.is_symlink(), "Expected a real directory: " + str(path))
    result = {}
    for child in sorted(path.rglob("*")):
        need(not child.is_symlink(), "Symlink in adoption tree: " + str(child))
        if child.is_file() and child.name != "writer.lock":
            result[str(child.relative_to(path))] = metadata(child)
    return result


def load_manifest(path, record_path):
    path, record_path = Path(path).resolve(), Path(record_path).resolve()
    m = read(path)
    need(m["manifest_path"] == str(path)
         and m["identity"] == digest({k: v for k, v in m.items() if k != "identity"}),
         "Manifest identity/path changed")
    verify_files(path.parent / "runtime", m["runtime_files"])
    verify_files(path.parent / "verifier", m["verifier_files"])
    for split, paths in m["data"].items():
        need(len(paths) == 1 and file_sha(paths[0]) == m["data_summary"][split]["sha256"],
             "Frozen data changed: " + split)
    need(command(["git", "-C", m["verl_src"], "rev-parse", "HEAD"], timeout=20) == m["verl_revision"],
         "Pinned upstream revision changed")
    need(not command(["git", "-C", m["verl_src"], "status", "--porcelain"], timeout=20),
         "Pinned upstream has local changes")
    m["execution"] = verify_record(m, record_path)
    s = m["spec"]
    need((s["train_batch_size"], s["rollout_n"], s["max_context_tokens"], s["updates"])
         == (2048, 1, 65536, 200), "Adoption requires the agreed full training budget")
    return m


def data_rows(m, split):
    import pyarrow.parquet as pq
    rows = pq.read_table(m["data"][split][0], columns=["sample_id", "domain", "source"]).to_pylist()
    need(digest([r["sample_id"] for r in rows]) == m["data_summary"][split]["ids_sha256"],
         "Frozen sample IDs changed: " + split)
    return rows


def baseline(m, arm, receipt, validation_rows):
    run = control_root(m) / "qualification" / arm
    attempt = run / "attempts" / receipt["qualification_segments"][0]["attempt_id"]
    path, request_path = attempt / "validation_0.json", attempt / "request.json"
    row, request = read(regular(path)), read(regular(request_path))
    expected = dict(manifest=m["manifest_path"], identity=m["identity"], phase="qualification", arm=arm,
                    folder=str(attempt), run_dir=str(run), execution_record=m["execution"]["record_path"],
                    execution_identity=m["execution"]["execution_identity"],
                    runtime_identity=m["execution"]["runtime_identity"],
                    execution_profile=m["execution"]["qualification_transition"][0])
    need(all(request.get(k) == v for k, v in expected.items()), "Baseline attempt is not the qualified 0->4 segment")
    need(all(row.get(k) == v for k, v in dict(status="PASS", step=0, identity=m["identity"], arm=arm,
                                             split="validation", test_used_for_selection=False).items()),
         "Invalid qualification baseline receipt")
    generations = regular(attempt / "validation_generations/0.jsonl")
    extra = {k: [] for k in ("reward", "mopd_source", "mopd_domain", "mopd_sample_id")}
    expected_rows = {r["sample_id"]: r for r in validation_rows}
    with generations.open() as stream:
        for line in stream:
            item = json.loads(line)
            need(item.get("step") == 0, "Baseline generations contain another step")
            sid = item["mopd_sample_id"]
            need(sid in expected_rows and item["mopd_domain"] == expected_rows[sid]["domain"]
                 and item["mopd_source"] == expected_rows[sid]["source"], "Baseline generation provenance changed")
            for key in extra:
                extra[key].append(item[key] if key != "reward" else item.get("reward", item["score"]))
    need(len(extra["mopd_sample_id"]) == len(set(extra["mopd_sample_id"])) == len(expected_rows)
         and set(extra["mopd_sample_id"]) == set(expected_rows), "Baseline is not the complete unique validation set")
    counts = Counter(extra["mopd_domain"])
    need(counts == {d: m["spec"]["validation_per_domain"] for d in DOMAINS}, "Baseline domain counts changed")
    calculated = source_metrics(extra)
    need(SELECTION_KEY in calculated and all(k in row["metrics"] and math.isclose(
        row["metrics"][k], v, rel_tol=1e-8, abs_tol=1e-8) for k, v in calculated.items()),
        "Baseline metrics disagree with the full generation evidence")
    canonical = dict(row, **checkpoints.evaluation_reference(None, m["identity"], arm, 0, None))
    return dict(source=str(path), source_sha256=file_sha(path), request=str(request_path),
                request_sha256=file_sha(request_path), generations=str(generations),
                generations_sha256=file_sha(generations), samples=len(expected_rows),
                domain_counts=dict(counts), canonical=canonical)


def controller_roots(m):
    base = Path(m["manifest_path"]).parent
    return [base] + sorted(p for p in (base / "executions").iterdir() if p.is_dir() and not p.name.startswith("."))


def states_at(root):
    path = Path(root) / "states.json"
    return read(path) if path.exists() else {}


def make_plan(m, operation, expected_jobs, git, copy_mode="copy"):
    need(copy_mode in ("copy", "hardlink"), "Unknown checkpoint copy mode")
    base, control = Path(m["manifest_path"]).parent, control_root(m)
    operation = safe_id(operation)
    folder = base / "qualification_adoptions" / operation
    need(not folder.is_symlink() and not folder.parent.is_symlink(), "Unsafe adoption directory")
    roots = controller_roots(m)
    states = {str(root): states_at(root) for root in roots}
    for arm in ARMS:
        state = states[str(control)].get("train:" + arm, {})
        need((state.get("job") or {}).get("job_id") == expected_jobs[arm]
             and not state.get("pending"), "Expected formal job changed: " + arm)
    train_rows, validation_rows = data_rows(m, "train"), data_rows(m, "validation")
    arms = {}
    for arm in ARMS:
        source_run = control / "qualification" / arm
        receipt_path = source_run / "receipt.json"
        receipt = validate_qualification(read(receipt_path), m, arm)
        sampler = PlannedSampler(train_rows, m["spec"], arm).identity
        source = checkpoints.latest(source_run / "checkpoints", m["identity"], arm, sampler)
        need(source == source_run / "checkpoints/global_step_8" and receipt["checkpoint"] == str(source),
             "Qualification does not end at the committed step 8")
        proof = read(source / "mopd_complete.json")
        need(proof["status"] == "PASS" and proof["step"] == 8 and proof["resumed_from"] == 4,
             "Unexpected checkpoint-8 lineage")
        segment = read(source / "qualification_segment.json")
        need(segment == receipt["qualification_segments"][1], "Checkpoint segment differs from qualification proof")
        destination = base / "runs" / arm
        need(not (destination / "receipt.json").exists(), "Formal training already has a completion receipt")
        need(not (destination / "qualification_adoption.json").exists(), "Formal run has already been adopted")
        arms[arm] = dict(source=str(source), destination=str(destination),
                         staging=str(folder / "staging" / arm), archive=str(folder / "archive" / arm),
                         sampler_identity=sampler, checkpoint_marker=proof, source_files=tree(source),
                         receipt=str(receipt_path), receipt_sha256=file_sha(receipt_path),
                         baseline=baseline(m, arm, receipt, validation_rows), old_files=tree(destination))
    payload = dict(schema=SCHEMA, operation=operation, folder=str(folder), identity=m["identity"], copy_mode=copy_mode,
                   manifest=m["manifest_path"], execution_record=m["execution"]["record_path"],
                   execution_identity=m["execution"]["execution_identity"], runtime_identity=m["execution"]["runtime_identity"],
                   expected_jobs=expected_jobs, git=git, arms=arms, controller_states=states,
                   source_step=8, first_update=9, final_update=200, formal_validation_steps=[0] + list(range(10, 201, 10)))
    return dict(payload, plan_identity=digest(payload))


def checked_plan(m, folder):
    folder = Path(folder)
    no_links(folder, Path(m["manifest_path"]).parent)
    plan = read(folder / "plan.json")
    need(plan.get("copy_mode", "copy") in ("copy", "hardlink"), "Unknown checkpoint copy mode")
    need(plan["schema"] == SCHEMA and plan["plan_identity"] == digest({k: v for k, v in plan.items() if k != "plan_identity"}),
         "Adoption plan changed")
    need(plan["manifest"] == m["manifest_path"] and plan["identity"] == m["identity"]
         and plan["execution_identity"] == m["execution"]["execution_identity"]
         and plan["runtime_identity"] == m["execution"]["runtime_identity"]
         and folder == Path(m["manifest_path"]).parent / "qualification_adoptions" / safe_id(plan["operation"]),
         "Plan belongs to another campaign/execution/path")
    for arm, item in plan["arms"].items():
        need(arm in ARMS and item["source"] == str(control_root(m) / "qualification" / arm / "checkpoints/global_step_8")
             and item["destination"] == str(Path(m["manifest_path"]).parent / "runs" / arm)
             and item["staging"] == str(folder / "staging" / arm)
             and item["archive"] == str(folder / "archive" / arm), "Adoption path changed")
        for key in ("source", "destination", "staging", "archive"):
            no_links(item[key], Path(m["manifest_path"]).parent)
    need(set(plan["arms"]) == set(ARMS), "Adoption requires both arms")
    return plan


def validate_sources(m, plan):
    for arm, item in plan["arms"].items():
        source = Path(item["source"])
        need(tree(source) == item["source_files"] and read(source / "mopd_complete.json") == item["checkpoint_marker"],
             "Qualification checkpoint changed: " + arm)
        need(checkpoints.latest(source.parent, m["identity"], arm, item["sampler_identity"]) == source,
             "Qualification latest checkpoint changed")
        need(file_sha(item["receipt"]) == item["receipt_sha256"], "Qualification receipt changed")
        validate_qualification(read(item["receipt"]), m, arm)
        b = item["baseline"]
        for field in ("source", "request", "generations"):
            need(file_sha(regular(b[field])) == b[field + "_sha256"], "Qualification baseline evidence changed")


def publish_new(path, payload):
    path = Path(path)
    if path.exists():
        need(read(path) == payload, "Refusing to replace existing adoption evidence: " + str(path))
    else:
        write(path, payload)


def progress(folder, **fields):
    write(Path(folder) / "progress.json", dict(at_utc=now(), **fields))


def hash_progress(path, folder, phase, completed_bytes, total_bytes):
    hasher, size, at = hashlib.sha256(), 0, time.monotonic()
    with regular(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            hasher.update(chunk)
            size += len(chunk)
            if time.monotonic() - at >= 5:
                progress(folder, phase=phase, file=str(path), file_bytes=size,
                         completed_bytes=completed_bytes, total_bytes=total_bytes)
                at = time.monotonic()
    return hasher.hexdigest()


def copy_with_progress(source, partial, folder, completed_bytes, total_bytes):
    # copy2 preserves the nanosecond mtimes embedded in native commit markers.
    # Only this operation's unpublished partial file may be restarted.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(shutil.copy2, source, partial)
        while True:
            try:
                future.result(timeout=2)
                return
            except FutureTimeout:
                progress(folder, phase="copying", file=str(source),
                         file_bytes=partial.stat().st_size if partial.exists() else 0,
                         completed_bytes=completed_bytes, total_bytes=total_bytes)


def hardlink_identity(source, target):
    a, b = regular(source).stat(), regular(target).stat()
    need((a.st_dev, a.st_ino) == (b.st_dev, b.st_ino) and a.st_nlink >= 2,
         "Hardlink does not share the qualified source inode")
    return dict(device=a.st_dev, inode=a.st_ino, minimum_links=2)


def prepare(m, plan):
    folder = Path(plan["folder"])
    no_links(folder / "operation.lock", Path(m["manifest_path"]).parent)
    folder.mkdir(parents=True, exist_ok=True)
    with locked(folder / "operation.lock"):
        publish_new(folder / "plan.json", plan)
        checked_plan(m, folder)
        hardlink = plan.get("copy_mode", "copy") == "hardlink"
        need(not (folder / "activation.json").exists(), "Activation has begun; resume activate instead")
        validate_sources(m, plan)
        ledger_path = folder / "copied_files.json"
        ledger = read(ledger_path) if ledger_path.exists() else {}
        total = sum(info["bytes"] for arm in plan["arms"].values() for info in arm["source_files"].values())
        completed = 0
        remaining = sum(info["bytes"] for arm in plan["arms"].values() for name, info in arm["source_files"].items()
                        if not (Path(arm["staging"]) / "checkpoints/global_step_8" / name).exists())
        need(hardlink or shutil.disk_usage(folder).free >= remaining + min(total // 100, 10 << 30),
             "Insufficient staging disk space")
        with ExitStack() as lease:
            for arm in ARMS:
                lease.enter_context(locked(control_root(m) / "qualification" / arm / "writer.lock"))
            for arm in ARMS:
                item = plan["arms"][arm]
                stage = Path(item["staging"])
                no_links(stage, folder)
                stage.mkdir(parents=True, exist_ok=True)
                for name, expected in sorted(item["source_files"].items(), key=lambda x: (x[0] == "mopd_complete.json", x[0])):
                    source, target = Path(item["source"]) / name, stage / "checkpoints/global_step_8" / name
                    no_links(target, folder)
                    key = arm + "/" + name
                    need(metadata(source) == expected, "Source file changed: " + str(source))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if hardlink:
                        need(source.stat().st_dev == target.parent.stat().st_dev,
                             "Hardlink adoption requires the same filesystem; copy fallback is disabled")
                        if not target.exists():
                            partial = target.with_name("." + target.name + ".adoption-link")
                            if not partial.exists():
                                os.link(source, partial, follow_symlinks=False)
                            hardlink_identity(source, partial)
                            rename_new(partial, target)
                        link = hardlink_identity(source, target)
                        source_sha = expected["sha256"]  # Small-file hashes remain in the native inventory.
                    else:
                        source_sha = hash_progress(source, folder, "hash_source", completed, total)
                    if not hardlink and not target.exists():
                        partial = target.with_name("." + target.name + ".adoption-partial")
                        need(not partial.is_symlink(), "Unsafe partial copy")
                        copy_with_progress(source, partial, folder, completed, total)
                        need(metadata(partial) == expected and hash_progress(partial, folder, "hash_copy", completed, total) == source_sha,
                             "Copied file differs from qualification source")
                        with partial.open("rb") as stream:
                            os.fsync(stream.fileno())
                        rename_new(partial, target)
                    elif not hardlink:
                        need(metadata(target) == expected and hash_progress(target, folder, "verify_existing", completed, total) == source_sha,
                             "Existing staging file changed; refusing to overwrite it")
                    need(metadata(source) == expected, "Source changed during copy")
                    event = dict(source=str(source), destination=str(target), **expected, full_sha256=source_sha)
                    if hardlink:
                        event["hardlink"] = link
                    need(key not in ledger or ledger[key] == event, "Completed copy evidence changed")
                    ledger[key] = event
                    write(ledger_path, ledger)
                    completed += expected["bytes"]
                    progress(folder, phase="file_verified", file=key, completed_files=len(ledger),
                             completed_bytes=completed, total_bytes=total, copy_mode=plan.get("copy_mode", "copy"))
                no_links(stage / "validation/step_0.json", folder)
                no_links(stage / "qualification_adoption.json", folder)
                publish_new(stage / "validation/step_0.json", item["baseline"]["canonical"])
                publish_new(stage / "qualification_adoption.json", dict(
                    schema=SCHEMA, operation=plan["operation"], plan_identity=plan["plan_identity"],
                    source=item["source"], source_checkpoint_identity=digest(item["checkpoint_marker"]),
                    baseline_source=item["baseline"]["source"], archive=item["archive"], loaded_step=8))
                validate_staging(m, plan, arm, stage, ledger, full_hash=False)
            validate_sources(m, plan)
        publish_new(folder / "prepared.json", dict(status="PREPARED", plan_identity=plan["plan_identity"],
                                                  copied_files_sha256=digest(ledger), total_bytes=total))
        progress(folder, phase="PREPARED", completed_files=len(ledger), completed_bytes=completed,
                 total_bytes=total, copy_mode=plan.get("copy_mode", "copy"))
        return read(folder / "prepared.json")


def validate_staging(m, plan, arm, stage, ledger, full_hash=True):
    item, stage = plan["arms"][arm], Path(stage)
    cp = stage / "checkpoints/global_step_8"
    need(tree(cp) == item["source_files"], "Staged checkpoint inventory changed")
    need(checkpoints.latest(cp.parent, m["identity"], arm, item["sampler_identity"]) == cp,
         "Staged native checkpoint is not resumable")
    need(sorted(p.name for p in (stage / "checkpoints").iterdir()) == ["global_step_8"], "Unexpected staged checkpoint")
    need(sorted(p.name for p in (stage / "validation").iterdir()) == ["step_0.json"]
         and read(stage / "validation/step_0.json") == item["baseline"]["canonical"], "Unexpected formal selection receipt")
    tag = read(stage / "qualification_adoption.json")
    need(tag["plan_identity"] == plan["plan_identity"] and tag["source"] == item["source"], "Staged adoption identity changed")
    for name in item["source_files"]:
        proof = ledger[arm + "/" + name]
        need(proof["source"] == str(Path(item["source"]) / name)
             and proof["destination"] == str(Path(item["staging"]) / "checkpoints/global_step_8" / name),
             "Copy ledger path changed")
        if plan.get("copy_mode", "copy") == "hardlink":
            need(hardlink_identity(proof["source"], cp / name) == proof["hardlink"],
                 "Qualified source inode changed after preparation")
        elif full_hash:
            need(hash_progress(cp / name, plan["folder"], "verify_activation", 0, 0) == proof["full_sha256"],
                 "Staged file hash changed before activation")


def rename_new(source, destination):
    source, destination = Path(source), Path(destination)
    need(not destination.exists() and not destination.is_symlink(), "Refusing to replace destination: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    for directory in {source.parent, destination.parent}:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def guard_controllers(m, plan, pbs, stack, activated=False):
    base, control = Path(m["manifest_path"]).parent, control_root(m)
    stack.enter_context(locked(base / "execution_launch.lock"))
    roots = controller_roots(m)
    need({str(p) for p in roots} == set(plan["controller_states"]), "Controller set changed; review a new plan")
    for root in roots:
        no_links(root / "launch.lock", base)
        stack.enter_context(locked(root / "launch.lock"))
    for root in roots:
        no_links(root / "states.lock", base)
        no_links(root / "guardian.lock", base)
        stack.enter_context(locked(root / "states.lock"))
        stack.enter_context(locked(root / "guardian.lock"))
    observations = []
    seen = set()
    for root in roots:
        states = states_at(root)
        original = plan["controller_states"][str(root)]
        need((root / "STOP").is_file() or not states, "Controller is not stopped: " + str(root))
        expected = deepcopy(original)
        if activated and root == control:
            for arm in ARMS:
                expected["train:" + arm] = fresh_state(arm, plan)
        need(states == expected, "Controller states changed after planning: " + str(root))
        need(not any(s.get("pending") for s in states.values()), "Resolve pending submissions before adoption")
        need(not any(s.get("phase") == "test" for s in states.values()), "Final testing already registered")
        for state in original.values():
            for key in ("job", "previous_job"):
                job = state.get(key)
                if not job or job["job_id"] in seen:
                    continue
                seen.add(job["job_id"])
                row = pbs.observe(job)
                need(finished(row), "PBS allocation is not terminal: " + job["job_id"])
                observations.append(dict(job=job, pbs=row, observed_utc=now()))
    return observations


def fresh_state(arm, plan):
    return dict(phase="train", arm=arm, status="READY", job=None, pending=None,
                qualification_adoption=plan["operation"], adopted_checkpoint_step=8)


def activate(m, folder, pbs=None):
    folder = Path(folder)
    no_links(folder / "operation.lock", Path(m["manifest_path"]).parent)
    with locked(folder / "operation.lock"):
        plan = checked_plan(m, folder)
        prepared, ledger = read(folder / "prepared.json"), read(folder / "copied_files.json")
        need(prepared["status"] == "PREPARED" and prepared["plan_identity"] == plan["plan_identity"]
             and prepared["copied_files_sha256"] == digest(ledger), "Preparation/copy evidence changed")
        journal_path = folder / "activation.json"
        journal = read(journal_path) if journal_path.exists() else None
        control = control_root(m)
        new_states = deepcopy(plan["controller_states"][str(control)])
        for arm in ARMS:
            new_states["train:" + arm] = fresh_state(arm, plan)
        states_already_written = states_at(control) == new_states
        with ExitStack() as stack:
            observations = guard_controllers(m, plan, pbs or AuditedPBS(), stack, activated=states_already_written)
            for arm in ARMS:
                item = plan["arms"][arm]
                for name in ("destination", "archive", "staging"):
                    path = Path(item[name])
                    if path.exists():
                        need(not path.is_symlink(), "Unsafe formal/staging directory")
                        no_links(path / "writer.lock", Path(m["manifest_path"]).parent)
                        stack.enter_context(locked(path / "writer.lock"))
                stack.enter_context(locked(control / "qualification" / arm / "writer.lock"))
            validate_sources(m, plan)
            # Validate both entire staged states before moving either old run.
            for arm in ARMS:
                item = plan["arms"][arm]
                dest, archive, stage = (Path(item[k]) for k in ("destination", "archive", "staging"))
                need(not (dest / "receipt.json").exists() and not (archive / "receipt.json").exists(),
                     "A formal completion receipt prevents adoption")
                if archive.exists():
                    need(journal is not None and tree(archive) == item["old_files"], "Archived formal history changed")
                    need(stage.exists() != dest.exists(), "Ambiguous partially activated run")
                else:
                    need(dest.exists() and stage.exists() and tree(dest) == item["old_files"], "Old formal history changed")
                validate_staging(m, plan, arm, stage if stage.exists() else dest, ledger)
            if journal is None:
                journal = dict(schema=SCHEMA, status="ACTIVATING", plan_identity=plan["plan_identity"],
                               started_utc=now(), old_states=plan["controller_states"], git=plan["git"],
                               pbs=observations, arms={arm: {key: plan["arms"][arm][key]
                                                           for key in ("source", "destination", "staging", "archive")}
                                                       for arm in ARMS})
                write(journal_path, journal)
            else:
                need(journal["plan_identity"] == plan["plan_identity"], "Activation journal belongs to another plan")
            for arm in ARMS:
                item = plan["arms"][arm]
                dest, archive, stage = (Path(item[k]) for k in ("destination", "archive", "staging"))
                if not archive.exists():
                    rename_new(dest, archive)
                if stage.exists():
                    rename_new(stage, dest)
                select_best(dest, m["identity"], arm)
                evidence = inspect_resume(dest, m, arm, item["sampler_identity"])
                need(evidence["step"] == 8 and evidence["best_selection"]["step"] == 0,
                     "Adopted run has an unexpected resume/selection state")
                journal["arms"][arm]["resume"] = evidence
                write(journal_path, journal)
            if not states_already_written:
                write(control / "states.json", new_states)
            journal.update(status="ACTIVATED", completed_utc=journal.get("completed_utc", now()),
                           new_states=new_states, last_terminal_observation=observations)
            write(journal_path, journal)
            progress(folder, phase="ACTIVATED", next_update=9, controller_still_stopped=True)
            return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "activate"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execution-record", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--expected-mix-job", required=True)
    parser.add_argument("--expected-only-job", required=True)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    m = load_manifest(args.manifest, args.execution_record)
    folder = Path(m["manifest_path"]).parent / "qualification_adoptions" / safe_id(args.operation)
    expected = dict(zip(ARMS, (args.expected_mix_job, args.expected_only_job)))
    git = dict(revision=command(["git", "-C", str(REPO), "rev-parse", "HEAD"], timeout=20),
               script_sha256=file_sha(__file__), script=str(Path(__file__).resolve()))
    if (folder / "plan.json").exists():
        plan = checked_plan(m, folder)
        need(plan["expected_jobs"] == expected and plan["git"]["script_sha256"] == git["script_sha256"],
             "Operation job IDs or operator code changed")
        need(plan.get("copy_mode", "copy") == args.copy_mode, "Existing plan has another copy mode")
    else:
        plan = make_plan(m, args.operation, expected, git, args.copy_mode)
    if args.action == "plan":
        logical_bytes = sum(f["bytes"] for arm in plan["arms"].values() for f in arm["source_files"].values())
        emit(dict(event="plan", plan=plan, logical_checkpoint_bytes=logical_bytes,
                  required_copy_bytes=0 if args.copy_mode == "hardlink" else logical_bytes))
    elif args.action == "prepare":
        emit(dict(event="prepared", result=prepare(m, plan)))
    else:
        need((folder / "prepared.json").exists(), "Run prepare before activate")
        emit(dict(event="activated", result=activate(m, folder)))


if __name__ == "__main__":
    main()
