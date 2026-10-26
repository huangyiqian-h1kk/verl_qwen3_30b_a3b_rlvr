#!/usr/bin/env python3
"""Plan/apply the approved Oct 27 D4 checkpoint cleanup. No job submission.

The reviewed report and eight protected object identities are fixed. Plan only
writes a new, hashed manifest. Apply requires that manifest's SHA256, fresh PBS
and lease checks, and valid protected archives. Deletes only enumerated tensor
payloads; preserves logs/provenance. Interrupted apply uses the SAME plan.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from d4_training_continuation import checkpoints as ck
from d4_training_continuation.best_checkpoints import model_files, validate_files
from d4_training_continuation.common import digest, need, now, read, sha, write
_spec = importlib.util.spec_from_file_location("finish_training", Path(__file__).with_name("finish_training.py"))
f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f)

VERSION = "2026-10-27.cleanup.1"
REPORT_HASHES = {
    "stop.json": "09facd6ce8b3dcf665c5655eee6e22605eea4a07b72dfd7a01f2e60418b2b371",
    "storage.json": "708065656afed0adf5a71877956011ab0a427a8629c3bb99fc8d3d3aafe13856",
    "checkpoint_retention_review.json": "fe9019fddc7eed1f89f8d92fc8ea0690bbd396173f6a989c8c957e170b5ed936",
}
# kind: key -> (step, final mode). Never select by step number alone.
KEEP = {
    "random": {
        "48109d3e7c1fa7ee607ca8340a229f645a1f528e792520f044b9366896a3735c": (150, "full"),
        "f96a49e4ded13d81670a5b06a572043712e846fb46d9f9e02203b7829deef2b9": (200, "full"),
        "f030bce4a1ecfcfb1f60bdb26a376928243b4547b357069fcc4b5a5587b16893": (125, "weights"),
        "c2828db7acf881f8bd0def073d37bafcf84e784b956a17885e0df9422b2af6db": (50, "unverified_weights"),
    },
    "domain": {
        "6bb22cbac40943dd8e1fbd00e5293bc4e42d3cf9ae73bcdc9df913f7f0b47e90": (175, "full"),
        "c23c2ffdb80828eb70594085e2205cc11fb158bd80f4e230c10185aa43e79db9": (225, "full"),
        "b36dbdb29006469beba4488359b4f0b0ab83d4fd196d6475a4bb4a5df8597c61": (100, "weights"),
        "a23c7208c35adf5ec1c2fe5efed5cd7c78121a8858619e1638b155487a3b499d": (200, "weights"),
    },
}
RETIRE = {
    "random": {"613d454d8699712968e4e92a3b89a83bf77f7efd87eeb9b9ff899ef0309edff4": 195},
    "domain": {"fa823947438d382a437d12c69a3cde043573a3e2ec0c7e5486ccfa9d2e163fae": 220},
}
MODULES = ("scripts/multidomain/cleanup_finished_training.py", "scripts/multidomain/finish_training.py",
           "d4_training_continuation/__init__.py", "d4_training_continuation/common.py",
           "d4_training_continuation/checkpoints.py", "d4_training_continuation/best_checkpoints.py",
           "d4_training_continuation/evaluation.py")
MARKERS = {"COMPLETE_CONTINUATION.json", "COMPLETE_0390.json", "COMPLETE.json", "COMPLETE_DB_DUAL.json"}


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def fingerprint(path):
    s = path.lstat()
    need(stat.S_ISREG(s.st_mode), "Not a regular file: " + str(path))
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                allocated=s.st_blocks * 512, nlink=s.st_nlink, mode=stat.S_IMODE(s.st_mode))


def same_file(path, expected, nlink=False):
    actual = fingerprint(path)
    fields = ("device", "inode", "bytes", "mtime_ns", "allocated", "mode") + (("nlink",) if nlink else ())
    need(all(actual[k] == expected[k] for k in fields), "File changed since plan: " + str(path))
    return actual


def rel(path, root):
    return str(f.safe(path, root).relative_to(root))


def target(root, relative):
    need(not Path(relative).is_absolute(), "Plan path must be relative")
    return f.safe(root / relative, root)


def operation_dir(root, operation):
    need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", operation), "Unsafe operation ID")
    return f.safe(root / "outputs/analysis" / ("d4_cleanup_" + operation), root)


def fresh_pbs(root):
    need(not f.owned_jobs(root, f.pbs_jobs()), "D4 PBS jobs still exist; cleanup refused")


@contextmanager
def fenced(root):
    """No signals/qdel: all owners/writers must already have stopped."""
    base = root / "outputs/multidomain"
    with ExitStack() as stack:
        locked = set()
        def lock(path):
            path = f.safe(path, root)
            if path not in locked:
                stack.enter_context(f.lease(path))
                locked.add(path)
        lock(base / "continuation_operations/apply.lock")
        service = base / "checkpoint_archive_service"
        need((service / "stop.request").is_file(), "Archive watcher not stopped")
        for name in ("watch.lock", "install.lock"):
            lock(service / name)
        for kind in f.KINDS:
            p = f.paths(root, kind)
            for directory, name in ((p["control"], "controller.lock"), (p["handoff"], "controller.lock"),
                                    (p["monitor"], "monitor.lock"), (p["single"], "submission.lock"),
                                    (p["old"], "submission.lock"), (p["run"] / "checkpoint_archive", "archive.lock")):
                lock(directory / name)
            state = read(p["control"] / "state.json")
            retired = read(p["control"] / "retired.json")
            need(state.get("phase") == "STOPPED_BY_USER" and state.get("controller_status") == "STOPPED"
                 and retired.get("version") == f.VERSION, "Setup is not persistently retired: " + kind)
            for key in ("monitor", "handoff"):
                need((p[key] / "stop.request").is_file(), "Legacy owner stop marker missing")
            plan = read(f.safe(p["control"] / "operations" / state["operation"] / "plan.json", root))
            need(plan.get("retirement_guard") == f.VERSION, "Frozen runtime retirement guard missing")
            template = f.safe(Path(plan["template"]), root)
            for name, checksum in plan["template_files"].items():
                need(sha(f.safe(template / name, template)) == checksum, "Frozen runtime changed")
            text = (template / "runtime/d4_training_continuation/cli.py").read_text()
            for function in ("spawn", "daemon"):
                expected = (f"def {function}(root, kind):\n"
                            "    # Persistent user-requested retirement; no automatic restart.\n"
                            f"    if Path({str(p['control'] / 'retired.json')!r}).exists():\n        return\n")
                need(expected in text, "Frozen retirement function changed: " + function)
        # Include historical check/qualification/backfill writers, not just active lineage.
        for directory, _ in f.scan_roots(root):
            for parent, dirs, files in os.walk(directory, followlinks=False):
                dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
                for name in files:
                    if name in ("stage.lock", "submission.lock", "controller.lock", "monitor.lock"):
                        lock(Path(parent) / name)
        fresh_pbs(root)
        yield


def scan(root):
    nodes, files, total = {}, {}, 0
    for directory, group in f.scan_roots(root):
        def fail(error):
            raise error
        for parent, dirs, names in os.walk(directory, followlinks=False, onerror=fail):
            for path in [Path(parent)] + [Path(parent) / x for x in dirs + names]:
                need(not path.is_symlink(), "Symlink in cleanup scope: " + str(path))
            for path in [Path(parent)] + [Path(parent) / n for n in names]:
                s = path.lstat()
                key = (s.st_dev, s.st_ino)
                if key not in nodes:
                    nodes[key] = dict(allocated=s.st_blocks * 512, paths=[], regular=stat.S_ISREG(s.st_mode), nlink=s.st_nlink)
                    total += s.st_blocks * 512
                if stat.S_ISREG(s.st_mode):
                    relative = rel(path, root)
                    spec = fingerprint(path)
                    files[relative] = spec
                    nodes[key]["paths"].append(relative)
    return files, nodes, total


def report(root, directory):
    values = {}
    for name, expected in REPORT_HASHES.items():
        path = f.safe(directory / name, root)
        need(sha(path) == expected, "This is not the reviewed Oct 27 report: " + name)
        values[name] = read(path)
    stop, storage = values["stop.json"], values["storage.json"]
    need(stop["root"] == str(root) == storage["root"] and stop["status"] == "STOPPED"
         and stop["owners_fenced"] and not stop["remaining_jobs"], "Unverified stop report")
    need(storage["inventory_status"] == "COMPLETE" and not storage["errors"], "Incomplete inventory")
    return storage, values["checkpoint_retention_review.json"]


def inspect_archives(root):
    objects, protected, indexes = {}, {}, {}
    for kind in f.KINDS:
        archive = f.paths(root, kind)["run"] / "checkpoint_archive"
        indexes[kind] = read(archive / "index.json")
        found = {}
        for file in sorted((archive / "objects").glob("*/manifest.json")):
            obj = read(file)
            key = file.parent.name
            need(key == obj["key"] and obj["kind"] == kind, "Archive identity differs")
            need(key == digest({k: obj["original"][k] for k in ("lineage", "step", "world_size", "files")}),
                 "Archive key does not bind its original inventory")
            need(key in KEEP[kind] or key in RETIRE[kind], "New/unreviewed archive object: " + key)
            expected_step = KEEP[kind].get(key, (RETIRE[kind].get(key),))[0]
            need(obj["step"] == expected_step, "Archive step differs")
            payload = f.safe(Path(obj["payload"]), file.parent)
            need(payload == file.parent / "payload" / ("global_step_" + str(obj["step"])), "Archive path differs")
            validate_files(payload, obj["inventory"])
            need(model_files(dict(obj["original"], path=str(payload), files=obj["model_inventory"])) == obj["model_inventory"],
                 "Model inventory differs")
            if obj["mode"] == "full":
                ck.validate(dict(obj["original"], path=str(payload)))
            if key in KEEP[kind]:
                mode = KEEP[kind][key][1]
                need((mode == "weights" and obj["mode"] in ("full", "weights")) or mode == obj["mode"], "Unexpected archive mode")
                protected[str(payload)] = dict(mode="model" if mode == "weights" else "review" if mode == "unverified_weights" else "full",
                                              model_files=sorted(obj["model_inventory"]))
            else:
                need(not obj.get("holds"), "Unexpected hold on retired archive")
            found[key] = dict(obj, manifest=rel(file, root))
        need(set(found) == set(KEEP[kind]) | set(RETIRE[kind]), "Reviewed archive object missing")
        for cohort in indexes[kind]["cohorts"].values():
            for metric, winner in cohort.items():
                need(winner["key"] in KEEP[kind], "An unretained object is a validation winner")
                if metric in ("overall", "overall_full"):
                    need(KEEP[kind][winner["key"]][1] == "full", "Overall best would lose full state")
        objects[kind] = found
    selected, summaries, errors = f.retention(root)
    need(not errors, "Retention verification failed: " + str(errors))
    need(set(selected) == set(protected) and all(selected[p]["mode"] == v["mode"] for p, v in protected.items()),
         "Current best/final selection differs from approved policy")
    return objects, protected, indexes


def make_edit(root, path, value):
    path = f.safe(path, root)
    return dict(type="json", path=rel(path, root), before=sha(path) if path.exists() else None, after=value)


def move_edit(root, source, destination):
    need(source.is_file() and not destination.exists(), "Cannot retire marker: " + str(source))
    return dict(type="rename", path=rel(source, root), destination=rel(destination, root), before=sha(source))


def metadata_edits(root, operation, objects, indexes, affected):
    edits, checkpoint_modes = [], {}
    for kind in f.KINDS:
        archive = f.paths(root, kind)["run"] / "checkpoint_archive"
        index = json.loads(json.dumps(indexes[kind]))
        for cohort in index["cohorts"].values():
            for winner in cohort.values():
                winner["mode"] = KEEP[kind][winner["key"]][1]
        index.update(full_keep=sorted(k for k, (_, m) in KEEP[kind].items() if m != "weights"),
                     weights_keep=sorted({w["key"] for c in index["cohorts"].values() for w in c.values()}),
                     live_lineages=[], retirement_cleanup=dict(version=VERSION, operation=operation, at_utc=now()))
        edits.append(make_edit(root, archive / "index.json", index))
        final = dict(version=VERSION, operation=operation, training_retired=True, objects=[])
        for key, obj in objects[kind].items():
            directory = archive / "objects" / key
            payload = Path(obj["payload"])
            if key not in KEEP[kind]:
                edits.append(move_edit(root, directory / "manifest.json", directory / "retired.json"))
                checkpoint_modes[str(payload)] = "retired"
                continue
            step, mode = KEEP[kind][key]
            final["objects"].append(dict(key=key, step=step, mode=mode, path=str(payload)))
            if mode == "weights" and obj["mode"] == "full":
                expected_hold = "historical-rescue-step-" + str(step)
                need(set(obj.get("holds", [])) <= {expected_hold}, "Unreviewed archive hold")
                after = {k: v for k, v in obj.items() if k != "manifest"}
                after.update(mode="weights", inventory=obj["model_inventory"], holds=[],
                             released_holds=obj.get("holds", []), retirement_cleanup=dict(version=VERSION, operation=operation))
                edits.append(make_edit(root, directory / "manifest.json", after))
                checkpoint_modes[str(payload)] = "weights"
        edits.append(make_edit(root, f.paths(root, kind)["control"] / "final_checkpoints.json", final))
    trackers = set()
    for cp, count in sorted(affected.items()):
        cp = Path(cp)
        for name in MARKERS:
            marker = cp / name
            if marker.exists():
                edits.append(move_edit(root, marker, cp / (name + ".before_cleanup")))
        for name in ("latest.json", "latest_checkpointed_iteration.txt"):
            tracker = cp.parent / name
            if tracker.exists():
                # Conservatively retire the pointer only if it points at this checkpoint.
                if name.endswith(".txt"):
                    points = tracker.read_text().strip() == cp.name.removeprefix("global_step_")
                else:
                    value = read(tracker)
                    points = value.get("path") == str(cp) or value.get("step") == int(cp.name.removeprefix("global_step_"))
                if points and tracker not in trackers:
                    edits.append(move_edit(root, tracker, tracker.with_name(tracker.name + ".before_cleanup")))
                    trackers.add(tracker)
        edits.append(make_edit(root, cp / "PAYLOAD_CLEANUP.json", dict(version=VERSION, operation=operation,
                    checkpoint=str(cp), final_mode=checkpoint_modes.get(str(cp), "retired"),
                    planned_unlinks=count, evidence="Original logs/configs/receipts retained; payload removal is journaled separately.")))
    paths = [e["path"] for e in edits]
    need(len(paths) == len(set(paths)), "Duplicate metadata operation")
    return edits


def make_plan(root, review, operation):
    output = operation_dir(root, operation)
    if (output / "plan.json").exists():
        plan = load_plan(root, operation)
        print_summary(plan, output)
        return plan
    need(not output.exists(), "Incomplete plan directory; use a NEW operation ID")
    with fenced(root):
        storage, reviewed = report(root, review)
        objects, protected, indexes = inspect_archives(root)
        files, nodes, total = scan(root)
        seen = {}
        for r in reviewed:
            cp = f.safe(Path(r["path"]), root)
            need(f.CP.fullmatch(cp.name), "Bad checkpoint in reviewed report")
            seen[str(cp)] = r
        observed = {str(f.checkpoint_parent(root / path)) for path in files if f.checkpoint_parent(root / path)}
        need(observed == set(seen), "Checkpoint paths changed since reviewed report")
        deletions, affected = {}, defaultdict(int)
        for path, spec in sorted(files.items()):
            file = root / path
            cp = f.checkpoint_parent(file)
            if cp is not None and f.removable_payload(file, cp, protected):
                deletions[path] = spec
                affected[str(cp)] += 1
        reclaim, external = 0, []
        for key, node in nodes.items():
            deleting = [p for p in node["paths"] if p in deletions]
            if not deleting:
                continue
            if len(node["paths"]) != node["nlink"]:
                external.append(deleting[0])
            if len(deleting) == node["nlink"]:
                reclaim += node["allocated"]
        need(not external, "New external hardlinks; re-review needed: " + str(external[:5]))
        need(reclaim == storage["totals"]["proposed_reclaimable_allocated_bytes"], "Reclaim set changed since reviewed report")
        # Bind every checkpoint to its original report totals before recording a file manifest.
        for cp, old in seen.items():
            group = [v for p, v in files.items() if str(f.checkpoint_parent(root / p)) == cp]
            need(len(group) == old["files"] and sum(v["bytes"] for v in group) == old["logical_bytes"],
                 "Checkpoint contents changed since reviewed report: " + cp)
        edits = metadata_edits(root, operation, objects, indexes, affected)
        plan = dict(version=VERSION, operation=operation, root=str(root), created_utc=now(),
                    code={p: sha(CODE / p) for p in MODULES}, report_hashes=REPORT_HASHES,
                    review_dir=str(review), protected=protected, objects=objects,
                    files=deletions, metadata=edits, affected_checkpoints=sorted(affected),
                    checkpoint_paths=sorted(seen),
                    before_unique_allocated_bytes=total, reclaimable_allocated_bytes=reclaim,
                    checkpoint_count=len(reviewed), expected_after_allocated_bytes=total-reclaim)
        plan["sha256"] = digest(plan)
        output.mkdir(parents=True)
        for i, edit in enumerate(edits):
            if edit["before"] is not None:
                source = target(root, edit["path"])
                backup = output / "metadata_before" / str(i)
                backup.parent.mkdir(exist_ok=True)
                backup.write_bytes(source.read_bytes())
                need(sha(backup) == edit["before"], "Backup checksum differs")
        write(output / "plan.json", plan)
        (output / "plan.sha256").write_text(plan["sha256"] + "\n")
        write(output / "state.json", dict(status="PLANNED", plan_sha256=plan["sha256"], payload_links_removed=0))
    print_summary(plan, output)
    return plan


def load_plan(root, operation):
    p = read(operation_dir(root, operation) / "plan.json")
    need(p.get("version") == VERSION and p.get("root") == str(root) and p.get("operation") == operation, "Wrong cleanup plan")
    need(p["sha256"] == digest({k: v for k, v in p.items() if k != "sha256"}), "Plan SHA256 changed")
    need(p["code"] == {name: sha(CODE / name) for name in MODULES}, "Use the SAME pinned cleanup code version")
    need(p["report_hashes"] == REPORT_HASHES, "Plan is not bound to approved report")
    _, reviewed = report(root, Path(p["review_dir"]))
    need(set(p["checkpoint_paths"]) == {r["path"] for r in reviewed}, "Plan checkpoint scope differs")
    expected_protected = set()
    for kind in f.KINDS:
        need(set(p["objects"][kind]) == set(KEEP[kind]) | set(RETIRE[kind]), "Plan archive membership differs")
        for key, obj in p["objects"][kind].items():
            expected = f.paths(root, kind)["run"] / "checkpoint_archive/objects" / key / "payload" / ("global_step_" + str(obj["step"]))
            need(obj["payload"] == str(expected) and obj["key"] == key and obj["kind"] == kind,
                 "Plan archive path/identity differs")
            need(key == digest({k: obj["original"][k] for k in ("lineage", "step", "world_size", "files")}),
                 "Plan archive inventory binding differs")
            if key in KEEP[kind]:
                expected_protected.add(str(expected))
                mode = KEEP[kind][key][1]
                need(p["protected"][str(expected)] == dict(mode="model" if mode == "weights" else "review" if mode == "unverified_weights" else "full",
                     model_files=sorted(obj["model_inventory"])), "Plan protection changed")
    need(set(p["protected"]) == expected_protected, "Plan protected paths differ")
    for relative in p["files"]:
        path = target(root, relative)
        cp = f.checkpoint_parent(path)
        need(str(cp) in p["checkpoint_paths"] and f.removable_payload(path, cp, p["protected"]),
             "Unapproved payload deletion path: " + relative)
    return p


def print_summary(plan, output):
    result = dict(status=read(output / "state.json")["status"], plan_sha256=plan["sha256"], plan=str(output / "plan.json"),
                  checkpoint_paths=plan["checkpoint_count"], affected_checkpoint_paths=len(plan["affected_checkpoints"]),
                  payload_links_to_remove=len(plan["files"]),
                  before_TiB=plan["before_unique_allocated_bytes"] / 2**40,
                  reclaimable_TiB=plan["reclaimable_allocated_bytes"] / 2**40,
                  projected_after_TiB=plan["expected_after_allocated_bytes"] / 2**40,
                  metadata_operations=len(plan["metadata"]), payloads_deleted_by_this_plan_command=0)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("PLAN_SHA256=" + plan["sha256"], flush=True)


def metadata_state(root, edit):
    path = target(root, edit["path"])
    if edit["type"] == "rename":
        dest = target(root, edit["destination"])
        if dest.exists():
            need(not path.exists() and sha(dest) == edit["before"], "Retired metadata changed")
            return "after"
        need(path.exists() and sha(path) == edit["before"], "Metadata source changed")
        return "before"
    if path.exists():
        if read(path) == edit["after"]:
            return "after"
        need(edit["before"] is not None and sha(path) == edit["before"], "Metadata changed: " + str(path))
    else:
        need(edit["before"] is None, "Metadata disappeared: " + str(path))
    return "before"


def sync_parent(path):
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def apply_metadata(root, edits):
    for edit in edits:
        if metadata_state(root, edit) == "after":
            continue
        path = target(root, edit["path"])
        if edit["type"] == "rename":
            os.rename(path, target(root, edit["destination"]))
            sync_parent(path)
        else:
            write(path, edit["after"])


def verify_protected(root, plan):
    for kind, entries in plan["objects"].items():
        for key, obj in entries.items():
            if key not in KEEP[kind]:
                continue
            payload = f.safe(Path(obj["payload"]), root)
            mode = KEEP[kind][key][1]
            validate_files(payload, obj["inventory"] if mode == "full" else obj["model_inventory"])
            if mode == "full":
                ck.validate(dict(obj["original"], path=str(payload)))
            else:
                need(model_files(dict(obj["original"], path=str(payload), files=obj["model_inventory"])) == obj["model_inventory"],
                     "Protected model inventory differs")


def payload_preflight(root, plan, started):
    scanned, _, _ = scan(root)
    observed = {str(f.checkpoint_parent(root / p)) for p in scanned if f.checkpoint_parent(root / p)}
    need(observed == set(plan["checkpoint_paths"]), "Checkpoint paths changed since preview")
    remaining, missing = {}, defaultdict(int)
    for relative, spec in plan["files"].items():
        path = target(root, relative)
        key = (spec["device"], spec["inode"])
        if path.exists():
            remaining[relative] = same_file(path, spec)
        else:
            need(started, "Payload disappeared before apply: " + str(path))
            missing[key] += 1
    # nlink changes caused by THIS plan are expected; outside changes are not.
    for relative, actual in remaining.items():
        original = plan["files"][relative]
        key = (original["device"], original["inode"])
        need(actual["nlink"] == original["nlink"] - missing[key], "Hardlink count changed outside this cleanup: " + relative)
    return remaining


def unlink_one(root, relative, spec):
    path = target(root, relative)
    same_file(path, spec)
    # Resolve each directory with O_NOFOLLOW and hold its fd through unlink.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = Path(relative).parts
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        s = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        need((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) ==
             (spec["device"], spec["inode"], spec["bytes"], spec["mtime_ns"]), "Payload changed immediately before unlink")
        os.unlink(parts[-1], dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)


def after_check(root, plan):
    verify_protected(root, plan)
    need(all(metadata_state(root, e) == "after" for e in plan["metadata"]), "Metadata transaction incomplete")
    need(not any(target(root, p).exists() for p in plan["files"]), "Planned payload links remain")
    fresh_pbs(root)
    for kind in f.KINDS:
        archive = f.paths(root, kind)["run"] / "checkpoint_archive"
        need({p.parent.name for p in (archive / "objects").glob("*/manifest.json")} == set(KEEP[kind]), "Final archive membership differs")
        for key, (_, mode) in KEEP[kind].items():
            obj = read(archive / "objects" / key / "manifest.json")
            need(obj["mode"] == mode, "Final archive mode differs")
    _, _, total = scan(root)
    return total


def bundle(output):
    path = output.with_suffix(".tar.gz")
    temporary = path.with_suffix(".tmp")
    # Files below output are small manifests/metadata, never model payloads.
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    os.replace(temporary, path)
    print("RESULT_BUNDLE=" + str(path), flush=True)


def apply(root, operation, confirmation):
    output = operation_dir(root, operation)
    plan = load_plan(root, operation)
    need(confirmation == plan["sha256"], "--confirm-plan must equal PLAN_SHA256 from preview")
    with f.lease(output / "apply.lock"):
        state = read(output / "state.json")
        try:
            with fenced(root):
                verify_protected(root, plan)
                for edit in plan["metadata"]:
                    metadata_state(root, edit)
                started = bool(state.get("started_utc"))
                remaining = payload_preflight(root, plan, started)
                if not started:
                    need(all(metadata_state(root, e) == "before" for e in plan["metadata"]), "Metadata changed before first apply")
                fresh_pbs(root)
                state.update(status="APPLYING", plan_sha256=plan["sha256"], started_utc=state.get("started_utc") or now())
                write(output / "state.json", state)
                apply_metadata(root, plan["metadata"])
                # No tensor unlink occurs before all archive/index/marker edits succeed.
                fresh_pbs(root)
                done = len(plan["files"]) - len(remaining)
                for relative in sorted(remaining):
                    unlink_one(root, relative, plan["files"][relative])
                    done += 1
                    if done % 100 == 0:
                        state.update(payload_links_removed=done, updated_utc=now())
                        write(output / "state.json", state)
                        print(f"[0390] Cleanup payload links {done}/{len(plan['files'])}", flush=True)
                total = after_check(root, plan)
                state.update(status="COMPLETE", finished_utc=now(), payload_links_removed=len(plan["files"]),
                             affected_checkpoint_paths=len(plan["affected_checkpoints"]),
                             planned_reclaimable_allocated_bytes=plan["reclaimable_allocated_bytes"],
                             before_unique_allocated_bytes=plan["before_unique_allocated_bytes"],
                             after_unique_allocated_bytes=total,
                             observed_scope_reduction_bytes=plan["before_unique_allocated_bytes"]-total,
                             after_TiB=total / 2**40,
                             observed_reduction_TiB=(plan["before_unique_allocated_bytes"]-total) / 2**40,
                             protected_archives_verified=True, remaining_scoped_jobs={}, error=None)
                write(output / "state.json", state)
        except BaseException as error:
            state.update(status="INCOMPLETE", error=type(error).__name__ + ": " + str(error), updated_utc=now())
            write(output / "state.json", state)
            raise
        finally:
            bundle(output)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "apply", "verify"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--operation", default="20261027-best-final")
    p.add_argument("--review-dir", type=Path)
    p.add_argument("--confirm-plan")
    a = p.parse_args(argv)
    root = a.root.resolve()
    need((root / "outputs/multidomain").is_dir(), "Wrong training root")
    if a.action == "plan":
        need(a.review_dir is not None, "--review-dir is required")
        make_plan(root, a.review_dir.absolute(), a.operation)
    elif a.action == "apply":
        need(a.confirm_plan, "--confirm-plan is required")
        apply(root, a.operation, a.confirm_plan)
    else:
        plan = load_plan(root, a.operation)
        with fenced(root):
            total = after_check(root, plan)
        print(json.dumps(dict(status="VERIFIED", unique_allocated_TiB=total / 2**40,
                              protected_archives_verified=True, remaining_scoped_jobs={}), indent=2))


if __name__ == "__main__":
    main()
