"""Full checkpoints, immutable pins and explicit lineage authority.

Large payloads use size/mtime inventory, not a claim of full tensor equality.
Small metadata is SHA256 checked. Pins share blocks with the original files.
"""
import os
from pathlib import Path
import shutil
import uuid
from .common import below, digest, need, now, read, sha, write

MARKER = "COMPLETE_CONTINUATION.json"


def inventory(path):
    root = Path(path)
    result = {}
    for p in sorted(root.rglob("*")):
        need(not p.is_symlink(), "Checkpoint symlink is forbidden")
        if p.is_file() and p.name != MARKER:
            s = p.stat()
            need(s.st_size > 0, "Empty checkpoint file: " + str(p))
            result[str(p.relative_to(root))] = dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                sha256=sha(p) if s.st_size <= 1048576 else None)
    need(result, "Empty checkpoint")
    return result


def components(path, step, world=None):
    path = Path(path)
    need((path / "data.pt").is_file(), "Missing dataloader checkpoint")
    spec = read(path / "actor/ckpt_contents.json")
    need(spec.get("schema_version") == 2 and spec.get("global_step") == step,
         "Checkpoint schema/step mismatch")
    need(spec.get("world_size") in (8, 16) and (world is None or spec["world_size"] == world),
         "Checkpoint world mismatch")
    for name in ("model", "optimizer", "lr_scheduler", "rng_state"):
        item = spec["contents"][name]
        rel = Path(item["path"])
        need(not rel.is_absolute() and ".." not in rel.parts, "Unsafe component path")
        folder = below(path / "actor" / rel, path)
        payload = list(folder.rglob("*.distcp"))
        if name == "rng_state":
            payload += list(folder.rglob("*.pt"))
        need(item.get("format") == "megatron_dist_checkpoint" and payload
             and all(p.stat().st_size > 0 for p in payload), "Missing full state: " + name)
    return spec


def validate(record):
    path = Path(record["path"])
    components(path, record["step"], record["world_size"])
    need(inventory(path) == record["files"], "Checkpoint inventory changed: " + str(path))
    if record.get("lineage"):
        marker = read(path / MARKER)
        need(marker["lineage"] == record["lineage"] and marker["step"] == record["step"]
             and marker["files"] == record["files"], "Checkpoint lineage differs")
    return record


def latest_legacy(run, kind):
    base = Path(run) / "checkpoints"
    tracker = base / "latest_checkpointed_iteration.txt"
    value = tracker.read_text().strip()
    need(value.isdigit() and 0 < int(value) <= 468, "No completed legacy checkpoint")
    step = int(value)
    p = below(base / f"global_step_{step}", base)
    components(p, step, 8)
    if kind == "domain":
        need((p / "sampling.json").is_file(), "Domain checkpoint lacks sampling plan")
    record = dict(path=str(p), step=step, world_size=8, files=inventory(p), lineage=None)
    need(tracker.read_text().strip() == value, "Tracker advanced during checkpoint capture; retry")
    return record


def latest(run, lineage):
    result = []
    for marker in (Path(run) / "checkpoints").glob("global_step_*/" + MARKER):
        row = read(marker)
        need(row.get("lineage") == lineage, "Mixed checkpoint lineages in one run")
        record = dict(row, path=str(marker.parent))
        validate(record)
        result.append(record)
    return max(result, key=lambda r: r["step"]) if result else None


def pin(record, destination):
    destination = Path(destination)
    manifest = destination / "source.json"
    if manifest.exists():
        existing = validate(read(manifest))
        need(all(existing.get(k) == record.get(k) for k in ("step", "world_size", "lineage", "files")),
             "Pin already belongs to another checkpoint")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    intent = destination / "pin_intent.json"
    target = destination / ("global_step_" + str(record["step"]))
    if intent.exists():
        need(read(intent) == record, "Pin intent belongs to another checkpoint")
        if target.exists():
            # Recover a crash between the atomic directory rename and source.json.
            # The original writer may already have pruned its copy of this save.
            pinned = dict(record, path=str(target), origin=record["path"], pinned_utc=now())
            validate(pinned)
            write(manifest, pinned)
            return pinned
    else:
        need(not target.exists(), "Unjournaled pin exists")
        write(intent, record)
    validate(record)
    temp = destination / (".linking_" + uuid.uuid4().hex)
    temp.mkdir()
    try:
        names = list(record["files"])
        if record.get("lineage"):
            names.append(MARKER)
        for name in names:
            source, dest = Path(record["path"]) / name, temp / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, dest, follow_symlinks=False)
        need(inventory(temp) == record["files"], "Checkpoint changed while pinning")
        need(not target.exists(), "Unjournaled pin exists")
        temp.rename(target)
        pinned = dict(record, path=str(target), origin=record["path"], pinned_utc=now())
        write(manifest, pinned)
        return validate(pinned)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def commit(staged, run, step, world, lineage, kind, parent, phase):
    staged, base = Path(staged), Path(run) / "checkpoints"
    components(staged, step, world)
    for rank in range(world):
        state = read(staged / "actor" / f"continuation_state_rank_{rank}.json")
        need(state.get("optimizer_audit_policy") == "fused-adam-empty-local-group-step-v1",
             "Checkpoint has unbound optimizer audit")
    if kind == "domain":
        need(read(staged / "sampling.json") == read(Path(parent["path"]) / "sampling.json"),
             "Saved domain sampling identity changed")
    record = dict(step=step, world_size=world, lineage=lineage, kind=kind, phase=phase,
                  parent=dict(step=parent["step"], files_digest=digest(parent["files"])),
                  files=inventory(staged), at_utc=now())
    write(staged / MARKER, record)
    target = base / f"global_step_{step}"
    need(not target.exists(), "Refusing to overwrite a completed checkpoint")
    base.mkdir(parents=True, exist_ok=True)
    staged.rename(target)
    write(base / "latest.json", dict(record, path=str(target)))
    from .common import atomic_text
    atomic_text(base / "latest_checkpointed_iteration.txt", str(step) + "\n")
    return dict(record, path=str(target))


def retain(run, lineage, keep=2):
    # External readers must hard-link their source before this writer advances.
    base = Path(run) / "checkpoints"
    markers = sorted(base.glob("global_step_*/" + MARKER),
                     key=lambda p: int(p.parent.name.removeprefix("global_step_")), reverse=True)
    for marker in markers[keep:]:
        need(read(marker)["lineage"] == lineage, "Cannot prune a foreign lineage")
        shutil.rmtree(below(marker.parent, base))


def remove_pin(destination):
    destination = Path(destination)
    p = destination / "source.json"
    if p.exists():
        r = read(p)
        target = below(r["path"], destination)
        if target.exists():
            validate(r)
            shutil.rmtree(target)
        write(destination / "retired.json", dict(at_utc=now(), source=r))
