"""Commit marker after native model/optimizer/scheduler/RNG and DataLoader save."""
from pathlib import Path
import shutil
import uuid
from d4_teachers.common import digest, file_sha, need, now, read, write


def inventory(folder):
    result = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.name != "mopd_complete.json":
            stat = p.stat()
            result[str(p.relative_to(folder))] = {
                "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha(p) if stat.st_size <= 1048576 else None}
    return result


def validate_native(folder, step):
    folder = Path(folder)
    need((folder / "data.pt").stat().st_size > 0 and (folder / "driver.pt").stat().st_size > 0,
         "Missing DataLoader/driver RNG state")
    native = read(folder / "actor/ckpt_contents.json")
    need(native["framework"] == "megatron" and native["global_step"] == step and native["world_size"] == 8,
         "Native checkpoint step/topology mismatch")
    for key in ("model", "optimizer", "lr_scheduler", "rng_state"):
        entry = native["contents"][key]
        path = (folder / "actor" / entry["path"]).resolve()
        need((folder / "actor").resolve() in path.parents, "Checkpoint content escapes actor directory")
        candidates = list(path.rglob("*")) if path.is_dir() else [path]
        need(any(p.is_file() and p.stat().st_size > 0 for p in candidates), "Missing checkpoint content: " + key)


def commit(folder, identity, arm, step, sampler_identity, resumed_from=0):
    validate_native(folder, step)
    proof = dict(status="PASS", identity=identity, arm=arm, step=step, sampler_identity=sampler_identity,
                 resumed_from=resumed_from,
                 contents=inventory(folder))
    write(Path(folder) / "mopd_complete.json", proof)
    return proof


def latest(root, identity, arm, sampler_identity=None):
    committed = []
    for p in Path(root).glob("global_step_*/mopd_complete.json"):
        proof = read(p)
        need(proof["identity"] == identity and proof["arm"] == arm, "Checkpoint belongs to another experiment")
        need(p.parent.name == "global_step_" + str(proof["step"]), "Checkpoint directory/step mismatch")
        if sampler_identity is not None:
            need(proof["sampler_identity"] == sampler_identity, "Checkpoint sampler changed")
        committed.append((proof["step"], p.parent, proof))
    if not committed:
        return None
    _, folder, proof = max(committed, key=lambda t: t[0])
    validate_native(folder, proof["step"])
    need(inventory(folder) == proof["contents"], "Committed checkpoint changed; refusing silent rollback")
    return folder


def evaluation_reference(root, identity, arm, step, sampler_identity):
    """Bind a validation to the saved model state, or the unchanged base at step 0."""
    if step == 0:
        return dict(checkpoint=None, checkpoint_identity=digest([identity, "base"]))
    folder = Path(root).resolve() / f"global_step_{step}"
    proof = read(folder / "mopd_complete.json")
    need(proof["identity"] == identity and proof["arm"] == arm and proof["step"] == step
         and proof["sampler_identity"] == sampler_identity, "Evaluation checkpoint identity mismatch")
    validate_native(folder, step)
    need(inventory(folder) == proof["contents"], "Evaluation checkpoint changed")
    return dict(checkpoint=str(folder), checkpoint_identity=digest(proof))


def restore_selected(selection, root, identity, arm, sampler_identity):
    """An evaluation job loads its frozen selection, never the latest train step."""
    need(selection["identity"] == identity and selection["arm"] == arm
         and selection["split"] == "validation" and not selection["test_used_for_selection"],
         "Test selection is not bound to validation")
    row = read(selection["validation_receipt"])
    need(digest(row) == selection["validation_sha256"], "Selected validation receipt changed")
    need(row["identity"] == identity and row["arm"] == arm and row["step"] == selection["step"]
         and row["split"] == "validation" and row["status"] == "PASS"
         and row["metrics"][selection["selection_key"]] == selection["score"],
         "Selected score/step does not match validation")
    ref = evaluation_reference(root, identity, arm, selection["step"], sampler_identity)
    need(all(selection[k] == v and row[k] == v for k, v in ref.items()),
         "Selected checkpoint does not match validation")
    return Path(ref["checkpoint"]) if ref["checkpoint"] else None


def prune(run, identity, arm, sampler_identity, keep_recent, final_step):
    """Under the run writer lock, keep recent full states and frozen selections.

    Retire directories by atomic rename before deletion. A kill during rmtree
    cannot leave a half-deleted checkpoint in the namespace used for recovery.
    Journals are retained after deletion, including the original commit marker.
    """
    from .metrics import select_best
    run = Path(run)
    root = run / "checkpoints"
    need(type(keep_recent) is int and keep_recent >= 2, "Unsafe checkpoint retention count")
    need(not root.is_symlink(), "Checkpoint root must not be a symlink")
    root = root.resolve()
    entries = {}
    for marker in root.glob("global_step_*/mopd_complete.json"):
        proof = read(marker)
        step, folder = proof["step"], marker.parent
        need(type(step) is int and step > 0 and proof["status"] == "PASS"
             and proof["identity"] == identity and proof["arm"] == arm
             and proof["sampler_identity"] == sampler_identity,
             "Retention checkpoint belongs to another experiment/sampler")
        need(not folder.is_symlink() and folder.name == f"global_step_{step}"
             and folder.resolve().parent == root, "Unsafe checkpoint retention path")
        entries[step] = (folder, proof)
    # Rebuild the pointer from immutable validation receipts before removing any
    # model. This also covers a kill between writing a receipt and its best pointer.
    best = select_best(run, identity, arm)
    protected = set(sorted(entries)[-keep_recent:]) | {best["step"], final_step}
    restore_selected(best, root, identity, arm, sampler_identity)
    completed = run / "receipt.json"
    if completed.exists():
        proof = read(completed)
        need(proof["status"] == "PASS" and proof["identity"] == identity and proof["arm"] == arm,
             "Completed training selection changed")
        selection = proof["selection"]
        restore_selected(selection, root, identity, arm, sampler_identity)
        protected.add(selection["step"])
    # Validate even the recovery checkpoints before pruning older states.
    for step, (folder, proof) in entries.items():
        validate_native(folder, step)
        need(inventory(folder) == proof["contents"], "Committed checkpoint changed before retention")
    journal = run / "checkpoint_retention"
    trash = root / ".retention_trash"
    need(not journal.is_symlink() and not trash.is_symlink(), "Unsafe retention journal/trash path")
    journal.mkdir(exist_ok=True)
    trash.mkdir(parents=True, exist_ok=True)

    def finish(path, event):
        need(event["identity"] == identity and event["arm"] == arm
             and event["sampler_identity"] == sampler_identity
             and event["status"] in ("PENDING", "DELETED"), "Unbound retention event")
        if event["status"] == "DELETED":
            return
        step = event["step"]
        need(type(step) is int and step > 0 and step not in protected, "Retention would delete a protected checkpoint")
        source, target = root / f"global_step_{step}", trash / path.stem
        need(not source.is_symlink() and not target.is_symlink(), "Unsafe retired checkpoint path")
        need(event["source"] == str(source) and event["trash"] == str(target), "Retention event path changed")
        if source.exists():
            need(not target.exists() and step in entries
                 and entries[step][1] == event["checkpoint_marker"], "Retired checkpoint identity changed")
            source.rename(target)
        if target.exists():
            shutil.rmtree(target)
        write(path, dict(event, status="DELETED", deleted_utc=now()))

    # Finish interrupted deletions before creating new ones. No old model is
    # resurrected or considered by latest(), even if its trash removal was killed.
    for path in sorted(journal.glob("*.json")):
        finish(path, read(path))
    for step, (folder, proof) in sorted(entries.items()):
        if step in protected or not folder.exists():
            continue
        path = journal / (uuid.uuid4().hex + ".json")
        event = dict(status="PENDING", identity=identity, arm=arm, sampler_identity=sampler_identity,
                     step=step, source=str(folder), trash=str(trash / path.stem),
                     checkpoint_marker=proof, planned_utc=now())
        write(path, event)
        finish(path, event)
    events = [read(p) for p in journal.glob("*.json")]
    result = dict(status="PASS", identity=identity, arm=arm, keep_recent=keep_recent,
                  best_step=best["step"], final_step=final_step,
                  retained_steps=sorted(s for s, (p, _) in entries.items() if p.exists()),
                  deleted_steps=sorted({e["step"] for e in events if e["status"] == "DELETED"}), at_utc=now())
    write(run / "retention.json", result)
    return result
