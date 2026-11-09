"""Commit marker after native model/optimizer/scheduler/RNG and DataLoader save."""
from pathlib import Path
from d4_teachers.common import digest, file_sha, need, read, write


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
