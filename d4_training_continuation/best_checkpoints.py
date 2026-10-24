"""Immutable best-model archives, separate from rolling checkpoints/recovery pins.

No tensor deserialization or GPU imports. Large shards are checked against the
writer's size/mtime inventory; this is NOT a claim of tensor/content equality.
Hard links protect against unlink/retention, not disk failure or in-place writes.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import time
import uuid

from . import checkpoints as ck, evaluation as ev
from .common import below, digest, need, now, paths, read, sha, write

POLICY = dict(version="2026-10-24.best-checkpoints.1", rolling_full=2,
              overall="four-domain-macro-mean", domain_weights=1,
              separate_validation_protocols=True, ties="keep-earlier-step")


def area(root, kind):
    return paths(root, kind)["run"] / "checkpoint_archive"


@contextmanager
def lease(directory, timeout=45):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "archive.lock").open("a") as handle:
        until = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                need(time.monotonic() < until, "Checkpoint archive is busy; retry")
                time.sleep(.1)
        yield


def file_matches(path, expected):
    need(not path.is_symlink() and path.is_file(), "Missing/unsafe archive file: " + str(path))
    stat = path.stat()
    need(stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"],
         "Archive file inventory changed: " + str(path))
    if expected.get("sha256"):
        need(sha(path) == expected["sha256"], "Archive metadata checksum changed: " + str(path))


def validate_files(root, inventory):
    root = Path(root)
    need(inventory, "Empty archive inventory")
    for name, expected in inventory.items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts, "Unsafe archive member")
        path = root / relative
        for parent in (path, *path.parents):
            if parent == root.parent:
                break
            need(not parent.is_symlink(), "Archive source contains a symlink")
        below(path, root)
        file_matches(path, expected)


def model_files(record):
    """Preserve the original Megatron model format plus its loading metadata."""
    root = Path(record["path"])
    spec = read(root / "actor/ckpt_contents.json")
    need(spec.get("schema_version") == 2 and spec.get("global_step") == record["step"]
         and spec.get("world_size") == record["world_size"], "Wrong model checkpoint metadata")
    model = spec["contents"]["model"]
    relative = Path(model["path"])
    need(not relative.is_absolute() and ".." not in relative.parts
         and model.get("format") == "megatron_dist_checkpoint", "Unsupported model checkpoint")
    prefix = "actor/" + str(relative).rstrip("/") + "/"
    # A shared model/optimizer payload cannot honestly be called weight-only.
    for section in ("optimizer", "lr_scheduler", "rng_state"):
        other = spec.get("contents", {}).get(section, {}).get("path")
        need(other != model["path"], "Model and training state share one payload")
    selected = {k: v for k, v in record["files"].items() if k.startswith(prefix)
                or k in ("actor/ckpt_contents.json", "actor/transformer_config.json")
                or k.startswith("actor/huggingface/")}
    need(prefix + ".metadata" in selected and any(k.endswith(".distcp") for k in selected),
         "Missing model shard inventory/metadata")
    need("actor/transformer_config.json" in selected, "Missing transformer configuration")
    validate_files(root, selected)
    return selected


def link_tree(source, target, inventory):
    """Publish only a complete, revalidated link set; interrupted copies are private."""
    source, target = Path(source), Path(target)
    if target.exists():
        validate_files(target, inventory)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / (".linking_" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        validate_files(source, inventory)
        for name in inventory:
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source / name, destination, follow_symlinks=False)
        validate_files(temporary, inventory)
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class Archive:
    def __init__(self, root, kind):
        self.root, self.kind = Path(root).resolve(), kind
        self.path = area(root, kind)

    def object_dir(self, key):
        need(len(key) == 64 and all(c in "0123456789abcdef" for c in key), "Invalid checkpoint key")
        return self.path / "objects" / key

    def _put(self, record, hold=None, trusted_inventory=True):
        p = paths(self.root, self.kind)
        allowed = [p[name] for name in ("run", "single", "old")]
        allowed.append(self.root / "outputs/multidomain/validation_backfill_sources" /
                       (self.kind + "_step_" + str(record["step"])))
        need(any(Path(record["path"]).resolve().is_relative_to(x) for x in allowed),
             "Checkpoint source is outside this setup")
        key = ev.checkpoint_key(record)
        directory = self.object_dir(key)
        manifest = directory / "manifest.json"
        prior = read(manifest) if manifest.exists() else None
        if prior:
            need(prior["kind"] == self.kind and prior["key"] == key, "Foreign archive object")
            # Resume pins may have been deleted after a previous successful rescue.
            self.verify(prior)
            if hold and hold not in prior["holds"]:
                prior["holds"].append(hold)
                write(manifest, prior)
            return prior
        intent = directory / "intent.json"
        if intent.exists():
            pending = read(intent)
            need(ev.checkpoint_key(pending["record"]) == key, "Archive intent identity changed")
            hold = hold or pending.get("hold")
            trusted_inventory = trusted_inventory and pending["trusted_inventory"]
        else:
            write(intent, dict(record=record, hold=hold, trusted_inventory=trusted_inventory, at_utc=now()))
        payload = directory / "payload" / ("global_step_" + str(record["step"]))
        # Recover a crash after the atomic link-tree rename but before the
        # manifest write, even if rolling retention has removed the origin.
        source_record = dict(record, path=str(payload)) if payload.exists() else record
        model = model_files(source_record)
        mode, full_error = "weights", None
        if trusted_inventory:
            try:
                ck.validate(source_record)
                mode = "full"
            except (OSError, RuntimeError, KeyError, ValueError) as error:
                full_error = str(error)
        else:
            mode = "unverified_weights"
        inventory = record["files"] if mode == "full" else model
        if mode == "full":
            # Include completion metadata so the archive remains an actual full
            # continuation source. It is deliberately outside the original inventory.
            inventory = dict(inventory)
            marker = Path(source_record["path"]) / ck.MARKER
            if record.get("lineage"):
                stat = marker.stat()
                inventory[ck.MARKER] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(marker))
        link_tree(source_record["path"], payload, inventory)
        result = dict(version=POLICY["version"], kind=self.kind, key=key, step=record["step"],
                      lineage=record.get("lineage"), mode=mode, original=record, payload=str(payload),
                      inventory=inventory, model_inventory=model, full_error=full_error,
                      trusted_inventory=trusted_inventory, holds=[hold] if hold else [], created_utc=now())
        self.verify(result)
        write(manifest, result)
        (directory / "retired.json").unlink(missing_ok=True)
        return result

    def put(self, record, hold=None, trusted_inventory=True):
        with lease(self.path):
            return self._put(record, hold, trusted_inventory)

    def verify(self, obj):
        payload = below(obj["payload"], self.object_dir(obj["key"]))
        validate_files(payload, obj["inventory"])
        if obj["mode"] == "full":
            ck.validate(dict(obj["original"], path=str(payload)))
        return obj

    def objects(self):
        return [read(p) for p in (self.path / "objects").glob("*/manifest.json")]

    def accept(self, request, record):
        """Only complete checkpoint/protocol-bound evaluations may select best."""
        need(request["kind"] == self.kind and request["identity"] == digest(request["contract"]),
             "Evaluation request identity/setup mismatch")
        need(request.get("validation_semantics") == request["contract"].get("validation_semantics"),
             "Evaluation semantics are not bound")
        need(request["lineage"] == record["lineage"]
             and Path(request["control_dir"]).resolve() == paths(self.root, self.kind)["control"],
             "Evaluation checkpoint/control belongs to another setup")
        receipt = ev.completed(request, record)
        need(receipt is not None, "No completed validation receipt")
        with lease(self.path):
            obj = self._put(record)
            need(obj["trusted_inventory"], "Unverified legacy weights cannot select best")
            proof = dict(receipt=receipt, request_identity=request["identity"],
                         kind=self.kind, key=obj["key"], protocol=ev.protocol(request),
                         model_context=request.get("config"), verified_utc=now())
            target = self.path / "evaluations" / (obj["key"] + "_" + proof["protocol"] + ".json")
            if target.exists():
                previous = read(target)
                need(previous["receipt"] == receipt, "Previously archived evaluation changed")
                return previous
            write(target, proof)
            return proof

    def reconcile(self, approved_lineages, live_lineages):
        """Publish winners BEFORE releasing any archive-owned payload links.

Only our archive is compacted. Never mutate source checkpoints, recovery pins,
backfill sources, or training/controller state. Unknown evaluations stay pinned.
"""
        with lease(self.path):
            objects = {x["key"]: x for x in self.objects()}
            cohorts, evaluated = {}, set()
            for path in sorted((self.path / "evaluations").glob("*.json")):
                proof = read(path)
                obj = objects.get(proof["key"])
                if not obj or obj["lineage"] not in approved_lineages:
                    continue
                need(proof["kind"] == self.kind, "Foreign archive evaluation")
                receipt = proof["receipt"]
                need(receipt["checkpoint_key"] == obj["key"] and receipt["protocol"] == proof["protocol"],
                     "Archive evaluation binding changed")
                metrics = ev.numeric_metrics(receipt["metrics"])
                scores = {d: metrics["val-core/" + d + "/acc/mean@1"] for d in ev.DOMAINS}
                scores["overall"] = sum(scores.values()) / len(ev.DOMAINS)
                if obj["mode"] == "full":
                    scores["overall_full"] = scores["overall"]
                cohort = cohorts.setdefault(proof["protocol"], {})
                evaluated.add(obj["key"])
                for metric, score in scores.items():
                    old = cohort.get(metric)
                    value = dict(key=obj["key"], step=obj["step"], score=score, mode=obj["mode"],
                                 path=obj["payload"], lineage=obj["lineage"])
                    if old is None or (-score, obj["step"], obj["key"]) < (-old["score"], old["step"], old["key"]):
                        cohort[metric] = value
            full_keep, weights_keep = set(), set()
            for cohort in cohorts.values():
                weights_keep.update(v["key"] for v in cohort.values())
                full_keep.add(cohort["overall"]["key"])
                if "overall_full" in cohort:
                    full_keep.add(cohort["overall_full"]["key"])
            for lineage in live_lineages:
                recent = sorted((x for x in objects.values() if x["lineage"] == lineage),
                                key=lambda x: x["step"], reverse=True)[:POLICY["rolling_full"]]
                full_keep.update(x["key"] for x in recent)
            # All explicit rescue holds and not-yet-evaluated validation points
            # are preserved. Do not silently expire an interrupted validation.
            for obj in objects.values():
                if obj["holds"] or (obj["step"] % 25 == 0 and obj["key"] not in evaluated):
                    full_keep.add(obj["key"])
            keep = full_keep | weights_keep
            for key in keep:
                self.verify(objects[key])
            for cohort in cohorts.values():
                for winner in cohort.values():
                    if winner["key"] not in full_keep and winner["mode"] == "full":
                        winner["mode"] = "weights"
            index = dict(version=POLICY["version"], kind=self.kind, at_utc=now(), cohorts=cohorts,
                         full_keep=sorted(full_keep), weights_keep=sorted(weights_keep),
                         approved_lineages=sorted(x for x in approved_lineages if x),
                         live_lineages=sorted(x for x in live_lineages if x))
            write(self.path / "index.json", index)
            # Retain manifest/provenance even when payload is retired. Retired
            # objects are not considered by objects()/ranking on subsequent ticks.
            for key, obj in objects.items():
                directory = self.object_dir(key)
                payload = below(obj["payload"], directory)
                if key not in keep:
                    # Removing the manifest from the live namespace is atomic.
                    # A crash during unlinking cannot leave a live index entry
                    # whose payload is half-deleted.
                    os.replace(directory / "manifest.json", directory / "retired.json")
                    if payload.exists():
                        shutil.rmtree(payload)
                elif key not in full_keep and obj["mode"] == "full":
                    # Publish a valid weight-only representation first, then
                    # unlink only the extra files owned by this archive.
                    reduced = dict(obj, mode="weights", inventory=obj["model_inventory"])
                    write(directory / "manifest.json", reduced)
                    for name in obj["inventory"].keys() - reduced["inventory"].keys():
                        (payload / name).unlink(missing_ok=True)
            # Finish a prior crash after manifest retirement and before unlink.
            for retired in (self.path / "objects").glob("*/retired.json"):
                if not (retired.parent / "manifest.json").exists() and retired.parent.name not in keep:
                    old = read(retired)
                    payload = below(old["payload"], retired.parent)
                    if payload.exists():
                        shutil.rmtree(payload)
            return index


def capture_saved(request, record):
    if request.get("best_checkpoint_policy"):
        need(request["best_checkpoint_policy"] == POLICY, "Unknown best checkpoint policy")
        # Capture in the writer, before rolling retention can unlink anything.
        archive = Archive(request["root"], request["kind"])
        archive.put(record)
        from .best_archiver import scope
        approved, live = scope(request["root"], request["kind"])
        archive.reconcile(approved, live | {request["lineage"]})


def capture_evaluation(request, record):
    if request.get("best_checkpoint_policy"):
        need(request["best_checkpoint_policy"] == POLICY, "Unknown best checkpoint policy")
        archive = Archive(request["root"], request["kind"])
        archive.accept(request, record)
        from .best_archiver import scope
        approved, live = scope(request["root"], request["kind"])
        archive.reconcile(approved, live | {request["lineage"]})
