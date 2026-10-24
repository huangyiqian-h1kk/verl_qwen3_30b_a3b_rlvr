"""Online rescue/bridge for immutable, already-running training snapshots."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

from . import checkpoints as ck, evaluation as ev
from .best_checkpoints import Archive, POLICY, area, model_files
from .common import active, digest, files, locked, need, now, paths, read, sha, verify_files, write

TARGETS = {"random": (50, 100, 125, 150), "domain": (75, 100, 125, 150)}


def service(root):
    return Path(root).resolve() / "outputs/multidomain/checkpoint_archive_service"


def requests(root, kind):
    result = []
    for path in (paths(root, kind)["run"] / "jobs").glob("*/request.json"):
        row = read(path)
        if row.get("kind") == kind and row.get("identity") == digest(row.get("contract")):
            result.append(row)
    return result


def scope(root, kind):
    p = paths(root, kind)
    state = read(p["control"] / "state.json")
    approved = {r["lineage"] for r in requests(root, kind) if r.get("role") == "formal"}
    authority = state.get("authority") or {}
    if authority.get("lineage"):
        approved.add(authority["lineage"])
    live = {authority["lineage"]} if authority.get("lineage") else set()
    # A qualifying candidate's scores never select best before authority commits.
    for role in ("candidate",):
        job = state.get(role)
        if job and not state.get("handoff_failed") and state.get("phase") in ("QUALIFYING", "STOPPING_SINGLE", "CATCH_UP"):
            path = Path(job["folder"]) / "request.json"
            if path.exists():
                live.add(read(path)["lineage"])
    from .opportunities import IN_FLIGHT
    opportunity = state.get("opportunity") or {}
    job = opportunity.get("job") if opportunity.get("phase") in IN_FLIGHT else None
    if job and (Path(job["folder"]) / "request.json").exists():
        live.add(read(Path(job["folder"]) / "request.json")["lineage"])
    return approved, live


def candidates(root, kind, historical=False):
    """Bounded metadata traversal: never recursively walk datasets or tensors."""
    p = paths(root, kind)
    rows, diagnostics = [], []
    def add(value, origin, trusted=True):
        if isinstance(value, dict) and all(k in value for k in ("step", "path", "files", "world_size", "lineage")):
            rows.append((value, origin, trusted))
    for request in requests(root, kind):
        add(request.get("source", {}).get("record"), request["job_dir"] + "/request.json")
        for phase in ("train", "A", "B", "C", "D", "E"):
            for name in ("resume_source.json", "source/source.json"):
                path = Path(request["job_dir"]) / phase / name
                if path.exists():
                    add(read(path), str(path))
    for path in (p["control"] / "resume_sources").glob("*/source.json"):
        add(read(path), str(path))
    for path in (p["control"] / "fallback").glob("*/source.json"):
        add(read(path), str(path))
    for path in (p["run"] / "lineages").glob("*/checkpoints/global_step_*/" + ck.MARKER):
        add(dict(read(path), path=str(path.parent)), str(path))
    for path in (Path(root) / "outputs/multidomain/validation_backfills").glob("*/state.json"):
        add(read(path).get("sources", {}).get(kind), str(path))
    # Persisted archives remain valid after the original resume source disappears.
    for obj in Archive(root, kind).objects():
        add(dict(obj["original"], path=obj["payload"]), str(area(root, kind)), obj["trusted_inventory"])
    for path in (area(root, kind) / "objects").glob("*/intent.json"):
        if (path.parent / "manifest.json").exists() or (path.parent / "retired.json").exists():
            continue
        intent = read(path)
        record = intent["record"]
        payload = path.parent / "payload" / ("global_step_" + str(record["step"]))
        if payload.exists():
            add(dict(record, path=str(payload)), str(path), intent["trusted_inventory"])
    if historical:
        for run in (p["single"], p["old"], *sorted((p["run"] / "lineages").glob("*"))):
            for step in TARGETS[kind]:
                cp = run / "checkpoints" / ("global_step_" + str(step))
                specfile = cp / "actor/ckpt_contents.json"
                if not specfile.exists() or (cp / ck.MARKER).exists():
                    continue
                try:
                    spec = read(specfile)
                    model = Path(spec["contents"]["model"]["path"])
                    need(not model.is_absolute() and ".." not in model.parts, "Unsafe legacy model path")
                    inventory = {}
                    sources = list((cp / "actor" / model).rglob("*")) + [specfile, cp / "actor/transformer_config.json"]
                    for file in sources:
                        if file.is_file():
                            stat = file.stat()
                            inventory[str(file.relative_to(cp))] = dict(bytes=stat.st_size,
                                mtime_ns=stat.st_mtime_ns, sha256=sha(file) if stat.st_size <= 1048576 else None)
                    record = dict(path=str(cp), step=step, world_size=spec["world_size"], lineage=None, files=inventory)
                    model_files(record)
                    add(record, str(specfile), False)
                except (OSError, RuntimeError, ValueError, KeyError) as error:
                    diagnostics.append(dict(path=str(cp), error=str(error)))
    unique = {}
    for record, origin, trusted in rows:
        unique[(ev.checkpoint_key(record), record["path"])] = (record, origin, trusted)
    return list(unique.values()), diagnostics


def tick(root, kind, historical=False):
    archive = Archive(root, kind)
    approved, live = scope(root, kind)
    records, errors = candidates(root, kind, historical)
    captured, bykey = {}, {}
    # A rescued source is processed before metadata from old, retired copies.
    records.sort(key=lambda x: (not Path(x[0]["path"]).exists(), -x[0]["step"]))
    for record, origin, trusted in records:
        key = ev.checkpoint_key(record)
        if key in captured:
            continue
        relevant = record["step"] % 25 == 0 or record.get("lineage") in live
        if not relevant:
            continue
        objfile = archive.object_dir(key) / "manifest.json"
        retired = archive.object_dir(key) / "retired.json"
        if retired.exists() and not objfile.exists() and not historical:
            new_receipt = any(not (archive.path / "evaluations" / p.name).exists()
                              for p in (paths(root, kind)["control"] / "evaluations").glob(key + "_*.json"))
            if not new_receipt:
                continue
        try:
            # Explicit historical rescue is sticky and has no automated expiry.
            hold = ("historical-rescue-step-" + str(record["step"])) if historical and record["step"] in TARGETS[kind] else None
            obj = archive.put(record, hold=hold, trusted_inventory=trusted)
            captured[key] = dict(step=obj["step"], mode=obj["mode"], path=obj["payload"], hold=hold)
            bykey[key] = dict(obj["original"], path=obj["payload"])
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            if Path(record["path"]).exists():
                errors.append(dict(path=record["path"], step=record["step"], error=str(error)))
    reqs = {r["job_dir"]: r for r in requests(root, kind)}
    receipt_errors = []
    for path in sorted((paths(root, kind)["control"] / "evaluations").glob("*.json")):
        receipt = read(path)
        key = receipt.get("checkpoint_key")
        request = reqs.get((receipt.get("origin") or {}).get("job_dir"))
        if not request:
            origin_job = (receipt.get("origin") or {}).get("job") or {}
            request = reqs.get(origin_job.get("folder"))
        if key not in bykey or request is None:
            continue
        proof = archive.path / "evaluations" / (key + "_" + receipt["protocol"] + ".json")
        if proof.exists():
            # Mutation is reported; do not silently overwrite a verified receipt.
            if read(proof)["receipt"] != receipt:
                receipt_errors.append(dict(path=str(path), error="Previously archived receipt changed"))
            continue
        try:
            archive.accept(request, bykey[key])
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            receipt_errors.append(dict(path=str(path), error=str(error)))
    index = archive.reconcile(approved, live)
    result = dict(kind=kind, at_utc=now(), captured=list(captured.values()),
                  errors=errors, receipt_errors=receipt_errors, cohorts=index["cohorts"],
                  historical=historical, payload_integrity="inventory-checked; not tensor-load-tested")
    if historical:
        result["rescue_targets"] = {str(step): [x for x in captured.values() if x["step"] == step]
                                    for step in TARGETS[kind]}
        write(archive.path / "rescue_report.json", result)
    write(archive.path / "last_scan.json", result)
    return result


def brief(result):
    return {k: result.get(k) for k in ("kind", "at_utc", "errors", "receipt_errors", "cohorts", "rescue_targets") if k in result}


def status(root):
    directory = service(root)
    state = read(directory / "state.json") if (directory / "state.json").exists() else {}
    heartbeat = state.get("heartbeat_utc")
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(heartbeat)).total_seconds() if heartbeat else None
    result = dict(service=state, lock_held=active(directory / "watch.lock"), heartbeat_age_seconds=age)
    result["bridge_active"] = result["lock_held"] and state.get("status") == "RUNNING" and age is not None and age < 120
    result["setups"] = {}
    for kind in ("random", "domain"):
        p = area(root, kind)
        plan_state = paths(root, kind)["control"] / "state.json"
        policy, policy_error = None, None
        if plan_state.exists():
            try:
                s = read(plan_state)
                plan = read(paths(root, kind)["control"] / "operations" / s["operation"] / "plan.json")
                policy = plan.get("best_checkpoint_policy")
            except (OSError, ValueError, KeyError) as error:
                policy_error = str(error)
        result["setups"][kind] = dict(native_plan_policy=policy,
            policy_read_error=policy_error,
            scan=brief(read(p / "last_scan.json")) if (p / "last_scan.json").exists() else None,
            rescue=read(p / "rescue_report.json").get("rescue_targets") if (p / "rescue_report.json").exists() else None,
            deployment=read(p / "deployment.json") if (p / "deployment.json").exists() else None)
    return result


def start(root):
    directory = service(root)
    if active(directory / "watch.lock"):
        return
    source = Path(__file__).parent
    manifest = {p.name: sha(p) for p in source.glob("*.py")}
    frozen = directory / "runtimes" / digest(manifest)
    package = frozen / "d4_training_continuation"
    entry_text = "from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent))\n" \
                 "from d4_training_continuation.best_archiver import main\nmain()\n"
    if not frozen.exists():
        frozen.parent.mkdir(parents=True, exist_ok=True)
        temp = frozen.parent / (".staging_" + uuid.uuid4().hex)
        try:
            staged = temp / "d4_training_continuation"
            staged.mkdir(parents=True)
            for name in manifest:
                shutil.copy2(source / name, staged / name)
            verify_files(staged, manifest)
            write(temp / "manifest.json", manifest)
            (temp / "entry.py").write_text(entry_text)
            temp.rename(frozen)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    verify_files(package, manifest)
    entry = frozen / "entry.py"
    need(entry.read_text() == entry_text, "Frozen archive entry changed")
    (directory / "stop.request").unlink(missing_ok=True)
    with (directory / "watch.log").open("a") as out:
        subprocess.Popen([sys.executable, "-B", "-I", str(entry), "_watch", "--root", str(Path(root).resolve())],
                         stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)


def watch(root):
    directory = service(root)
    with locked(directory / "watch.lock"):
        state = dict(pid=os.getpid(), host=socket.gethostname(), started_utc=now(),
                     runtime=str(Path(__file__).parent), version=POLICY["version"], errors={})
        write(directory / "state.json", dict(state, status="STARTING", heartbeat_utc=now()))
        try:
            while not (directory / "stop.request").exists():
                for kind in ("domain", "random"):
                    try:
                        tick(root, kind)
                        state["errors"].pop(kind, None)
                        from .best_upgrade import deploy_kind
                        deploy_kind(root, kind, signal_local=False)
                    except (OSError, RuntimeError, ValueError, KeyError) as error:
                        state["errors"][kind] = type(error).__name__ + ": " + str(error)
                    write(directory / "state.json", dict(state, status="RUNNING", heartbeat_utc=now()))
                for _ in range(15):
                    if (directory / "stop.request").exists():
                        break
                    time.sleep(1)
        finally:
            write(directory / "state.json", dict(state, status="STOPPED", heartbeat_utc=now()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("rescue", "install", "scan", "status", "stop", "_watch", "self-test"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.action == "self-test":
        import unittest
        tests = Path(__file__).resolve().parents[1] / "tests/multidomain"
        suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_training_continuation_best*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.testsRun and result.wasSuccessful() else 1)
    if args.action == "_watch":
        return watch(root)
    if args.action == "status":
        result = status(root)
    elif args.action == "stop":
        service(root).mkdir(parents=True, exist_ok=True)
        (service(root) / "stop.request").touch()
        result = dict(status="ARCHIVER_STOP_REQUESTED", training_jobs_unchanged=True)
    else:
        result = {}
        with locked(service(root) / "install.lock"):
            for kind in ("random", "domain"):
                try:
                    result[kind] = brief(tick(root, kind, historical=args.action in ("rescue", "install")))
                except (OSError, RuntimeError, ValueError, KeyError) as error:
                    result[kind] = dict(error=str(error))
            if args.action == "install":
                # Bridge first: template/controller upgrade must never postpone rescue.
                start(root)
                from .best_upgrade import deploy_kind
                for kind in ("domain", "random"):
                    try:
                        result[kind]["deployment"] = deploy_kind(root, kind, signal_local=True)
                    except (OSError, RuntimeError, ValueError, KeyError) as error:
                        result[kind]["deployment_error"] = str(error)
                until = time.monotonic() + 10
                while time.monotonic() < until and not status(root)["bridge_active"]:
                    time.sleep(.25)
                result["archive_status"] = status(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "install" and (not result["archive_status"]["bridge_active"]
                                    or any(result[kind].get("error") for kind in ("random", "domain"))):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
