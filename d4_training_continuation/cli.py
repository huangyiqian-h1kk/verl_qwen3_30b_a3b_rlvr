"""Stage migration before admitting candidates; repeated operation IDs are safe."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from . import VERSION
from .common import acquire, active, extended_deadline, locked, need, now, paths, read, safe_id, verify_files, write
from .source import inspect, prepare
from .controller import Controller, authoritative_checkpoint
from .scheduler import PBS, Unknown

KINDS = ("domain", "random")


def initial_state(plan, hours):
    p = paths(plan["root"], plan["kind"])
    old = plan["old_state"]
    return dict(version=VERSION, operation=plan["operation"], kind=plan["kind"],
                phase="QUALIFYING", promoted=False, handoff_failed=False, single_fenced=False,
                active=old["single"], authority=dict(run=str(p["single"]), legacy=True, lineage=None),
                old_candidate=old.get("dual"), old_candidate_retired=False, candidate=None,
                candidate_submitted=False, candidate_admitted=False, pending={},
                deadline_utc=extended_deadline(old.get("deadline_utc"), hours),
                previous_deadline_utc=old.get("deadline_utc"), extension_hours=hours,
                controller_status="PREPARED", migration_complete=False, created_utc=now())


def status(root, query=True):
    output = {}
    for kind in KINDS:
        p = paths(root, kind)
        if not (p["control"] / "state.json").exists():
            output[kind] = dict(installed=False)
            continue
        state = read(p["control"] / "state.json")
        keys = ("version", "operation", "kind", "phase", "promoted", "handoff_failed", "failure_reason",
                "controller_status", "pid", "host", "heartbeat_utc", "deadline_utc", "previous_deadline_utc",
                "extension_hours", "migration_complete", "candidate_admitted", "active", "candidate",
                "old_candidate", "old_candidate_retired", "pending", "retry_selection", "training_blocked", "last_error", "wait_reason")
        result = {k: state.get(k) for k in keys}
        result["controller_active"] = active(p["control"] / "controller.lock")
        result["deadline_beijing"] = datetime.fromisoformat(state["deadline_utc"]).astimezone(ZoneInfo("Asia/Shanghai")).isoformat()
        if query:
            result["pbs"] = {}
            for role in ("active", "candidate"):
                if state.get(role):
                    try:
                        row = PBS().observe(state[role])
                        result["pbs"][role] = {k: row.get(k) for k in ("Job_Name", "queue", "job_state", "Exit_status", "Hold_Types", "Resource_List", "exec_vnode", "comment")}
                    except (Unknown, RuntimeError) as e:
                        result["pbs"][role] = dict(query_error=str(e))
        output[kind] = result
    return output


def preflight(root):
    result = {}
    for kind in KINDS:
        p = paths(root, kind)
        if (p["control"] / "state.json").exists():
            state = read(p["control"] / "state.json")
            need(state["version"] == VERSION, "Existing continuation version requires explicit migration")
            plan = read(p["control"] / "operations" / state["operation"] / "plan.json")
            verify_files(plan["template"], plan["template_files"])
            result[kind] = dict(status="INSTALLED", operation=state["operation"], deadline=state["deadline_utc"])
            continue
        item = inspect(root, kind)
        state = item["state"]
        rows = {role: PBS().observe(state[role]) for role in ("single", "dual") if state.get(role)}
        checkpoint = authoritative_checkpoint(dict(run=str(p["single"]), legacy=True), kind)
        result[kind] = dict(status="READY", single=state["single"], old_candidate=state.get("dual"),
                            checkpoint_step=checkpoint["step"], deadline=state.get("deadline_utc"),
                            pbs_states={k: v["job_state"] for k, v in rows.items()})
    need(importlib.metadata.version("megatron-core") == "0.13.1", "Expected pinned Megatron 0.13.1")
    return result


def spawn(root, kind):
    p = paths(root, kind)
    if active(p["control"] / "controller.lock"):
        return
    state = read(p["control"] / "state.json")
    operation = p["control"] / "operations" / state["operation"]
    plan = read(operation / "plan.json")
    verify_files(plan["template"], plan["template_files"])
    entry = operation / "controller_entry.py"
    entry.write_text("import sys\n" + f"sys.path.insert(0, {str(operation / 'template/runtime')!r})\n"
                     "from d4_training_continuation.cli import main\nmain()\n")
    with (p["control"] / "controller.log").open("a") as out:
        subprocess.Popen([sys.executable, "-I", str(entry), "_run", "--root", str(p["root"]), "--kind", kind],
                         stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)


def apply(root, operation, hours):
    safe_id(operation)
    preflight(root)  # No old stop request is written until BOTH setups are prepared.
    area = Path(root) / "outputs/multidomain/continuation_operations"
    with locked(area / "apply.lock"):
        for kind in KINDS:
            p = paths(root, kind)
            existing = p["control"] / "state.json"
            if existing.exists():
                state = read(existing)
                need(state["operation"] == operation and state["extension_hours"] == hours,
                     "Use the SAME apply operation; use extend for another deadline extension")
            else:
                plan = prepare(root, kind, operation)
                write(existing, initial_state(plan, hours))
        for kind in KINDS:
            spawn(root, kind)
        until = time.monotonic() + 150
        while True:
            states = status(root, query=False)
            if all(s.get("controller_active") and s.get("migration_complete") for s in states.values()):
                break
            need(time.monotonic() < until, "Owner adoption not yet confirmed; rerun SAME apply operation. See controller.log")
            time.sleep(2)
        # Candidate failure/submission cannot prevent either single monitor from running.
        for kind in KINDS:
            write(paths(root, kind)["control"] / "admit.json", dict(operation=operation, at_utc=now()))
        write(area / (operation + ".json"), dict(status="MONITORS_ADOPTED_CANDIDATES_ADMITTED", at_utc=now(),
               deadlines={k: s["deadline_utc"] for k, s in states.items()}))
    # A bounded receipt wait makes an asynchronous admission distinguishable
    # from successful qsub. Owners continue even when this CLI times out.
    until = time.monotonic() + 90
    while time.monotonic() < until:
        current = status(root, query=False)
        if all(s.get("candidate") or s.get("handoff_failed") or s.get("last_error") for s in current.values()):
            break
        time.sleep(2)
    report = status(root)
    return dict(deployment_status="CANDIDATES_SUBMITTED" if all(s.get("candidate") for s in report.values())
                else "MONITORS_ADOPTED_CANDIDATES_PENDING_OR_BLOCKED", setups=report)


def extend(root, operation, hours):
    safe_id(operation)
    area = Path(root) / "outputs/multidomain/continuation_operations"
    with locked(area / "apply.lock"):
        for kind in KINDS:
            p = paths(root, kind)
            state = read(p["control"] / "state.json")
            file = p["control"] / "extensions" / (operation + ".json")
            if file.exists():
                need(read(file)["hours"] == hours, "Extension operation already has another duration")
            else:
                previous = state["deadline_utc"]
                for old in (p["control"] / "extensions").glob("*.json"):
                    previous = max(previous, read(old)["target"])
                write(file, dict(operation=operation, hours=hours, previous=previous,
                                 target=extended_deadline(previous, hours), at_utc=now()))
            spawn(root, kind)
    return status(root)


def daemon(root, kind):
    p = paths(root, kind)
    with ExitStack() as stack:
        stack.enter_context(locked(p["control"] / "controller.lock"))
        controller = Controller(root, kind)
        controller.save(controller_status="ADOPTING", pid=os.getpid(), host=socket.gethostname())
        # Spawn succeeded; old owner may now be asked to release its leases.
        for directory in (p["old_control"], p["monitor"]):
            (directory / "stop.request").touch()
        legacy_locks = [p["old_control"] / "controller.lock", p["monitor"] / "monitor.lock",
                        p["single"] / "submission.lock", p["old"] / "submission.lock"]
        for path in legacy_locks:
            while True:
                try:
                    handle = acquire(path)
                    stack.callback(handle.close)
                    break
                except BlockingIOError:
                    controller.save(wait_reason="Waiting for legacy lease: " + str(path), heartbeat_utc=now())
                    time.sleep(2)
        if not controller.state.get("migration_complete"):
            # The old owner might have resubmitted while the new process started.
            refreshed = inspect(root, kind)
            old = refreshed["state"]
            need(refreshed["request"]["info"]["identity"] == controller.plan["acceptance"]["identity"],
                 "Legacy accepted inputs changed during adoption")
            controller.save(active=old["single"], old_candidate=old.get("dual"), migration_complete=True)
            controller.event("OWNERSHIP_ADOPTED", single=old["single"])
        controller.save(controller_status="RUNNING", wait_reason=None)
        stopped = [False]
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
        try:
            while not stopped[0]:
                for extension in (p["control"] / "extensions").glob("*.json"):
                    row = read(extension)
                    if row["target"] > controller.state["deadline_utc"]:
                        controller.save(deadline_utc=row["target"], monitoring_finished=False,
                                        phase="ACTIVE" if controller.state.get("promoted") else "SINGLE_ONLY"
                                        if controller.state.get("handoff_failed") else controller.state["phase"])
                admitted = p["control"] / "admit.json"
                if admitted.exists():
                    need(read(admitted)["operation"] == controller.state["operation"], "Wrong admission operation")
                    controller.state["candidate_admitted"] = True
                try:
                    controller.tick()
                    controller.save(last_error=None, heartbeat_utc=now())
                except (Unknown, BlockingIOError, OSError) as e:
                    controller.save(last_error=str(e), heartbeat_utc=now())
                except Exception as e:
                    # Invalid candidate evidence stops only THIS transition, while
                    # the original single owner remains alive and visible.
                    controller.save(last_error=type(e).__name__ + ": " + str(e), heartbeat_utc=now())
                    controller.event("ERROR", error=str(e))
                    if not controller.state.get("promoted") and controller.state.get("candidate"):
                        controller.fail_candidate("Qualification/ownership error: " + str(e))
                monitor = read(p["monitor"] / "state.json")
                monitor.update(status="RUNNING", managed_by="d4_training_continuation", pid=os.getpid(),
                               host=socket.gethostname(), deadline_utc=controller.state["deadline_utc"],
                               job_id=(controller.state.get("active") or {}).get("job_id"),
                               continuation_state=str(p["control"] / "state.json"))
                write(p["monitor"] / "state.json", monitor)
                if controller.state.get("monitoring_finished"):
                    break
                # Stop/extension files are observed promptly without fork-heavy polling.
                for _ in range(15):
                    if stopped[0]:
                        break
                    if not controller.state.get("candidate_admitted") and (p["control"] / "admit.json").exists():
                        break
                    time.sleep(2)
        finally:
            controller.save(controller_status="STOPPED", stopped_utc=now())
            monitor = read(p["monitor"] / "state.json")
            monitor.update(status="STOPPED", stop_reason="continuation_owner_stopped; use new CLI resume")
            write(p["monitor"] / "state.json", monitor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "apply", "status", "resume", "extend", "repair-cancellation", "_run", "self-test"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--operation")
    parser.add_argument("--extra-hours", type=float, default=24)
    parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args(argv)
    if args.action == "_run":
        need(args.kind is not None, "Missing setup")
        return daemon(args.root, args.kind)
    if args.action == "self-test":
        import unittest
        root = Path(__file__).resolve().parents[1]
        suite = unittest.defaultTestLoader.discover(str(root / "tests/multidomain"), pattern="test_training_continuation*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() and result.testsRun else 1)
    if args.action in ("apply", "extend"):
        need(args.operation, "A stable --operation ID is required")
        result = globals()[args.action](args.root, args.operation, args.extra_hours)
    elif args.action == "repair-cancellation":
        need(args.kind is not None, "repair-cancellation requires --kind random or --kind domain")
        from .cancellation_repair import repair
        result = repair(args.root, args.kind)
    elif args.action == "resume":
        area = args.root / "outputs/multidomain/continuation_operations"
        with locked(area / "apply.lock"):
            for kind in KINDS:
                spawn(args.root, kind)
        result = status(args.root)
    else:
        result = globals()[args.action](args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
