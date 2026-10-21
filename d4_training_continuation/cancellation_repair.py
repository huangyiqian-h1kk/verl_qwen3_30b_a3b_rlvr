"""Install the PBS cancellation and controller-launch fixes into a frozen owner.

The original template and every submitted job remain immutable. A replacement
template changes scheduler.py and cli.py only. Publishing its plan pointer happens while
the old controller is stopped and its ownership lock is held by this process.
"""
import os
from pathlib import Path
import shutil
import signal
import socket
import time
import uuid

from . import VERSION
from . import process_signal
from .common import acquire, atomic_text, below, files, locked, need, now, paths, read, safe_id, sha, verify_files, write

REPAIR = "pbs-finished-without-exit-v2"
SCHEDULER = "runtime/d4_training_continuation/scheduler.py"
CLI = "runtime/d4_training_continuation/cli.py"
ORIGINAL_SHA = "26d7baacdd5a3193d246028006ee31fc8172b114e916c7a7d113034da5f0dcad"
ORIGINAL_CLI_SHAS = {
    # Frozen initial owner (ad5bef7), then the 919c045/15ca818 operations CLI.
    "27fdf7f11fe7fabefa00b9be42a8ef11e45ab36be416e87cf64c76afbe97acdb",
    "b6a7e28dbfeb20e59ae2e0fea0c533c0f55f248ccd508a4a11bcb1347ac47d3b",
}


def prepare_template(root, kind):
    """Verify and stage before stopping an owner; never edit its old template."""
    p = paths(root, kind)
    state = read(p["control"] / "state.json")
    need(state["version"] == VERSION and state.get("migration_complete"),
         "Cancellation repair requires an adopted continuation owner")
    safe_id(state["operation"])
    operation = p["control"] / "operations" / state["operation"]
    planfile = operation / "plan.json"
    plan = read(planfile)
    need(plan["operation"] == state["operation"] and plan["kind"] == kind
         and Path(plan["root"]).resolve() == p["root"], "Wrong continuation plan")
    template = below(plan["template"], operation)
    verify_files(template, plan["template_files"])
    replacements = {SCHEDULER: Path(__file__).with_name("scheduler.py"),
                    CLI: Path(__file__).with_name("cli.py")}
    revised_hashes = {name: sha(source) for name, source in replacements.items()}
    allowed = {SCHEDULER: {ORIGINAL_SHA}, CLI: ORIGINAL_CLI_SHAS}
    for name, new_sha in revised_hashes.items():
        need(plan["template_files"].get(name) in allowed[name] | {new_sha},
             "Unsupported frozen " + Path(name).stem + " source; no automatic replacement")
    if all(plan["template_files"].get(name) == checksum for name, checksum in revised_hashes.items()):
        return planfile, plan, plan
    manifest = dict(plan["template_files"], **revised_hashes)
    repair = operation / "repairs" / REPAIR
    target = repair / "template"
    repair.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = repair / (".staging_" + uuid.uuid4().hex)
        try:
            shutil.copytree(template, temporary, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name, source in replacements.items():
                atomic_text(temporary / name, source.read_text())
            need(files(temporary) == manifest, "Repair changed files other than scheduler.py and cli.py")
            verify_files(temporary, manifest)
            os.rename(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    need(files(target) == manifest, "Existing repaired template differs from this repair")
    verify_files(target, manifest)
    before = repair / "plan.before.json"
    if before.exists():
        need(read(before) == plan, "Repair baseline plan changed")
    else:
        write(before, plan)
    revised = dict(plan, template=str(target), template_files=manifest)
    return planfile, plan, revised


def _process_identity(pid):
    proc = Path("/proc") / str(pid)
    return proc.stat().st_uid, [os.fsdecode(x) for x in (proc / "cmdline").read_bytes().split(b"\0") if x]


def _signal_owner(state, entry, root, kind):
    """Signal only this user's exact local controller; pidfd prevents PID reuse."""
    host = str(state.get("host", ""))
    need(host.split(".")[0] == socket.gethostname().split(".")[0],
         "Run repair on controller host " + host + "; no remote PID will be signalled")
    pid = state.get("pid")
    need(isinstance(pid, int) and pid > 1 and pid != os.getpid(), "Invalid controller PID")
    try:
        descriptor = process_signal.open_process(pid)
    except ProcessLookupError:
        return
    try:
        try:
            uid, argv = _process_identity(pid)
        except FileNotFoundError:
            return  # The controller exited; the caller still needs its lease.
        need(uid == os.getuid(), "Controller process owner mismatch")
        expected = ["-I", str(entry), "_run", "--root", str(root), "--kind", kind]
        need(argv[1:] == expected, "Controller command line mismatch; no signal sent")
        try:
            process_signal.send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            pass
    finally:
        os.close(descriptor)


def stop_and_acquire(control, root, kind, timeout=45):
    """Return the exclusive lease after graceful shutdown; never send SIGKILL."""
    lock = control / "controller.lock"
    try:
        return acquire(lock)
    except BlockingIOError:
        pass
    state = read(control / "state.json")
    safe_id(state["operation"])
    entry = control / "operations" / state["operation"] / "controller_entry.py"
    _signal_owner(state, entry, root, kind)
    until = time.monotonic() + timeout
    while True:
        try:
            return acquire(lock)
        except BlockingIOError:
            need(time.monotonic() < until,
                 "Controller is still releasing its lease; rerun the same repair command")
            time.sleep(0.25)


def repair(root, kind):
    from .cli import runtime_status, spawn
    p = paths(root, kind)
    area = p["root"] / "outputs/multidomain/continuation_operations"
    with locked(area / "apply.lock"):
        planfile, original, revised = prepare_template(root, kind)
        changed = original != revised
        binding_before = runtime_status(root, kind)
        restart_needed = changed or not binding_before["verified"]
        if restart_needed:
            # A missing Python wrapper is supported. A missing kernel capability
            # or denied permission still stops here, before signalling any owner.
            capability = process_signal.probe()
            with stop_and_acquire(p["control"], p["root"], kind):
                # Other deploy/resume commands use apply.lock; a raced plan edit
                # is still rejected rather than overwritten.
                need(read(planfile) == original, "Plan changed during controller shutdown")
                verify_files(revised["template"], revised["template_files"])
                write(planfile, revised)  # Single atomic pointer publication.
                write(p["control"] / "repairs" / (REPAIR + ".json"), dict(
                    repair=REPAIR, at_utc=now(), operation=revised["operation"],
                    original_template=original["template"], template=revised["template"],
                    scheduler_sha256=revised["template_files"][SCHEDULER],
                    cli_sha256=revised["template_files"][CLI],
                    previous_runtime=binding_before,
                    process_signal=capability,
                    deadline_utc=read(p["control"] / "state.json")["deadline_utc"],
                ))
            # Also covers an already updated plan with an old-code owner, and
            # an interrupted run after publishing the new plan pointer.
            spawn(p["root"], kind)
    until = time.monotonic() + 15
    running = False
    while True:
        state = read(p["control"] / "state.json")
        binding = runtime_status(root, kind, state)
        running = binding["verified"]
        if running or time.monotonic() >= until:
            break
        time.sleep(0.25)
    return dict(repair=REPAIR, kind=kind, template_updated=changed,
                controller_restart_requested=restart_needed, controller_runtime=binding,
                status="REPAIRED_CONTROLLER_RUNNING" if running else "REPAIRED_CONTROLLER_UNVERIFIED",
                scheduler_sha256=revised["template_files"][SCHEDULER],
                deadline_utc=state["deadline_utc"], phase=state["phase"],
                promoted=state.get("promoted"), handoff_failed=state.get("handoff_failed"),
                old_candidate_retired=state.get("old_candidate_retired"),
                active=state.get("active"), candidate=state.get("candidate"),
                last_error=state.get("last_error"))
