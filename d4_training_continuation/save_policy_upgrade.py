"""Upgrade only the random owner; live training snapshots remain immutable."""
import os
from pathlib import Path
import shutil
import socket
import time
import uuid
from . import VERSION, process_signal
from .common import active, below, digest, expired, files, locked, need, now, paths, read, safe_id, sha, verify_files, write
from .cancellation_repair import stop_and_acquire
from .save_policy import POLICY, training_config
from .checkpoint_restart import arm

PREFIX = "runtime/d4_training_continuation/"
PREVIOUS = {'__init__.py': 'a107d95b407454e8de1369501be73da50dd14d8240c9c5b83edd86244ed5871b',
 'cancellation_repair.py': 'dfb3e732a7f74ff9b443bd96eb400faa733b7843fa0a9f93524399015ee93c03',
 'capacity.py': '33ffd402d20d3c5e5ef04dcd4df174aa9537a2fc2209e5cf38081720b9ec9e51',
 'checkpoints.py': '3ac6bfe482417c9099bca333acd5ddc688d86936b3c45659db692e74b14ef0f5',
 'cli.py': 'c1d25d0db81e206fc3e5ea2967d0b0d49b5b320cf54e8df128871d69567e3568',
 'common.py': 'a25a3c738cae4a142f3e1b32875c3b8c49c6324b7b49620bcb3727ad5f78d0d9',
 'controller.py': 'f252c0ff8512cd22720f2d444d9d40287c0e6db66bd5e745f9105a096332372b',
 'evidence.py': '59d54b9c9c200003531c0b22903f523a525738b869db9e194d3ccccfda9f5fc1',
 'opportunities.py': '0c57ef3db64f9785ff4c4402678f83ba6b66ed2c9ffb252482ca64ec9a755bfb',
 'opportunity_upgrade.py': 'ab5c0bb36db602c5fd89d7c34dc11c8985b692d7413a372f9d89ca589b85e41b',
 'optimizer_repair.py': '2ae0cd2b4171c1961fe61785ac1b6af8e2838bc8253c768b251050bf2acd786b',
 'pinned_audit.py': '568f7b096d988f2f9b3b05091c578e0962a13202fca2992f2f58b2c12b82de1a',
 'process_signal.py': '73092baa6cb732f1f813dd86fc138f0aa411f228946e2f4c3914cd575642460c',
 'runtime.py': '2842bf2767fd5f94647d27addb121003b2fa1046aedd5145ef73b20162f901e7',
 'scheduler.py': 'd6711a759b413892eeacee0acc42bfd1ba2e6198e7fe2a623065ed9150f7c5c8',
 'source.py': 'b1cb230bd243fcde98ee952ff43ab42f4dba29558b719431acd3f93b419724a7',
 'trainer.py': 'f692f8705507a125b5269888314db2fc5c3b1ec1ae5b36372d47ea09f0d1a3bd',
 'worker.py': '4d10cd88fa55b60a852bbfe9b8cbbf16fa40f51c21eb2ef6755208eb19017ecc'}  # Published 0a0038b.


def stable(state):
    from .opportunities import IN_FLIGHT
    need(state.get("migration_complete") and state.get("phase") in ("ACTIVE", "SINGLE_ONLY"),
         "Finish the existing handoff before repairing persistence")
    need(not state.get("pending") and not state.get("rollback_required")
         and (state.get("opportunity") or {}).get("phase") not in IN_FLIGHT,
         "Resolve the in-flight submission/handoff before repairing persistence")
    need(not expired(state["deadline_utc"]), "Monitor deadline expired; explicitly extend monitoring first")


def prepare(root, kind="random"):
    need(kind == "random", "This deployed repair is scoped to random; domain already saves every 5 steps")
    from .opportunity_upgrade import BASELINE, LEGACY
    p = paths(root, kind)
    state = read(p["control"] / "state.json")
    need(state["version"] == VERSION, "Unsupported continuation owner version")
    stable(state)
    safe_id(state["operation"])
    operation = p["control"] / "operations" / state["operation"]
    planfile = operation / "plan.json"
    original = read(planfile)
    need(original["operation"] == state["operation"] and original["kind"] == kind
         and Path(original["root"]).resolve() == p["root"], "Wrong continuation plan")
    template = below(original["template"], operation)
    verify_files(template, original["template_files"])
    training_config(original["old_request"]["config"])
    sources = {PREFIX + p.name: p for p in Path(__file__).parent.glob("*.py")}
    replacement = {name: sha(path) for name, path in sources.items()}
    for relative, checksum in original["template_files"].items():
        if relative.startswith(PREFIX):
            name = relative[len(PREFIX):]
            allowed = {BASELINE.get(name), PREVIOUS.get(name), replacement.get(relative)} | LEGACY.get(name, set())
            need(checksum is not None and checksum in allowed and relative in sources,
                 "Unsupported frozen continuation source: " + relative)
    manifest = dict(original["template_files"], **replacement)
    if original.get("training_policy") == POLICY and original["template_files"] == manifest:
        return planfile, original, original
    area = operation / "repairs" / ("save5-eval25-" + digest(dict(files=manifest, policy=POLICY))[:16])
    target = area / "template"
    area.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = area / (".staging_" + uuid.uuid4().hex)
        try:
            shutil.copytree(template, temporary, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name, source in sources.items():
                destination = temporary / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            need(files(temporary) == manifest, "Persistence repair changed unrelated snapshot files")
            verify_files(temporary, manifest)
            os.rename(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    need(files(target) == manifest, "Staged persistence repair differs")
    revised = dict(original, template=str(target), template_files=manifest, training_policy=dict(POLICY))
    return planfile, original, revised


def repair(root, kind="random", restart_after_step=None):
    from .cli import runtime_status, spawn
    p = paths(root, kind)
    area = p["root"] / "outputs/multidomain/continuation_operations"
    with locked(area / "apply.lock"):
        planfile, original, revised = prepare(root, kind)
        state = read(p["control"] / "state.json")
        if active(p["control"] / "controller.lock"):
            need(state.get("host", "").split(".")[0] == socket.gethostname().split(".")[0],
                 "Run repair-save-policy on controller host " + str(state.get("host")))
        desired = arm(state, restart_after_step) if restart_after_step is not None else state.get("checkpoint_restart")
        before = runtime_status(root, kind)
        restart = original != revised or not before["verified"] or desired != state.get("checkpoint_restart")
        if restart:
            capability = process_signal.probe()
            try:
                with stop_and_acquire(p["control"], p["root"], kind):
                    need(read(planfile) == original, "Plan changed during repair")
                    state = read(p["control"] / "state.json")
                    stable(state)
                    desired = arm(state, restart_after_step) if restart_after_step is not None else state.get("checkpoint_restart")
                    # Immutable originals remain available; old_request is kept
                    # verbatim for provenance. Only new jobs normalize the policy.
                    receipt = p["control"] / "repairs/save5-eval25/upgrade.before.json"
                    if not receipt.exists():
                        write(receipt, dict(plan=original, state=state, at_utc=now(), process_signal=capability))
                    write(planfile, revised)
                    if desired is not None:
                        state["checkpoint_restart"] = desired
                        write(p["control"] / "state.json", state)
            finally:
                spawn(p["root"], kind)
    until = time.monotonic() + 20
    while True:
        state = read(p["control"] / "state.json")
        binding = runtime_status(root, kind, state)
        verified = binding["verified"] and binding.get("first_check_completed")
        if verified or time.monotonic() >= until:
            break
        time.sleep(.25)
    return dict(status="SAVE_POLICY_MONITOR_RUNNING" if verified else "SAVE_POLICY_MONITOR_UNVERIFIED",
                policy=POLICY, kind=kind, template_updated=original != revised,
                controller_restart_requested=restart, controller_runtime=binding,
                checkpoint_restart=state.get("checkpoint_restart"), active=state.get("active"),
                deadline_utc=state["deadline_utc"], deadlines_extended=False,
                current_training_hot_patched=False, last_error=state.get("last_error"))
