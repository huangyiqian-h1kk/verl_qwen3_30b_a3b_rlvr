"""Install opportunistic monitoring in new frozen templates; keep live jobs intact."""
import os
from pathlib import Path
import shutil
import socket
import time
import uuid
from . import VERSION, process_signal
from .common import active, below, digest, files, locked, need, now, paths, read, safe_id, sha, verify_files, write
from .cancellation_repair import stop_and_acquire
from .opportunities import POLICY

BASELINE = {'__init__.py': 'a107d95b407454e8de1369501be73da50dd14d8240c9c5b83edd86244ed5871b',
 'cancellation_repair.py': 'dfb3e732a7f74ff9b443bd96eb400faa733b7843fa0a9f93524399015ee93c03',
 'capacity.py': '33ffd402d20d3c5e5ef04dcd4df174aa9537a2fc2209e5cf38081720b9ec9e51',
 'checkpoints.py': '27098ac767b36d8abc409ec050cf4e7be2f6a99b8085f3394b671d58384cbf90',
 'cli.py': 'b8ede01948994e5bdbc00ddfaacd48779715db43fc691a239720b9be92b4107b',
 'common.py': 'a25a3c738cae4a142f3e1b32875c3b8c49c6324b7b49620bcb3727ad5f78d0d9',
 'controller.py': 'f6944479ced5f0bb0eaa39834446714a2f3ca775d7ffed8f8b00021dfcfa1ff0',
 'evidence.py': '59d54b9c9c200003531c0b22903f523a525738b869db9e194d3ccccfda9f5fc1',
 'optimizer_repair.py': '2ae0cd2b4171c1961fe61785ac1b6af8e2838bc8253c768b251050bf2acd786b',
 'pinned_audit.py': '568f7b096d988f2f9b3b05091c578e0962a13202fca2992f2f58b2c12b82de1a',
 'process_signal.py': '73092baa6cb732f1f813dd86fc138f0aa411f228946e2f4c3914cd575642460c',
 'runtime.py': 'c54b59ec059631b5b3169ac5ec8c729bda974426085abda3dab769aee1a52ca1',
 'scheduler.py': 'd6711a759b413892eeacee0acc42bfd1ba2e6198e7fe2a623065ed9150f7c5c8',
 'source.py': '1738c0756d32f0777afd5f8fb4c288ac063bd58b9035aed6470b88668f8e958b',
 'trainer.py': 'dbab5ba75d8353f91503b1608e57447c1534e5986403b5ae352524e1a3cc77e1',
 'worker.py': '4d10cd88fa55b60a852bbfe9b8cbbf16fa40f51c21eb2ef6755208eb19017ecc'}
PREFIX = "runtime/d4_training_continuation/"
LEGACY = {"cli.py": {"27fdf7f11fe7fabefa00b9be42a8ef11e45ab36be416e87cf64c76afbe97acdb",
                     "b6a7e28dbfeb20e59ae2e0fea0c533c0f55f248ccd508a4a11bcb1347ac47d3b"},
          "scheduler.py": {"26d7baacdd5a3193d246028006ee31fc8172b114e916c7a7d113034da5f0dcad"}}


def prepare(root, kind):
    from .save_policy_upgrade import PREVIOUS
    from .validation_upgrade import PUBLISHED_SAVE5
    p = paths(root, kind)
    state = read(p["control"] / "state.json")
    need(state["version"] == VERSION and state.get("migration_complete"), "Expected adopted continuation owner")
    safe_id(state["operation"])
    operation = p["control"] / "operations" / state["operation"]
    planfile = operation / "plan.json"
    original = read(planfile)
    need(original["operation"] == state["operation"] and original["kind"] == kind
         and Path(original["root"]).resolve() == p["root"], "Wrong continuation plan")
    template = below(original["template"], operation)
    verify_files(template, original["template_files"])
    sources = {PREFIX + f.name: f for f in Path(__file__).parent.glob("*.py")}
    replacement = {name: sha(file) for name, file in sources.items()}
    for relative, checksum in original["template_files"].items():
        if not relative.startswith(PREFIX):
            continue
        name = relative[len(PREFIX):]
        allowed = {BASELINE.get(name), PREVIOUS.get(name), PUBLISHED_SAVE5.get(name), replacement.get(relative)} | LEGACY.get(name, set())
        need(checksum in allowed and checksum is not None and relative in sources,
             "Unsupported frozen continuation source: " + relative)
    manifest = dict(original["template_files"], **replacement)
    if original.get("opportunistic") == POLICY and original["template_files"] == manifest:
        return planfile, original, original
    need(state["phase"] in ("ACTIVE", "SINGLE_ONLY", "EXPIRED", "DONE"), "Finish the in-flight initial handoff before upgrading")
    need(not state.get("pending") and not state.get("rollback_required"), "Resolve existing submission/rollback first")
    area = operation / "upgrades" / ("opportunities-" + digest(dict(files=manifest, policy=POLICY))[:16])
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
            need(files(temporary) == manifest, "Upgrade changed the training/reward snapshot unexpectedly")
            verify_files(temporary, manifest)
            os.rename(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    need(files(target) == manifest, "Staged opportunity runtime differs")
    revised = dict(original, template=str(target), template_files=manifest, opportunistic=dict(POLICY))
    return planfile, original, revised


def enable(root):
    from .cli import KINDS, runtime_status, spawn
    area = Path(root).resolve() / "outputs/multidomain/continuation_operations"
    receipts = {}
    with locked(area / "apply.lock"):
        prepared = {kind: prepare(root, kind) for kind in KINDS}
        # Verify both host identities before stopping either owner. No remote PID signalling.
        for kind in KINDS:
            p = paths(root, kind)
            state = read(p["control"] / "state.json")
            if active(p["control"] / "controller.lock"):
                need(state.get("host", "").split(".")[0] == socket.gethostname().split(".")[0],
                     "Run enable-opportunistic on controller host " + str(state.get("host")))
        capability = process_signal.probe()
        for kind in KINDS:
            p = paths(root, kind)
            planfile, original, revised = prepared[kind]
            before = runtime_status(root, kind)
            restart = original != revised or not before["verified"]
            if restart:
                try:
                    with stop_and_acquire(p["control"], p["root"], kind):
                        need(read(planfile) == original, "Plan changed during upgrade")
                        # The old owner may have progressed during staging; never replace an in-flight transaction.
                        state = read(p["control"] / "state.json")
                        need(state["phase"] in ("ACTIVE", "SINGLE_ONLY", "EXPIRED", "DONE")
                             and not state.get("pending") and not state.get("rollback_required"),
                             "Owner entered a handoff or submission during upgrade; rerun after it settles")
                        write(p["control"] / "opportunity_upgrade.before.json",
                              dict(plan=original, state=state, at_utc=now(), process_signal=capability))
                        write(planfile, revised)
                finally:
                    # If publishing failed, this restarts the still-current original plan.
                    spawn(p["root"], kind)
            receipts[kind] = dict(template_updated=original != revised, controller_restart_requested=restart)
    until = time.monotonic() + 15
    while True:
        for kind in KINDS:
            p = paths(root, kind)
            state = read(p["control"] / "state.json")
            receipts[kind].update(controller_runtime=runtime_status(root, kind, state),
                                  active=state.get("active"), candidate=state.get("candidate"),
                                  deadline_utc=state["deadline_utc"], phase=state["phase"],
                                  opportunity=state.get("opportunity"), opportunity_poll=state.get("opportunity_poll"))
        if all(r["controller_runtime"]["verified"] and r["controller_runtime"].get("first_check_completed")
               for r in receipts.values()) or time.monotonic() >= until:
            break
        time.sleep(.25)
    verified = all(r["controller_runtime"]["verified"] and r["controller_runtime"].get("first_check_completed")
                   for r in receipts.values())
    return dict(status="OPPORTUNISTIC_MONITORS_RUNNING" if verified else "OPPORTUNISTIC_MONITORS_UNVERIFIED",
                policy=POLICY, deadlines_extended=False, setups=receipts)
