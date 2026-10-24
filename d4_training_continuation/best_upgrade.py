"""Stage native archive hooks; bridge old jobs without killing a PBS allocation.

A different login host's PID is NEVER signalled. Its bridge is active immediately;
native template adoption waits for a free owner lease or a same-host install.
"""
from pathlib import Path
import shutil
import socket
import uuid

from .best_checkpoints import POLICY, area
from .common import active, below, digest, expired, files, locked, need, now, paths, read, sha, verify_files, write

PREFIX = "runtime/d4_training_continuation/"
BASELINE = {
    "__init__.py": "a107d95b407454e8de1369501be73da50dd14d8240c9c5b83edd86244ed5871b",
    "backfill.py": "746f1513ac70938f64bef376d460f1cdfd2d8319b7dfa7a7dce49964c2e46eb8",
    "cancellation_repair.py": "dfb3e732a7f74ff9b443bd96eb400faa733b7843fa0a9f93524399015ee93c03",
    "capacity.py": "33ffd402d20d3c5e5ef04dcd4df174aa9537a2fc2209e5cf38081720b9ec9e51",
    "checkpoint_restart.py": "90a829a6da632c731ccf22866710993ef2bd057837bfeb890704b2346a0b6abd",
    "checkpoints.py": "3ac6bfe482417c9099bca333acd5ddc688d86936b3c45659db692e74b14ef0f5",
    "cli.py": "e48e6ddcf6a977e56be9ea9da936ca5e72fef984ed6cb3dedb21f00407599d00",
    "common.py": "a25a3c738cae4a142f3e1b32875c3b8c49c6324b7b49620bcb3727ad5f78d0d9",
    "controller.py": "422eb91acd5bf2a518f8b4d79928058e9b81e96ff8a3089617dd4da6a79be264",
    "evaluation.py": "192a526419399d602bdf5d752b99468d5580d4c34f99188f32f98c11f2470162",
    "evidence.py": "59d54b9c9c200003531c0b22903f523a525738b869db9e194d3ccccfda9f5fc1",
    "opportunities.py": "4a85105954418767d57151b6d303a85fc7b2b1cccb4f0a982cb5961906450810",
    "opportunity_upgrade.py": "b434e4d4eac82316933eb184775b5aa990d58f7f70a72ea63cfc5e52fb545f28",
    "optimizer_repair.py": "2ae0cd2b4171c1961fe61785ac1b6af8e2838bc8253c768b251050bf2acd786b",
    "pinned_audit.py": "568f7b096d988f2f9b3b05091c578e0962a13202fca2992f2f58b2c12b82de1a",
    "process_signal.py": "73092baa6cb732f1f813dd86fc138f0aa411f228946e2f4c3914cd575642460c",
    "runtime.py": "a90846603716a8cc4267f122cf8142f14732f6384403ddba6455e4defdff605b",
    "save_policy.py": "6077e79e9b73f6f595586ec1426c90c5bc75f7e2c675a347772c7ed79121aa2d",
    "save_policy_upgrade.py": "d14179d0a59483d3e942f2d0df8fc550b81cd86f5c35cd536b92c41db39e603d",
    "scheduler.py": "d6711a759b413892eeacee0acc42bfd1ba2e6198e7fe2a623065ed9150f7c5c8",
    "source.py": "cc06c8b3f35724bfa1a0cb867ab71683a4327f9e8ff4d94e80ec4009234bf61a",
    "trainer.py": "23ab0e38f4f1462b18bbd23a316d4f683b7f92bfdd66b693dfe570aa6bbc4c4d",
    "validation_journal.py": "6753300aca245f67d42b6acfa6739a394d11e8d6fe5a8b697b895149fe67b6da",
    "validation_upgrade.py": "9030c37cef022282548be73711a38a2f025b0da57ba6bcaf054a8800013bc2f4",
    "worker.py": "4d10cd88fa55b60a852bbfe9b8cbbf16fa40f51c21eb2ef6755208eb19017ecc"
}


def prepare(root, kind):
    p = paths(root, kind)
    state = read(p["control"] / "state.json")
    need(state.get("migration_complete"), "Continuation ownership has not been adopted")
    operation = p["control"] / "operations" / state["operation"]
    planfile = operation / "plan.json"
    plan = read(planfile)
    need(plan["kind"] == kind and Path(plan["root"]).resolve() == p["root"], "Wrong training plan")
    template = below(plan["template"], operation)
    verify_files(template, plan["template_files"])
    sources = {PREFIX + f.name: f for f in Path(__file__).parent.glob("*.py")}
    replacement = {name: sha(file) for name, file in sources.items()}
    for name, checksum in plan["template_files"].items():
        if name.startswith(PREFIX):
            need(checksum in (BASELINE.get(name[len(PREFIX):]), replacement.get(name)),
                 "Unsupported continuation source; do not overwrite: " + name)
    manifest = dict(plan["template_files"], **replacement)
    if plan.get("best_checkpoint_policy") == POLICY and plan["template_files"] == manifest:
        return planfile, plan, plan
    need(plan.get("validation_policy") and plan.get("training_policy"),
         "Install completed-validation/save5 policy before best-model selection")
    target = operation / "upgrades" / ("best-" + digest(manifest)[:16]) / "template"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.parent / (".staging_" + uuid.uuid4().hex)
        try:
            shutil.copytree(template, temp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name, source in sources.items():
                shutil.copy2(source, temp / name)
            need(files(temp) == manifest, "Archive upgrade changed unrelated frozen files")
            temp.rename(target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    verify_files(target, manifest)
    return planfile, plan, dict(plan, template=str(target), template_files=manifest, best_checkpoint_policy=dict(POLICY))


def deploy_kind(root, kind, signal_local=False):
    from .cancellation_repair import stop_and_acquire
    from .cli import spawn, runtime_status
    p = paths(root, kind)
    receipt = area(root, kind) / "deployment.json"
    with locked(p["root"] / "outputs/multidomain/continuation_operations/apply.lock"):
        planfile, original, revised = prepare(root, kind)
        state = read(p["control"] / "state.json")
        running = active(p["control"] / "controller.lock")
        binding = runtime_status(root, kind, state)
        report = dict(kind=kind, at_utc=now(), bridge="independent service", controller_host=state.get("host"),
                      deadline_utc=state["deadline_utc"], active=state.get("active"))
        if original == revised and (binding["verified"] or not running):
            report.update(status="NATIVE_PLAN_INSTALLED", controller_runtime=binding)
        elif running and (not signal_local or state.get("host", "").split(".")[0] != socket.gethostname().split(".")[0]):
            report.update(status="BRIDGE_ACTIVE_NATIVE_PENDING_OWNER", staged_template=revised["template"])
        elif state.get("pending") or state.get("rollback_required") or state["phase"] not in ("ACTIVE", "SINGLE_ONLY", "EXPIRED", "DONE"):
            report.update(status="BRIDGE_ACTIVE_NATIVE_PENDING_TRANSITION")
        else:
            # A stopped owner is safe to adopt from any login node. Local live
            # owners use the existing pidfd/command-line-checked graceful stop.
            try:
                with stop_and_acquire(p["control"], p["root"], kind):
                    current = read(p["control"] / "state.json")
                    need(read(planfile) == original, "Plan changed during native archive installation")
                    need(not current.get("pending") and not current.get("rollback_required")
                         and current["phase"] in ("ACTIVE", "SINGLE_ONLY", "EXPIRED", "DONE"),
                         "Owner entered a transition; keep bridge and retry native adoption")
                    before = Path(revised["template"]).parent / "before.json"
                    if not before.exists():
                        write(before, dict(plan=original, at_utc=now()))
                    write(planfile, revised)
            finally:
                # Do not extend expired training supervision as a side effect.
                if not expired(read(p["control"] / "state.json")["deadline_utc"]):
                    spawn(root, kind)
            report.update(status="NATIVE_PLAN_INSTALLED", controller_runtime=runtime_status(root, kind))
        # Existing GPU processes retain their frozen imports; report separately.
        job = (read(p["control"] / "state.json").get("active") or {})
        reqfile = Path(job.get("folder", "/nonexistent")) / "request.json"
        report["active_job_native_policy"] = read(reqfile).get("best_checkpoint_policy") if reqfile.exists() else None
        write(receipt, report)
        return report
