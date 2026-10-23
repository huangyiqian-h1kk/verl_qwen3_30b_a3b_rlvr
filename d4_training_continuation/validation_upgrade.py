"""Deploy durable validation, save5, opportunities and one idempotent +24h extension."""
from contextlib import ExitStack
from pathlib import Path
import socket
import time
from . import VERSION
from .common import active, extended_deadline, locked, need, now, paths, read, safe_id, write


def stable(state):
    from .opportunities import IN_FLIGHT
    need(state.get("version") == VERSION and state.get("migration_complete"), "Expected adopted continuation owner")
    need(state["phase"] in ("ACTIVE", "SINGLE_ONLY", "EXPIRED", "DONE")
         and not state.get("pending") and not state.get("rollback_required")
         and (state.get("opportunity") or {}).get("phase") not in IN_FLIGHT,
         "An existing handoff/submission must settle before this runtime upgrade")


def restart_target(state, kind):
    """Choose a FUTURE checkpoint boundary; never hard-code old step 125."""
    if kind != "random" or state.get("training_done") or not state.get("active"):
        return None
    request = read(Path(state["active"]["folder"]) / "request.json")
    if request["config"]["training"]["save_freq"] == 5:
        return None
    from .controller import authoritative_checkpoint
    record = authoritative_checkpoint(state["authority"], kind)
    progress = record["step"]
    for phase in ("train", "E"):
        for path in (Path(state["active"]["folder"]) / phase).glob("step_*.json"):
            row = read(path)
            if row.get("status") == "PASS" and row.get("actor_update_completed"):
                progress = max(progress, int(row["step"]))
    target = ((progress + 24) // 25) * 25
    return target if target < 468 else None


def report(root, ledger_path, ledger):
    from . import backfill
    from .cli import KINDS, runtime_status
    campaign = Path(ledger["campaign"])
    try:
        backfill.tick(campaign)
        ledger.pop("backfill_refresh_error", None)
    except BlockingIOError:
        # The already-running domain owner owns the refresh; this is success
        # of lease exclusion, not a failed deployment or a reason to qsub again.
        ledger.pop("backfill_refresh_error", None)
    except (RuntimeError, OSError) as error:
        ledger["backfill_refresh_error"] = str(error)
    until = time.monotonic() + 25
    while True:
        bindings = {k: runtime_status(root, k) for k in KINDS}
        verified = all(v["verified"] and v.get("first_check_completed") for v in bindings.values())
        if verified or time.monotonic() >= until:
            break
        time.sleep(.25)
    campaign_state = read(campaign / "state.json")
    success = verified and (campaign_state["phase"] == "PASS" or
                            (campaign_state["phase"] == "QUEUED" and campaign_state.get("job")))
    ledger.update(status="RUNNING" if success else "NEEDS_RETRY",
                  runtime=bindings, backfill=campaign_state, updated_utc=now())
    write(ledger_path, ledger)
    return ledger


def deploy(root, operation, hours=24):
    from . import backfill
    from .cancellation_repair import stop_and_acquire
    from .checkpoint_restart import arm
    from .cli import KINDS, runtime_status, spawn
    from .opportunity_upgrade import prepare
    from .save_policy import POLICY as TRAINING
    from .validation_journal import POLICY as VALIDATION
    from .validation_journal import check_source
    safe_id(operation)
    root = Path(root).resolve()
    area = root / "outputs/multidomain/continuation_operations"
    ledger_path = area / (operation + ".validation_repair.json")
    with locked(area / "apply.lock"):
        # This MUST precede reactivating expired owners and their source cleanup.
        sources = backfill.protect(root)
        prepared = {k: prepare(root, k) for k in KINDS}
        for _, _, revised in prepared.values():
            check_source(Path(revised["old_request"]["config"]["verl_src"]) / "verl/trainer/ppo/ray_trainer.py")
        ledger = read(ledger_path) if ledger_path.exists() else None
        if ledger:
            need(ledger["operation"] == operation and ledger["extra_hours"] == hours, "Repair operation changed")
            deployed = all(original == revised and original.get("validation_policy") == VALIDATION
                and original.get("training_policy") == TRAINING
                and read(paths(root, k)["control"] / "state.json").get("validation_repair_operation") == operation
                and runtime_status(root, k)["verified"]
                for k, (_, original, revised) in prepared.items())
            if deployed and ledger.get("campaign"):
                # A successful deployment may be inspected/repeated from ANY
                # login host; do not signal or restart an already-attested owner.
                return report(root, ledger_path, ledger)
        for kind in KINDS:
            p = paths(root, kind)
            state = read(p["control"] / "state.json")
            stable(state)
            if active(p["control"] / "controller.lock"):
                need(state.get("host", "").split(".")[0] == socket.gethostname().split(".")[0],
                     "A live owner still holds the lease on " + str(state.get("host"))
                     + "; this process cannot signal a remote PID. Expired owners can be adopted on any login node.")
        if ledger:
            need(ledger["operation"] == operation and ledger["extra_hours"] == hours, "Repair operation changed")
        else:
            ledger = dict(operation=operation, extra_hours=hours, created_utc=now(), targets={}, previous={})
            for kind in KINDS:
                state = read(paths(root, kind)["control"] / "state.json")
                ledger["previous"][kind] = state["deadline_utc"]
                ledger["targets"][kind] = extended_deadline(state["deadline_utc"], hours)
            write(ledger_path, ledger)
        with ExitStack() as stack:
            # Even a partial publish restarts the currently durable, verified
            # plan after releasing its lease. The operation ledger fixes targets
            # so a subsequent retry completes the same transaction.
            for kind in reversed(KINDS):
                stack.callback(spawn, root, kind)
            for kind in KINDS:
                p = paths(root, kind)
                stack.enter_context(stop_and_acquire(p["control"], root, kind))
            revised_plans = {}
            for kind in KINDS:
                p = paths(root, kind)
                planfile, original, revised = prepared[kind]
                need(read(planfile) == original, "Plan changed during validation upgrade")
                state = read(p["control"] / "state.json")
                stable(state)
                before = area / operation / (kind + ".before.json")
                if not before.exists():
                    write(before, dict(plan=original, state=state, at_utc=now()))
                revised = dict(revised, training_policy=dict(TRAINING), validation_policy=dict(VALIDATION))
                revised_plans[kind] = revised
            campaign = backfill.prepare(root, operation, revised_plans, sources, min(ledger["targets"].values()))
            for kind in KINDS:
                p = paths(root, kind)
                state = read(p["control"] / "state.json")
                planfile, _, _ = prepared[kind]
                revised = dict(revised_plans[kind], validation_backfill=str(campaign))
                write(planfile, revised)
                target = max(state["deadline_utc"], ledger["targets"][kind])
                state.update(deadline_utc=target, monitoring_finished=False,
                             validation_repair_operation=operation)
                if state["phase"] == "EXPIRED":
                    state["phase"] = "ACTIVE" if state.get("promoted") else "SINGLE_ONLY"
                step = restart_target(state, kind)
                if step is not None and not state.get("checkpoint_restart"):
                    state["checkpoint_restart"] = arm(state, step)
                write(p["control"] / "state.json", state)
                # Same targets also make the normal owner extension scanner safe.
                write(p["control"] / "extensions" / (operation + ".json"), dict(
                    operation=operation, hours=hours, previous=ledger["previous"][kind],
                    target=ledger["targets"][kind], at_utc=ledger["created_utc"]))
            ledger.update(phase="DEPLOYED", campaign=str(campaign), protected_sources=sources, updated_utc=now())
            write(ledger_path, ledger)
    return report(root, ledger_path, ledger)

PUBLISHED_SAVE5 = {'__init__.py': 'a107d95b407454e8de1369501be73da50dd14d8240c9c5b83edd86244ed5871b',
 'cancellation_repair.py': 'dfb3e732a7f74ff9b443bd96eb400faa733b7843fa0a9f93524399015ee93c03',
 'capacity.py': '33ffd402d20d3c5e5ef04dcd4df174aa9537a2fc2209e5cf38081720b9ec9e51',
 'checkpoint_restart.py': '4bc665b20f5ec85281c86b9b18f139b6eb6a2315dd05191a4090aaaadb3ed50c',
 'checkpoints.py': '3ac6bfe482417c9099bca333acd5ddc688d86936b3c45659db692e74b14ef0f5',
 'cli.py': '7ef7724f8391d1b6b9339e2b8c4b9f56fa87cbbae1e3a41c68dbd65a7a47dec4',
 'common.py': 'a25a3c738cae4a142f3e1b32875c3b8c49c6324b7b49620bcb3727ad5f78d0d9',
 'controller.py': '422eb91acd5bf2a518f8b4d79928058e9b81e96ff8a3089617dd4da6a79be264',
 'evaluation.py': '6677735b230414c0c5c25a7d95613104cd1c7a9b63f9efa0b7ab529e69ced623',
 'evidence.py': '59d54b9c9c200003531c0b22903f523a525738b869db9e194d3ccccfda9f5fc1',
 'opportunities.py': '0c57ef3db64f9785ff4c4402678f83ba6b66ed2c9ffb252482ca64ec9a755bfb',
 'opportunity_upgrade.py': '5393509acc9ab0c5be9182d2826949fd02876340edd80a9fc04c4f665323406e',
 'optimizer_repair.py': '2ae0cd2b4171c1961fe61785ac1b6af8e2838bc8253c768b251050bf2acd786b',
 'pinned_audit.py': '568f7b096d988f2f9b3b05091c578e0962a13202fca2992f2f58b2c12b82de1a',
 'process_signal.py': '73092baa6cb732f1f813dd86fc138f0aa411f228946e2f4c3914cd575642460c',
 'runtime.py': 'a3f2e24ca739ff35127eee3bd2c4062e3a9e16ac645c262a0ee41526be410f28',
 'save_policy.py': '6077e79e9b73f6f595586ec1426c90c5bc75f7e2c675a347772c7ed79121aa2d',
 'save_policy_upgrade.py': '10861e9071d42ccb29b5cdc484d58a7717430bf8d019be5a993ba13fe9224901',
 'scheduler.py': 'd6711a759b413892eeacee0acc42bfd1ba2e6198e7fe2a623065ed9150f7c5c8',
 'source.py': 'deee45d8b852c45bacefcfa55985885b8cdd43ee2ad30f3fbad72c7123671e03',
 'trainer.py': '33e033423fddc31f45b45ede0acf49cb37efcdf446eb784642eaf97431681158',
 'worker.py': '4d10cd88fa55b60a852bbfe9b8cbbf16fa40f51c21eb2ef6755208eb19017ecc'}
