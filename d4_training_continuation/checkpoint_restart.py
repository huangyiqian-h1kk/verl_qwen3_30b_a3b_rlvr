"""Optional, one-shot restart after an immutable full checkpoint is protected."""
from pathlib import Path
from . import checkpoints as ck
from .common import expired, locked, need, now, read, write
from .save_policy import POLICY
from .scheduler import finished

DONE = {"COMPLETE", "NATURAL_EXIT", "SUPERSEDED", "EXPIRED"}


def arm(state, step):
    need(isinstance(step, int) and 0 < step < 468, "Invalid restart checkpoint step")
    existing = state.get("checkpoint_restart")
    if existing:
        need(existing["minimum_step"] == step, "A different restart is already recorded")
        return existing
    need(not expired(state["deadline_utc"]), "Monitor deadline already expired")
    need(state.get("active") and state.get("promoted") and state["phase"] == "ACTIVE",
         "Checkpoint restart requires an already promoted active random run")
    current = read(Path(state["active"]["folder"]) / "request.json")
    need(current.get("lineage") == state["authority"].get("lineage"), "Active restart lineage mismatch")
    save_freq = current.get("config", {}).get("training", {}).get("save_freq")
    need(save_freq in (5, 25), "Unrecognized active save interval")
    return dict(version=POLICY["version"], phase="SUPERSEDED" if save_freq == 5 else "WAIT_CHECKPOINT",
                reason="active_job_already_save5" if save_freq == 5 else "restore_save5",
                minimum_step=step, job=state["active"], authority=state["authority"], created_utc=now())


def tick(controller):
    c, s = controller, controller.state
    item = s.get("checkpoint_restart")
    if item and item["phase"] == "COMPLETE" and not item.get("pins_retired"):
        # A verified later save proves that the replacement restored and advanced.
        # Failed/queued replacements keep the protected recovery copies.
        try:
            from .controller import authoritative_checkpoint
            latest = authoritative_checkpoint(s["authority"], c.kind)
            if latest["lineage"] == item["resume"]["lineage"] and latest["step"] > item["resume"]["step"]:
                area = c.control / "repairs/save5-eval25/restart"
                for name in ("before_cancel", "resume"):
                    ck.remove_pin(area / name)
                old = Path(item["job"]["folder"])
                if old.parent == c.p["run"] / "jobs":
                    for name in ("A", "B", "C", "D", "E", "train"):
                        ck.remove_pin(old / name / "source")
                item.update(pins_retired=True, replacement_checkpoint=latest["step"])
                c.save(checkpoint_restart=item)
        except (OSError, RuntimeError) as error:
            c.save(cleanup_error="Persistence restart pins: " + str(error))
    if not item or item["phase"] in DONE:
        return False
    need(c.kind == "random" and c.plan.get("training_policy") == POLICY, "Unbound checkpoint restart")
    need(not s.get("rollback_required"), "Another rollback owns the writer")
    area = c.control / "repairs" / "save5-eval25" / "restart"
    journal = area / "qdel.json"

    def save(**fields):
        item.update(fields)
        c.save(checkpoint_restart=item)

    if s.get("active") != item["job"]:
        # submit() publishes the new active job before the final restart receipt.
        # Recover that crash without ever touching the replacement job.
        replacement = s.get("active")
        request = read(Path(replacement["folder"]) / "request.json") if replacement else {}
        origin = request.get("source", {}).get("record", {})
        expected = item.get("resume") or {}
        accepted = (expected and origin.get("step") == expected.get("step")
                    and origin.get("files") == expected.get("files")
                    and origin.get("lineage") == expected.get("lineage"))
        save(phase="COMPLETE" if accepted else "SUPERSEDED", replacement=replacement, finished_utc=now())
        return bool(accepted)
    need(s["authority"] == item["authority"], "Checkpoint authority changed during restart")
    row = c.pbs.observe(item["job"])
    if finished(row) and not journal.exists():
        save(phase="NATURAL_EXIT", observation=row, finished_utc=now())
        return False  # Normal policy still rejects application errors/unknown exit codes.
    if expired(s["deadline_utc"]) and not journal.exists():
        save(phase="EXPIRED", finished_utc=now())
        return False  # Never cancel a running allocation with no continuation window.
    if item["phase"] == "WAIT_CHECKPOINT":
        if row.get("job_state") != "R":
            return False
        from .controller import authoritative_checkpoint
        record = authoritative_checkpoint(s["authority"], c.kind)
        if record["step"] < item["minimum_step"]:
            save(observed_checkpoint_step=record["step"], checked_utc=now())
            return False
        if c.plan.get("validation_policy"):
            from .evaluation import due, import_legacy, completed
            request = read(Path(item["job"]["folder"]) / "request.json")
            if due(record["step"], request["config"]["training"]["test_freq"]):
                done = (completed(request, record) is not None if request.get("validation_policy")
                        else import_legacy(item["job"], record, probe=True))
                if not done:
                    save(observed_checkpoint_step=record["step"], wait_reason="scheduled_validation_in_progress",
                         checked_utc=now())
                    return False
        need(record["step"] < 468, "Final checkpoint should finish its final evaluation")
        save(phase="PINNING", checkpoint=record, wait_reason=None)
    if item["phase"] == "PINNING":
        protected = ck.pin(item["checkpoint"], area / "before_cancel")
        save(phase="CANCELLING", protected=protected)
    if item["phase"] == "CANCELLING":
        ck.validate(item["protected"])
        if not c.pbs.cancel(item["job"], journal):
            return True
        # If the application ended between our observations without qdel ever
        # being attempted, leave the failure to the ordinary retry classifier.
        if not journal.exists():
            save(phase="NATURAL_EXIT", finished_utc=now())
            return False
        need(read(journal)["job"] == item["job"], "Foreign cancellation journal")
        with locked(Path(item["authority"]["run"]) / "stage.lock"):
            from .controller import authoritative_checkpoint
            from .evaluation import import_legacy
            record = authoritative_checkpoint(item["authority"], c.kind)
            need(record["step"] >= item["protected"]["step"], "Final checkpoint regressed")
            import_legacy(item["job"], record)
            protected = ck.pin(record, area / "resume")
        save(phase="SUBMITTING", resume=protected)
    if item["phase"] == "SUBMITTING":
        from .controller import choose_nodes
        if "nodes" not in item:
            capacity = c.pbs.capacity()
            nodes, reason = choose_nodes(s["promoted"], capacity)
            save(nodes=nodes, capacity=capacity, reason=reason)
        c.save(retry_selection=dict(nodes=item["nodes"], reason=item["reason"],
               capacity=item["capacity"], controlled_restart=True, at_utc=now()))
        source = dict(record=item["resume"], lineage=item["authority"]["lineage"])
        replacement = c.submit("formal", item["nodes"], source)
        save(phase="COMPLETE", replacement=replacement, finished_utc=now())
        c.event("CHECKPOINT_POLICY_RESTARTED", previous=item["job"], replacement=replacement,
                resume_step=item["resume"]["step"], nodes=item["nodes"])
    return True
