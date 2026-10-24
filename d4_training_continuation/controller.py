"""One independent owner per setup; a durable commit defines the handoff boundary."""
from pathlib import Path
import shutil
import uuid
from . import TOTAL_STEPS
from . import checkpoints as ck
from .common import digest, expired, locked, need, now, paths, read, sha, write
from .scheduler import PBS, Unknown, retryable, terminal
from .source import make_job


def authoritative_checkpoint(authority, kind):
    if authority.get("record"):
        return ck.validate(authority["record"])
    if authority.get("legacy"):
        return ck.latest_legacy(authority["run"], kind)
    cp = ck.latest(authority["run"], authority["lineage"])
    if cp is None:
        cp = authority.get("bootstrap")
    need(cp is not None, "Authority has no full checkpoint")
    return ck.validate(cp)


def choose_nodes(promoted, capacity):
    if not promoted:
        return 1, "single_only_before_handoff"
    slots = capacity.get("free_slots")
    if slots is None:
        return 1, "capacity_unknown"
    need(isinstance(slots, int) and slots >= 0, "Invalid free-node count")
    return (2, "at_least_one_free_node") if slots >= 1 else (1, "no_free_node")


class Controller:
    def __init__(self, root, kind, pbs=None, job_factory=make_job):
        self.p = paths(root, kind)
        self.kind, self.pbs, self.job_factory = kind, pbs or PBS(), job_factory
        self.control = self.p["control"]
        self.state = read(self.control / "state.json")
        self.plan = read(self.control / "operations" / self.state["operation"] / "plan.json")

    def save(self, **changes):
        self.state.update(changes, updated_utc=now())
        write(self.control / "state.json", self.state)

    def event(self, event, **fields):
        folder = self.control / "events"
        folder.mkdir(exist_ok=True)
        write(folder / (now().replace(":", "") + "_" + uuid.uuid4().hex + ".json"),
              dict(event=event, kind=self.kind, at_utc=now(), **fields))

    def submit(self, role, nodes, source):
        pending = self.state.setdefault("pending", {})
        if role not in pending:
            if role == "formal":
                record = authoritative_checkpoint(source, self.kind)
                if self.plan.get("training_policy"):
                    from .evaluation import import_legacy
                    active = self.state.get("active")
                    # This path is reached only after terminal observation/stage
                    # fencing. Pins can be supplied by a controlled restart.
                    if active and "record" not in source:
                        import_legacy(active, record)
                pinned = ck.pin(record, self.control / "resume_sources" / uuid.uuid4().hex)
                source = dict(record=pinned, lineage=(self.state["authority"].get("lineage")
                             or self.plan["single_lineage"]))
            made = self.job_factory(self.plan, role, nodes, source, self.state["deadline_utc"])
            pending[role] = dict(made, nodes=nodes, source=source)
            self.save(pending=pending)
        item = pending[role]
        job = self.pbs.submit(item["folder"], item["job_name"])
        request = read(Path(job["folder"]) / "request.json")
        if role == "candidate":
            self.save(candidate=job, candidate_submitted=True)
        else:
            authority = dict(run=request["run_dir"], lineage=request["lineage"], legacy=False,
                             bootstrap=request["source"]["record"])
            self.save(active=job, authority=authority, rollback_required=False, training_blocked=None)
        del pending[role]
        self.save(pending=pending)
        self.event("SUBMITTED", role=role, nodes=item["nodes"], job=job)
        return job

    def retire_old_candidate(self):
        if self.state.get("old_candidate_retired"):
            return True
        old = self.state.get("old_candidate")
        if old and not self.pbs.cancel(old, self.control / "retire_old_candidate.json"):
            return False
        self.save(old_candidate_retired=True)
        self.event("OLD_CANDIDATE_RETIRED", job=old)
        return True

    def fail_candidate(self, reason):
        self.save(phase="SINGLE_ONLY", handoff_failed=True, failure_reason=reason,
                  rollback_required=bool(self.state.get("single_fenced")))
        self.event("HANDOFF_FAILED_SINGLE_ONLY", reason=reason)

    def monitor_formal(self, allow_submit=True, force_single=False):
        s = self.state
        active = s.get("active")
        if active:
            row = self.pbs.observe(active)
            self.save(active_observation=row)
            if not terminal(row):
                return
            if s.get("single_fenced") and not s.get("promoted") and s["phase"] != "SINGLE_ONLY":
                return
            with locked(Path(s["authority"]["run"]) / "stage.lock"):
                cp = authoritative_checkpoint(s["authority"], self.kind)
            if cp["step"] >= TOTAL_STEPS and str(row["Exit_status"]) == "0":
                self.save(training_done=True)
                return
            if not retryable(row) and not s.get("rollback_required"):
                self.save(training_blocked="Application exit " + str(row["Exit_status"]))
                return
        if not allow_submit and not s.get("rollback_required"):
            return
        if not active and not s.get("rollback_required") and not s.get("pending", {}).get("formal"):
            return
        # No cancellation/requeue when an active job is Q/H/R. Only reached after
        # a positively confirmed terminal job, or an intentional rollback.
        adaptive = s.get("promoted", False) and not force_single
        capacity = self.pbs.capacity() if adaptive else {}
        nodes, reason = choose_nodes(adaptive, capacity)
        self.save(retry_selection=dict(nodes=nodes, reason=reason, capacity=capacity, at_utc=now()))
        self.submit("formal", nodes, s["authority"])
        if active and Path(active["folder"]).parent == self.p["run"] / "jobs":
            # A newer resume source is now durably pinned. Retire temporary pins
            # from the ended process, retaining its small provenance records.
            try:
                previous = read(Path(active["folder"]) / "request.json")
                for name in ("A", "B", "C", "D", "E", "train"):
                    ck.remove_pin(Path(active["folder"]) / name / "source")
                if previous["role"] == "formal":
                    source = Path(previous["source"]["record"]["path"]).parent
                    from .common import below
                    below(source, self.control / "resume_sources")
                    ck.remove_pin(source)
            except (OSError, RuntimeError) as e:
                self.save(cleanup_error=str(e))
                self.event("SOURCE_CLEANUP_DEFERRED", error=str(e))

    def verify_evidence(self, job, filename):
        folder = Path(job["folder"])
        row = read(folder / filename)
        request = read(folder / "request.json")
        need(row["identity"] == request["identity"] and row["job_id"] == job["job_id"],
             "Candidate evidence belongs to another job")
        allocation = read(folder / "allocation.json")
        need(row["execution_id"] == allocation["execution_id"], "Candidate evidence belongs to an old execution")
        observed = self.pbs.observe(job)
        need(observed.get("job_state") == "R" and str(observed.get("run_count")) == str(allocation["run_count"])
             and observed.get("stime") == allocation["stime"], "Candidate is not the live verified execution")
        for name, expected in row["evidence"].items():
            from .common import below
            path = below(folder / name, folder)
            need(sha(path) == expected, "Candidate evidence changed: " + name)
        ck.validate(row["checkpoint"])
        return row

    def qualification(self):
        s = self.state
        job = s.get("candidate")
        if not job:
            if not s.get("candidate_submitted"):
                self.submit("candidate", 2, s["authority"])
            return
        row = self.pbs.observe(job)
        self.save(candidate_observation=row)
        if terminal(row):
            # INCLUDING 271: an unsuccessful initial transition is never retried.
            self.fail_candidate("Candidate ended before handoff commit: " + str(row["Exit_status"]))
            return
        ready_file = Path(job["folder"]) / "ready.json"
        if s["phase"] == "QUALIFYING":
            if not ready_file.exists():
                return
            ready = self.verify_evidence(job, "ready.json")
            from .evidence import verify_qualification
            verify_qualification(Path(job["folder"]), ready)
            # Pin before recording a cancellation intent, then pin the final
            # highest full single checkpoint after PBS termination/stage unlock.
            before = authoritative_checkpoint(s["authority"], self.kind)
            fallback = ck.pin(before, self.control / "fallback" / ("before_" + uuid.uuid4().hex))
            self.save(phase="STOPPING_SINGLE", single_fenced=True, fallback=fallback, ready=ready)
            self.event("SINGLE_FENCED", ready_step=ready["checkpoint"]["step"])
        if s["phase"] == "STOPPING_SINGLE":
            if not self.pbs.cancel(s["active"], self.control / "single_qdel.json"):
                return
            # The stage lock proves the old writer is actually gone after PBS F.
            with locked(Path(s["authority"]["run"]) / "stage.lock"):
                last = authoritative_checkpoint(s["authority"], self.kind)
                need(last["step"] >= s["fallback"]["step"], "Single checkpoint regressed")
                final = ck.pin(last, self.control / "fallback" / ("final_" + str(last["step"])))
            decision = dict(identity=s["ready"]["identity"], job_id=job["job_id"],
                            execution_id=s["ready"]["execution_id"], fallback=final,
                            target=max(final["step"], s["ready"]["checkpoint"]["step"]), at_utc=now())
            self.save(phase="CATCH_UP", fallback=final, release=decision)
            write(Path(job["folder"]) / "release.json", decision)
            self.event("CATCH_UP_RELEASED", target=decision["target"])
        if s["phase"] == "CATCH_UP":
            # Recover crash after state persisted but before release publication.
            write(Path(job["folder"]) / "release.json", s["release"])
            matched = Path(job["folder"]) / "matched.json"
            if not matched.exists():
                return
            proof = self.verify_evidence(job, "matched.json")
            need(proof["checkpoint"]["step"] == s["release"]["target"], "Candidate did not catch up")
            req = read(Path(job["folder"]) / "request.json")
            commit = dict(identity=req["identity"], job=job, checkpoint=proof["checkpoint"],
                          execution_id=proof["execution_id"], at_utc=now(),
                          previous_authority=s["authority"], previous_job=s["active"], fallback=s["fallback"],
                          authority=dict(run=req["run_dir"], lineage=req["lineage"], legacy=False))
            # Durable commit FIRST. A crash now is promoted, never rolled back.
            write(self.control / "handoff_commit.json", commit)
            self.reconcile_commit()

    def reconcile_commit(self):
        file = self.control / "handoff_commit.json"
        if not file.exists():
            return
        c = read(file)
        req = read(Path(c["job"]["folder"]) / "request.json")
        need(c["identity"] == req["identity"] and req["lineage"] == self.plan["candidate_lineage"],
             "Foreign handoff commit")
        if not self.state.get("promoted"):
            ck.validate(c["checkpoint"])
            self.save(promoted=True, phase="ACTIVE", active=c["job"], authority=c["authority"],
                      rollback_required=False, handoff_failed=False, training_blocked=None)
            self.event("HANDOFF_COMMITTED", job=c["job"])
        write(Path(c["job"]["folder"]) / "committed.json", c)
        # Cleanup happens only after the durable authority change. Metadata stays.
        if not self.state.get("fallback_retired"):
            try:
                self.retire_fallback(c)
            except (OSError, RuntimeError) as e:
                self.save(cleanup_error=str(e))
                self.event("FALLBACK_CLEANUP_DEFERRED", error=str(e))

    def retire_fallback(self, c, scope=None):
        retired = (Path(scope) / "fallback_retired.json").exists() if scope else self.state.get("fallback_retired")
        if not retired:
            previous = c.get("previous_authority")
            if previous:
                need(terminal(self.pbs.observe(c["previous_job"])), "Previous writer is not terminal; retain fallback")
                old_run = Path(previous["run"])
                need(old_run != Path(c["authority"]["run"]), "Cannot prune the formal lineage")
                # Only completed checkpoints of the one fenced previous writer.
                # Uncommitted saves and failed-candidate evidence are left intact.
                with locked(old_run / "stage.lock"):
                    for item in (old_run / "checkpoints").glob("global_step_*"):
                        from .common import below
                        need(not item.is_symlink(), "Cannot prune a checkpoint symlink")
                        below(item, old_run / "checkpoints")
                        if not item.name.removeprefix("global_step_").isdigit():
                            continue
                        step = int(item.name.removeprefix("global_step_"))
                        if step > c["fallback"]["step"]:
                            continue
                        ck.components(item, step, 8)
                        if not previous.get("legacy"):
                            need(read(item / ck.MARKER)["lineage"] == previous["lineage"], "Foreign previous checkpoint")
                        self.event("RETIRE_PREVIOUS_CHECKPOINT", path=str(item), step=step)
                        shutil.rmtree(item)
            for source in ((Path(scope) if scope else self.control) / "fallback").glob("*/source.json"):
                ck.remove_pin(source.parent)
            if scope:
                write(Path(scope) / "fallback_retired.json", dict(at_utc=now(), identity=c["identity"]))
                self.save(cleanup_error=None)
            else:
                self.save(fallback_retired=True, cleanup_error=None)

    def tick(self):
        self.reconcile_commit()
        from .checkpoint_restart import tick as restart_tick
        if restart_tick(self):
            return
        from .opportunities import Opportunities
        opportunities = Opportunities(self)
        if opportunities.owns_tick():
            return opportunities.tick()
        s = self.state
        deadline = expired(s["deadline_utc"])
        # Complete an acknowledged submission transaction before considering a
        # new generation. A crash after adoption must not retain a stale intent.
        for role, item in list(s.get("pending", {}).items()):
            destination = s.get("candidate" if role == "candidate" else "active")
            if destination and destination["folder"] == item["folder"]:
                del s["pending"][role]
                self.save(pending=s["pending"])
            elif deadline and not (Path(item["folder"]) / "qsub_attempted.json").exists() and not s.get("rollback_required"):
                del s["pending"][role]
                self.save(pending=s["pending"])
        if s.get("pending", {}).get("formal"):
            item = s["pending"]["formal"]
            self.submit("formal", item["nodes"], item["source"])
            return
        # Formal monitoring proceeds even if a candidate query/retirement fails.
        if not s.get("rollback_required"):
            self.monitor_formal(allow_submit=not deadline)
        if s.get("pending", {}).get("candidate"):
            item = s["pending"]["candidate"]
            self.submit("candidate", item["nodes"], item["source"])
            return
        if s.get("promoted"):
            if deadline or s.get("training_done"):
                self.save(phase="DONE" if s.get("training_done") else "EXPIRED", monitoring_finished=True)
            return
        if deadline or s.get("training_done"):
            if not s.get("handoff_failed"):
                self.fail_candidate("Monitor deadline" if deadline else "Single training complete")
        if s["phase"] == "SINGLE_ONLY":
            if not self.retire_old_candidate():
                return
            candidate = s.get("candidate")
            if candidate and not self.pbs.cancel(candidate, self.control / "failed_candidate_qdel.json"):
                return
            # A qdel intent may have succeeded before failure was observed.
            # Wait for both candidates and old single to terminate before restart.
            if s.get("rollback_required"):
                row = self.pbs.observe(s["active"])
                if terminal(row):
                    with locked(Path(s["authority"]["run"]) / "stage.lock"):
                        latest = authoritative_checkpoint(s["authority"], self.kind)
                        if latest["step"] < s["fallback"]["step"]:
                            self.save(authority=dict(s["authority"], record=s["fallback"]))
                    self.monitor_formal(allow_submit=True)  # One safety rollback even at deadline.
                    self.save(single_fenced=False)
                else:
                    if (self.control / "single_qdel.json").exists():
                        self.pbs.cancel(s["active"], self.control / "single_qdel.json")
                        return
                    self.save(rollback_required=False, single_fenced=False)
            if deadline and not s.get("rollback_required"):
                self.save(phase="EXPIRED", monitoring_finished=True)
            return
        if not s.get("candidate_admitted"):
            return
        if not self.retire_old_candidate():
            return
        self.qualification()
