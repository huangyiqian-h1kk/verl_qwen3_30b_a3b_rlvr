"""Opportunistic, independently journaled 1→2 handoffs; no live job mutation.

Normal retry ownership stays with Controller. Every opportunity has a separate
lineage, qdel journal, fallback pins and commit. A prior commit is never reused
to promote a new candidate. Capacity is an admission hint; PBS owns allocation.
"""
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from . import TOTAL_STEPS
from . import checkpoints as ck
from .common import active, expired, locked, need, now, paths, read, write
from .scheduler import Unknown, finished, retryable, terminal

POLICY = dict(version="2026-10-22.opportunities.1", poll_seconds=300,
              confirmations=2, min_free_slots=2, cooldown_seconds=1800)
IN_FLIGHT = {"SUBMITTING", "QUALIFYING", "STOPPING_SINGLE", "CATCH_UP", "ABORTING"}


def seconds_since(value):
    return (datetime.fromisoformat(now()) - datetime.fromisoformat(value)).total_seconds()


def nodes(row):
    resources = row.get("Resource_List", {})
    value = resources.get("nodect", row.get("Resource_List.nodect"))
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise Unknown("PBS did not report the active allocation node count")
    need(result in (1, 2), "Unexpected formal training allocation")
    return result


def external_interruption(row):
    """Distinguish administrator interruption from user/our own cancellation."""
    if not retryable(row):
        return False
    return str(row["Exit_status"]) == "-29" or "terminated by root@" in str(row.get("comment", "")).lower()


def application_failed(job):
    path = Path(job["folder"]) / "result.json"
    return path.exists() and read(path).get("status") == "FAIL"


def authorized(state, request, allow_committed=False):
    """Startup guard for an exact attempt, including when promoted was already true."""
    attempt = state.get("opportunity") or {}
    candidate = attempt.get("job") or attempt.get("pending") or {}
    return (attempt.get("id") == request.get("opportunity_id")
            and attempt.get("phase") in ((IN_FLIGHT - {"ABORTING"}) | ({"COMMITTED"} if allow_committed else set()))
            and candidate.get("folder") == request["job_dir"]
            and attempt.get("lineage") == request["lineage"])


class Opportunities:
    def __init__(self, controller):
        self.c = controller
        self.policy = controller.plan.get("opportunistic")
        self.area = controller.p["root"] / "outputs/multidomain/continuation_operations"

    @property
    def s(self):
        return self.c.state

    @property
    def attempt(self):
        return self.s.get("opportunity") or {}

    def directory(self):
        return self.c.control / "opportunities" / self.attempt["id"]

    def save(self, **changes):
        attempt = dict(self.attempt, **changes)
        # The controller state is the authority; the copy is an audit record.
        self.c.save(opportunity=attempt)
        write(self.directory() / "state.json", attempt)

    def owns_tick(self):
        return bool(self.policy and (self.attempt or
                    (self.s.get("phase") in ("ACTIVE", "SINGLE_ONLY")
                     and not self.s.get("rollback_required"))))

    def failure(self, reason, classification="validation"):
        self.save(phase="ABORTING", failure_class=classification, failure_reason=reason)
        self.c.event("OPPORTUNITY_ABORTING", attempt=self.attempt["id"],
                     reason=reason, failure_class=classification)

    def abort(self):
        a = self.attempt
        job = a.get("job")
        if not job and a.get("pending"):
            pending = a["pending"]
            if (Path(pending["folder"]) / "qsub_attempted.json").exists():
                job = self.c.pbs.recover(pending["folder"], pending["job_name"])
                need(job is not None, "Ambiguous opportunity submission remains unresolved")
                self.save(job=job, pending=None)
        if job and not self.c.pbs.cancel(job, self.directory() / "candidate_qdel.json"):
            return
        # Roll back only if this attempt fenced the original single. A new
        # formal receipt may already have been adopted before a controller crash.
        original = a.get("fenced_job")
        if original and (self.s.get("active") == original or self.s.get("rollback_required")):
            if not self.c.pbs.cancel(original, self.directory() / "single_qdel.json"):
                return
            with locked(Path(a["previous_authority"]["run"]) / "stage.lock"):
                from .controller import authoritative_checkpoint
                latest = authoritative_checkpoint(a["previous_authority"], self.c.kind)
                authority = a["previous_authority"]
                if latest["step"] < a["fallback"]["step"]:
                    authority = dict(authority, record=a["fallback"])
            # cancel() positively confirmed F, and the writer lease is released.
            # An intentionally cancelled PBS job need not carry an exit code.
            self.c.save(active=None, authority=authority, rollback_required=True)
            self.c.monitor_formal(allow_submit=True, force_single=True)
            if not self.s.get("active") or self.s.get("active") == original:
                return
        until = (datetime.fromisoformat(now()) + timedelta(seconds=self.policy["cooldown_seconds"])).isoformat()
        self.save(phase="FAILED", finished_utc=now(), retry_after_utc=until)
        self.c.save(phase="SINGLE_ONLY", handoff_failed=True,
                    failure_reason=a["failure_reason"], rollback_required=False, single_fenced=False,
                    opportunity_poll={"reason": "cooldown" if a["failure_class"] == "external" else "upgrade_blocked"})
        self.c.event("OPPORTUNITY_FAILED", attempt=a["id"], failure_class=a["failure_class"])

    def reconcile(self):
        if not self.attempt:
            return
        file = self.directory() / "commit.json"
        if not file.exists():
            return
        commit = read(file)
        a = self.attempt
        need(commit["attempt"] == a["id"] and commit["job"] == a["job"], "Foreign opportunity commit")
        request = read(Path(commit["job"]["folder"]) / "request.json")
        need(request["identity"] == commit["identity"] and request["lineage"] == a["lineage"],
             "Opportunity commit identity changed")
        if a["phase"] != "COMMITTED":
            ck.validate(commit["checkpoint"])
            # One atomic state change, before acknowledging the candidate.
            self.c.save(opportunity=dict(a, phase="COMMITTED", finished_utc=commit["at_utc"]),
                        promoted=True, phase="ACTIVE", active=commit["job"], authority=commit["authority"],
                        rollback_required=False, single_fenced=False, handoff_failed=False,
                        failure_reason=None, training_blocked=None, opportunity_poll={"reason": "dual_active"})
            self.c.event("OPPORTUNITY_COMMITTED", attempt=a["id"], job=commit["job"])
        write(Path(commit["job"]["folder"]) / "committed.json", commit)
        try:
            self.c.retire_fallback(commit, scope=self.directory())
        except (OSError, RuntimeError) as error:
            self.c.save(cleanup_error=str(error))

    def transition(self):
        a = self.attempt
        if a["phase"] == "ABORTING":
            return self.abort()
        if expired(self.s["deadline_utc"]):
            self.failure("Monitor deadline", "deadline")
            return self.abort()
        if a["phase"] == "SUBMITTING":
            if not a.get("pending"):
                plan = dict(self.c.plan, candidate_lineage=a["lineage"], opportunity_id=a["id"])
                made = self.c.job_factory(plan, "candidate", 2, self.s["authority"], self.s["deadline_utc"])
                self.save(pending=made)
            item = self.attempt["pending"]
            job = self.c.pbs.submit(item["folder"], item["job_name"])
            self.save(phase="QUALIFYING", job=job, pending=None)
            self.c.save(candidate=job)
            self.c.event("OPPORTUNITY_SUBMITTED", attempt=a["id"], job=job)
            return
        job = a["job"]
        row = self.c.pbs.observe(job)
        self.c.save(candidate_observation=row)
        if finished(row):
            if not terminal(row):
                raise Unknown("Candidate ended without a usable exit status")
            classification = "external" if external_interruption(row) and not application_failed(job) else "application"
            self.failure("Candidate ended before handoff commit: " + str(row["Exit_status"]), classification)
            return self.abort()
        if application_failed(job):
            self.failure("Candidate recorded an application failure", "application")
            return self.abort()
        folder = Path(job["folder"])
        if a["phase"] == "QUALIFYING":
            if not (folder / "ready.json").exists():
                return
            ready = self.c.verify_evidence(job, "ready.json")
            from .evidence import verify_qualification
            from .controller import authoritative_checkpoint
            verify_qualification(folder, ready)
            request = read(folder / "request.json")
            need(request.get("opportunity_id") == a["id"] and request["lineage"] == a["lineage"],
                 "Candidate belongs to another opportunity")
            need(nodes(self.c.pbs.observe(self.s["active"])) == 1, "Original training is no longer single-node")
            protection = self.directory() / "fallback/before"
            if (protection / "source.json").exists():
                fallback = ck.validate(read(protection / "source.json"))
            else:
                before = (read(protection / "pin_intent.json") if (protection / "pin_intent.json").exists()
                          else authoritative_checkpoint(self.s["authority"], self.c.kind))
                fallback = ck.pin(before, protection)
            self.save(phase="STOPPING_SINGLE", ready=ready, fallback=fallback,
                      previous_authority=self.s["authority"], fenced_job=self.s["active"])
        a = self.attempt
        if a["phase"] == "STOPPING_SINGLE":
            # Recheck a live, qualified execution immediately before cancellation.
            self.c.verify_evidence(job, "ready.json")
            if not self.c.pbs.cancel(a["fenced_job"], self.directory() / "single_qdel.json"):
                return
            with locked(Path(a["previous_authority"]["run"]) / "stage.lock"):
                from .controller import authoritative_checkpoint
                last = authoritative_checkpoint(a["previous_authority"], self.c.kind)
                need(last["step"] >= a["fallback"]["step"], "Single checkpoint regressed")
                final = ck.pin(last, self.directory() / "fallback/final")
            decision = dict(identity=a["ready"]["identity"], job_id=job["job_id"],
                            execution_id=a["ready"]["execution_id"], fallback=final,
                            target=max(final["step"], a["ready"]["checkpoint"]["step"]), at_utc=now())
            self.save(phase="CATCH_UP", fallback=final, release=decision)
        a = self.attempt
        if a["phase"] == "CATCH_UP":
            write(folder / "release.json", a["release"])
            if not (folder / "matched.json").exists():
                return
            proof = self.c.verify_evidence(job, "matched.json")
            need(proof["checkpoint"]["step"] == a["release"]["target"]
                 and proof["checkpoint"]["world_size"] == 16
                 and proof["checkpoint"]["lineage"] == a["lineage"], "Candidate did not catch up in its own lineage")
            request = read(folder / "request.json")
            commit = dict(attempt=a["id"], identity=request["identity"], job=job,
                          execution_id=proof["execution_id"], checkpoint=proof["checkpoint"], at_utc=now(),
                          previous_authority=a["previous_authority"], previous_job=a["fenced_job"],
                          fallback=a["fallback"],
                          authority=dict(run=request["run_dir"], lineage=a["lineage"], legacy=False))
            write(self.directory() / "commit.json", commit)
            self.reconcile()

    def may_retry(self, state=None):
        state = self.s if state is None else state
        attempt = state.get("opportunity") or {}
        if attempt.get("phase") == "FAILED":
            if attempt.get("failure_class") != "external":
                return False
            return expired(attempt["retry_after_utc"])
        if not state.get("handoff_failed"):
            return True
        # Upgrade of an old 271 latch: require the exact termination reason,
        # a fresh identity-checked PBS observation and no recorded FAIL result.
        job = state.get("candidate")
        if not job or not str(state.get("failure_reason", "")).startswith("Candidate ended before handoff commit:"):
            return False
        return external_interruption(self.c.pbs.observe(job)) and not application_failed(job)

    def peer_reservation(self):
        other = "domain" if self.c.kind == "random" else "random"
        p = paths(self.c.p["root"], other)
        if not (p["control"] / "state.json").exists():
            return 0, False
        state = read(p["control"] / "state.json")
        a = state.get("opportunity") or {}
        if a.get("phase") in IN_FLIGHT:
            job = a.get("job")
            if not job:
                return 2, False  # includes unresolved qsub
            row = self.c.pbs.observe(job)
            # A running allocation is already excluded by pbsnodes; do not
            # subtract it twice. Q/H and unknown termination reserve the request.
            return (0 if row.get("job_state") == "R" or terminal(row) else 2), False
        priority = False
        planfile = p["control"] / "operations" / state["operation"] / "plan.json"
        if (other == "domain" and read(planfile).get("opportunistic")
                and state.get("phase") in ("ACTIVE", "SINGLE_ONLY")
                and not state.get("training_done") and not state.get("training_blocked")
                and not expired(state["deadline_utc"]) and active(p["control"] / "controller.lock")
                and state.get("heartbeat_utc") and seconds_since(state["heartbeat_utc"]) < 600
                and self.may_retry(state) and state.get("active")):
            row = self.c.pbs.observe(state["active"])
            priority = row.get("job_state") == "R" and nodes(row) == 1
            # A setup waiting for an 8-rank checkpoint or nearly finished cannot
            # reserve future opportunities indefinitely.
            if priority:
                from .controller import authoritative_checkpoint
                cp = authoritative_checkpoint(state["authority"], other)
                priority = cp["world_size"] == 8 and cp["step"] + 5 <= TOTAL_STEPS
        return 0, priority

    def poll(self):
        from .checkpoint_restart import DONE
        restart = self.s.get("checkpoint_restart")
        if restart and restart["phase"] not in DONE:
            self.c.save(opportunity_poll=dict(at_utc=now(), consecutive=0, reason="checkpoint_restart_pending"))
            return
        previous = self.s.get("opportunity_poll") or {}
        if previous.get("at_utc") and seconds_since(previous["at_utc"]) < self.policy["poll_seconds"]:
            return
        result = dict(at_utc=now(), consecutive=0, reason="not_eligible")
        try:
            if not self.s.get("active") or not self.may_retry() or self.s.get("training_done") or self.s.get("training_blocked"):
                result["reason"] = "upgrade_blocked_or_cooldown"
                return
            row = self.c.pbs.observe(self.s["active"])
            if row.get("job_state") != "R" or nodes(row) != 1:
                result["reason"] = "formal_not_running_single"
                return
            from .controller import authoritative_checkpoint
            cp = authoritative_checkpoint(self.s["authority"], self.c.kind)
            if cp["world_size"] != 8 or cp["step"] + 5 > TOTAL_STEPS:
                result["reason"] = "waiting_for_single_checkpoint_or_training_complete"
                return
            # Serialize both owners through observing capacity and recording an
            # admission. A pending/queued peer consumes two prospective slots.
            with locked(self.area / "opportunity_capacity.lock"):
                reserved, domain_first = self.peer_reservation()
                capacity = self.c.pbs.capacity()
                slots = capacity.get("free_slots")
                result.update(capacity=capacity, peer_reserved_slots=reserved)
                if slots is None:
                    result["reason"] = "capacity_unknown"
                    return
                need(isinstance(slots, int) and slots >= 0, "Invalid free-slot snapshot")
                available = max(0, slots - reserved)
                result["available_after_pending"] = available
                if available < self.policy["min_free_slots"]:
                    result["reason"] = "fewer_than_two_free_slots"
                    return
                same_job = previous.get("job_id") == self.s["active"]["job_id"]
                fresh = previous.get("at_utc") and seconds_since(previous["at_utc"]) <= 2 * self.policy["poll_seconds"] + 60
                result.update(job_id=self.s["active"]["job_id"],
                              consecutive=(previous.get("consecutive", 0) if same_job and fresh else 0) + 1)
                if result["consecutive"] < self.policy["confirmations"]:
                    result["reason"] = "waiting_second_capacity_observation"
                    return
                if domain_first:
                    result["reason"] = "domain_first"
                    return
                attempt = dict(id=uuid.uuid4().hex, lineage=uuid.uuid4().hex, phase="SUBMITTING",
                               admitted_utc=now(), capacity=capacity, previous_candidate=self.s.get("candidate"))
                self.c.save(opportunity=attempt, handoff_failed=False, failure_reason=None)
                write(self.directory() / "state.json", attempt)
                self.c.event("OPPORTUNITY_ADMITTED", attempt=attempt["id"], free_slots=available)
                result["reason"] = "admitted"
        except (Unknown, OSError, ValueError, RuntimeError) as error:
            result.update(reason="capacity_or_eligibility_unknown", error=str(error), consecutive=0)
        finally:
            self.c.save(opportunity_poll=result)

    def tick(self):
        self.reconcile()  # durable commit wins even if the PBS job subsequently ended
        a = self.attempt
        if self.s.get("pending", {}).get("formal"):
            item = self.s["pending"]["formal"]
            self.c.submit("formal", item["nodes"], item["source"])
            return
        if a.get("phase") == "ABORTING":
            if not a.get("fenced_job"):
                self.c.monitor_formal(allow_submit=not expired(self.s["deadline_utc"]), force_single=True)
            self.abort()
            return
        fenced = a.get("phase") in ("STOPPING_SINGLE", "CATCH_UP")
        if not fenced:
            self.c.monitor_formal(allow_submit=not expired(self.s["deadline_utc"]),
                                  force_single=a.get("phase") in IN_FLIGHT or self.s.get("handoff_failed", False))
        if a.get("phase") in IN_FLIGHT:
            try:
                self.transition()
            except (Unknown, BlockingIOError, OSError):
                raise
            except Exception as error:
                self.failure(type(error).__name__ + ": " + str(error))
                self.abort()
            return
        if expired(self.s["deadline_utc"]) or self.s.get("training_done"):
            self.c.save(phase="DONE" if self.s.get("training_done") else "EXPIRED", monitoring_finished=True)
            return
        self.poll()
