"""Capture driver exit evidence before cleanup and explain retry decisions."""
from pathlib import Path
import traceback

from .job_exit import EVIDENCE_SCHEMA, RAY_DRIVER_SIGTERM, failure_result, ray_exit_trace, root_sigterm
from d4_teachers.common import now


EXIT_SCHEMA = "d4-mopd-exit-details-v2"


def failure_record(exc, identity):
    result = dict(status="FAIL", identity=identity, error=str(exc),
                  traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), at_utc=now())
    if type(exc) is SystemExit:
        message = getattr(exc, "ray_terminate_msg", None)
        result["exit_details"] = dict(schema=EXIT_SCHEMA, type="SystemExit",
            code=exc.code if type(exc.code) in (int, str, type(None)) else repr(exc.code),
            ray_terminate_msg=message if isinstance(message, (str, type(None))) else repr(message))
    return result


def retry_assessment(row, result, job, identity):
    """PBS.observe must first bind the row to this owner/queue/script/attempt.

    A historical Ray signal-bridge stack plus explicit PBS root termination is
    sufficient without the old post-shutdown marker. New contradictory evidence
    is never ignored. This is an additional gate, not a replacement for PBS policy.
    """
    def answer(allowed, reason):
        return dict(allowed=allowed, reason=reason)

    if result.get("status") != "FAIL":
        return answer(True, "no_application_failure")
    if not root_sigterm(row):
        return answer(False, "failure_without_pbs_root_sigterm")
    if result.get("identity") != identity:
        return answer(False, "unbound_failure_or_failed_node_preflight")
    trace = result.get("traceback", "")
    bridge = ray_exit_trace(result)
    direct = (result.get("error") == "15" and isinstance(trace, str)
              and trace.count("Traceback (most recent call last):") == 1
              and trace.rstrip().endswith("SystemExit: 15")
              and "in sigterm_handler" in trace and "sys.exit(signum)" in trace)
    if not (bridge or direct):
        return answer(False, "application_failure_not_ray_signal_exit")
    if "termination_evidence" in result:
        expected = dict(schema=EVIDENCE_SCHEMA, signal=15, ray_terminate_msg=RAY_DRIVER_SIGTERM,
                        identity=identity, attempt=str(Path(job["folder"]).resolve()))
        if not bridge or result["termination_evidence"] != expected:
            return answer(False, "conflicting_termination_evidence")
    if "exit_details" in result:
        expected = dict(schema=EXIT_SCHEMA, type="SystemExit", code=1 if bridge else 15,
                        ray_terminate_msg=RAY_DRIVER_SIGTERM if bridge else None)
        if result["exit_details"] != expected:
            return answer(False, "conflicting_exit_details")
        return answer(True, "pbs_root_sigterm_captured_before_cleanup")
    if "termination_evidence" in result:
        return answer(True, "pbs_root_sigterm_with_legacy_marker")
    return answer(True, "pbs_root_sigterm_with_ray_trace")


def entry_script(m, *, runtime=None):
    # Fresh manifests use their original immutable snapshot. A repaired guardian
    # explicitly supplies its verified revision; never import a mutable checkout.
    runtime = Path(runtime) if runtime is not None else Path(m["manifest_path"]).parent / "runtime"
    paths = [str(runtime), m["verl_src"]]
    return "import sys\nsys.path[:0]=" + repr(paths) + "\nfrom d4_mopd.cli import main\nmain()\n"
