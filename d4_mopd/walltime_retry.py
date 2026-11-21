"""Guardian policy for an identity-checked PBS walltime and a recorded Ray SIGTERM."""
from pathlib import Path
import re

from .job_exit import EVIDENCE_SCHEMA, RAY_DRIVER_SIGTERM, ray_exit_trace
from .job_exit_v2 import EXIT_SCHEMA, retry_assessment as prior_assessment


def retry_assessment(row, result, job, identity):
    comment = " ".join(str(row.get("comment", "")).split())
    walltime = (row.get("job_state") in ("F", "C") and str(row.get("Exit_status")) == "-29"
                and re.fullmatch(r"(?:Job run at .+ and )?exceeded resource walltime", comment,
                                 flags=re.IGNORECASE) is not None)
    if result.get("status") != "FAIL" or not walltime:
        return prior_assessment(row, result, job, identity)

    details = result.get("exit_details")
    expected_details = dict(schema=EXIT_SCHEMA, type="SystemExit", code=1,
                            ray_terminate_msg=RAY_DRIVER_SIGTERM)
    if not identity or result.get("identity") != identity or not ray_exit_trace(result):
        return dict(allowed=False, reason="walltime_without_bound_ray_exit")
    if details != expected_details or type(details["code"]) is not int:
        return dict(allowed=False, reason="walltime_without_captured_sigterm")
    if "termination_evidence" in result:
        marker = dict(schema=EVIDENCE_SCHEMA, signal=15, ray_terminate_msg=RAY_DRIVER_SIGTERM,
                      identity=identity, attempt=str(Path(job["folder"]).resolve()))
        if result["termination_evidence"] != marker:
            return dict(allowed=False, reason="conflicting_termination_evidence")
    return dict(allowed=True, reason="pbs_walltime_with_captured_ray_sigterm")
