"""Operational exit evidence; never infer a signal from SystemExit(1) alone."""
import json
from pathlib import Path
import re


RAY_DRIVER_SIGTERM = "SystemExit is raised (sys.exit is called). Exit code: 15."
EVIDENCE_SCHEMA = "d4-mopd-ray-sigterm-v1"


def ray_exit_trace(result):
    trace = result.get("traceback", "")
    return (result.get("status") == "FAIL" and result.get("error") == "1"
            and isinstance(trace, str) and trace.count("Traceback (most recent call last):") == 1
            and trace.rstrip().endswith("SystemExit: 1")
            and all(x in trace for x in ("CoreWorker.get_objects", "check_status",
                                        "raise_sys_exit_with_custom_error_message")))


def root_sigterm(row):
    return (row.get("job_state") == "F" and str(row.get("Exit_status")) == "271"
            and re.fullmatch(r"(?:Job run at .+ and )?terminated by root@[A-Za-z0-9][A-Za-z0-9._-]*",
                             " ".join(str(row.get("comment", "")).split())) is not None)


def record_ray_exit(exc, request_path):
    """Called by the entry wrapper after the unchanged driver wrote its FAIL.

    Ray 2.55.1's driver exits with 15 on SIGTERM. Its Cython signal bridge
    rethrows that as SystemExit(1), preserving the original 15 in this attribute.
    A generic SystemExit or an application exception must remain an ordinary FAIL.
    """
    if type(exc) is not SystemExit or exc.code != 1 or getattr(exc, "ray_terminate_msg", None) != RAY_DRIVER_SIGTERM:
        return
    request = json.loads(Path(request_path).read_text())
    path = Path(request["folder"]) / "result.json"
    if not path.is_file():
        return
    result = json.loads(path.read_text())
    if result.get("identity") != request["identity"] or not ray_exit_trace(result):
        return
    result["termination_evidence"] = dict(schema=EVIDENCE_SCHEMA, signal=15,
        ray_terminate_msg=exc.ray_terminate_msg, identity=request["identity"],
        attempt=str(Path(request["folder"]).resolve()))
    temporary = path.with_suffix(".termination.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def retry_allowed(row, result, job, identity):
    if result.get("status") != "FAIL":
        return True
    return (root_sigterm(row) and result.get("identity") == identity and ray_exit_trace(result)
            and result.get("termination_evidence") == dict(schema=EVIDENCE_SCHEMA, signal=15,
                ray_terminate_msg=RAY_DRIVER_SIGTERM, identity=identity,
                attempt=str(Path(job["folder"]).resolve())))


def failure_result(job):
    folder = Path(job["folder"])
    path = folder / "result.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    # A real preflight failure always overrides signal evidence from the driver.
    for name in ("cuda_runtime", "gpu_node"):
        for rank in (0, 1):
            path = folder / f"{name}_{rank}.json"
            proof = json.loads(path.read_text()) if path.exists() else {}
            if proof.get("status") == "FAIL":
                return dict(status="FAIL", error=f"{name} rank {rank}: " + proof["error"])
    return result


def entry_script(m, support, request_path):
    """All learner imports still resolve to the original manifest runtime."""
    paths = [str(Path(m["manifest_path"]).parent / "runtime"), m["verl_src"]]
    return ("import runpy, sys\nsys.path[:0]=" + repr(paths)
            + "\ntry:\n    from d4_mopd.cli import main\n    main()\n"
            + "except SystemExit as exc:\n"
            + "    runpy.run_path(" + repr(str(Path(support).resolve()))
            + ")[\"record_ray_exit\"](exc, " + repr(str(request_path)) + ")\n    raise\n")
