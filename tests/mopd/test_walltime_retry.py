from copy import deepcopy

import pytest

from d4_mopd import control, job_exit, job_exit_v2
from d4_teachers.common import write
from test_cuda_runtime import TRACE


@pytest.fixture
def expired(tmp_path):
    row = dict(job_state="F", Exit_status=-29, comment="Job run at example and exceeded resource walltime")
    details = dict(schema=job_exit_v2.EXIT_SCHEMA, type="SystemExit", code=1,
                   ray_terminate_msg=job_exit.RAY_DRIVER_SIGTERM)
    result = dict(status="FAIL", identity="example", error="1", traceback=TRACE, exit_details=details)
    return row, result, dict(folder=str(tmp_path), job_id="1.pbs1")


@pytest.mark.parametrize("change", [
    {"job_state": "R"}, {"Exit_status": 1}, {"comment": "walltime requested"},
    {"comment": "has not exceeded resource walltime"}, {"comment": "exceeded resource mem"},
])
def test_needs_explicit_terminal_walltime(expired, change):
    row, result, job = expired
    assert not control.retry_assessment(dict(row, **change), result, job, "example")["allowed"]


@pytest.mark.parametrize("change", [
    {"identity": "other"}, {"exit_details": None}, {"traceback": "SystemExit: 1"},
    {"traceback": "Traceback (most recent call last):\nRuntimeError: OOM\n" + TRACE},
    {"termination_evidence": None}, {"error": "CUDA out of memory"},
    {"exit_details": dict(schema=job_exit_v2.EXIT_SCHEMA, type="SystemExit", code=1,
                          ray_terminate_msg="application exit")},
])
def test_application_failures_stay_blocked(expired, change):
    row, result, job = expired
    assert not control.retry_assessment(row, dict(result, **change), job, "example")["allowed"]


def test_node_failure_overrides_walltime(expired, tmp_path):
    row, result, job = expired
    write(tmp_path / "result.json", result)
    write(tmp_path / "gpu_node_0.json", {"status": "FAIL", "error": "GPU preflight failed"})
    assert not control.retry_assessment(row, control.failure_result(job), job, "example")["allowed"]


def test_tick_retries_once_and_preserves_evidence(expired, tmp_path, manifest):
    row, result, job = expired
    result["identity"] = manifest["identity"]
    write(tmp_path / "result.json", result)
    state = {"train:domain_mix": dict(status="R", job=job, pending=None, phase="train", arm="domain_mix")}
    original = deepcopy(result)
    submitted = []
    class Scheduler:
        def observe(self, candidate):
            return row if candidate == job else {"job_state": "Q"}
        def submit(self, folder, job_name):
            submitted.append(folder)
            return dict(folder=folder, job_name=job_name, job_id="2.pbs1")
    pbs = Scheduler()
    control.tick(manifest, state, pbs)
    task = state["train:domain_mix"]
    assert task["status"] == "READY" and task["previous_job"] == job
    assert task["retry_reason"] == "pbs_walltime" and task["external_interruptions"] == 1
    assert task["retry_assessment"]["reason"] == "pbs_walltime_with_captured_ray_sigterm"
    control.tick(manifest, state, pbs)
    control.tick(manifest, state, pbs)
    assert task["status"] == "Q" and len(submitted) == 1
    assert control.failure_result(job) == original
