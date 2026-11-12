"""Shared-filesystem supervisor and identity-bound PBS submission for two arms."""
import os
from contextlib import ExitStack
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import uuid

from . import ARMS
from .assets import verify
from .job_exit import entry_script, failure_result, retry_allowed
from d4_teachers.common import active, file_sha, lock, need, now, read, write
from d4_training_continuation.common import atomic_text, locked
from d4_training_continuation.scheduler import PBS, Unknown, finished
from d4_teachers.controller import retry_reason


def task_root(m, phase, arm):
    folders = {"qualification": "qualification", "train": "runs", "test": "final_tests"}
    need(phase in folders and arm in ARMS, "Unknown MOPD task")
    return Path(m["manifest_path"]).parent / folders[phase] / arm


def entry_command(m):
    from .launcher_repair import active_runtime
    runtime = active_runtime(m)
    bootstrap = ("import sys; sys.path[:0]=" + repr([str(runtime), m["verl_src"]])
                 + "; from d4_mopd.cli import main; main()")
    return [m["python"], "-I", "-c", bootstrap]


def render_pbs(m, folder, job_name):
    q = shlex.quote
    runtime = Path(m["manifest_path"]).parent / "runtime"
    return "\n".join([
        "#!/bin/bash", "#PBS -P gcg51557", "#PBS -q R9920261000", "#PBS -v RTYPE=rt_HF",
        "#PBS -l select=2", "#PBS -l walltime=24:00:00", "#PBS -N " + job_name,
        "#PBS -r n", "#PBS -c n", "#PBS -j oe", "#PBS -k oe", "set -euo pipefail",
        "exec > >(tee -a " + q(str(folder / "pbs.log")) + ") 2>&1",
        "if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi",
        "module load gcc/13.2.0 hpcx/2.20",
        "awk '!seen[$0]++' \"$PBS_NODEFILE\" > " + q(str(folder / "hosts")),
        "[[ $(wc -l < " + q(str(folder / "hosts")) + ") -eq 2 ]]",
        "mpirun -np 2 --map-by ppr:1:node --bind-to none --hostfile " + q(str(folder / "hosts"))
        + " bash " + q(str(runtime / "d4_mopd/node.sh")) + " " + q(str(folder / "request.json")), ""])


def new_attempt(m, phase, arm):
    run = task_root(m, phase, arm)
    folder = run / "attempts" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    (run / "agent.yaml").write_text("- name: d4_plain_mopd\n  _target_: d4_mopd.agent_loop.MOPDAgentLoop\n")
    request = dict(manifest=m["manifest_path"], identity=m["identity"], phase=phase, arm=arm,
                   folder=str(folder), run_dir=str(run), job_name="0390_mp_" + {"qualification":"q", "train":"t", "test":"e"}[phase]
                   + ("mix" if arm == "domain_mix" else "only"))
    if phase == "test":
        proof = read(task_root(m, "train", arm) / "receipt.json")
        need(proof["status"] == "PASS" and proof["identity"] == m["identity"], "Training has not completed")
        request["selection"] = proof["selection"]
    write(folder / "request.json", request)
    # No mutable checkout is placed on PYTHONPATH, including inside Ray workers.
    (folder / "entry.py").write_text(entry_script(m, Path(__file__).with_name("job_exit.py"), folder / "request.json"))
    (folder / "python.txt").write_text(m["python"] + "\n")
    (folder / "stage.pbs").write_text(render_pbs(m, folder, request["job_name"]))
    return dict(folder=str(folder), job_name=request["job_name"])


def register_final_tests(m, state):
    for s in list(state.values()):
        if s["phase"] != "train" or s["status"] != "DONE":
            continue
        proof = read(task_root(m, "train", s["arm"]) / "receipt.json")
        need(proof["status"] == "PASS" and proof["identity"] == m["identity"]
             and proof["selection"]["identity"] == m["identity"]
             and proof["selection"]["arm"] == s["arm"], "Missing validation selection for final test")
        state.setdefault("test:" + s["arm"], dict(phase="test", arm=s["arm"], status="READY", job=None, pending=None))


def qualification_passed(m, arm):
    q = read(task_root(m, "qualification", arm) / "receipt.json")
    need(q.get("status") == "PASS" and q.get("identity") == m["identity"]
         and q.get("scope") == "qualification" and q.get("arm") == arm
         and q.get("step") == 8 and q.get("resumed_from") == 4,
         "GPU end-to-end qualification/reload is required before formal training: " + arm)


def advance_pipeline(m, state):
    """Parallel qualification chains, then training after every requested arm passes."""
    path = Path(m["manifest_path"]).parent / "pipeline.json"
    if not path.exists():
        return
    p = read(path)
    need(p["schema"] == "d4-mopd-pipeline-v1" and p["identity"] == m["identity"]
         and p["arms"] == [a for a in ARMS if a in p["arms"]] and p["arms"],
         "Pipeline identity or arm order changed")
    for arm in p["arms"]:
        key = "qualification:" + arm
        s = state.setdefault(key, dict(phase="qualification", arm=arm, status="READY", job=None, pending=None))
        need(s["phase"] == "qualification" and s["arm"] == arm, "Pipeline qualification task changed")
    # Register every chain before checking the barrier. setdefault preserves
    # running jobs and ambiguous submissions when an existing pipeline is resumed.
    all_passed = True
    for arm in p["arms"]:
        s = state["qualification:" + arm]
        if s["status"] != "DONE":
            all_passed = False  # BLOCKED/unknown is not a successful prerequisite.
        else:
            qualification_passed(m, arm)
    if not all_passed:
        return
    for arm in p["arms"]:
        state.setdefault("train:" + arm, dict(phase="train", arm=arm, status="READY", job=None, pending=None))


def tick(m, state, pbs):
    if (Path(m["manifest_path"]).parent / "STOP").exists():
        return state
    # Also repairs a restart between recording train success and registering test.
    advance_pipeline(m, state)
    register_final_tests(m, state)
    for key, s in list(state.items()):
        if s["status"] in ("DONE", "BLOCKED"):
            continue
        run = task_root(m, s["phase"], s["arm"])
        try:
            job = s.get("job")
            if not job:
                if (Path(m["manifest_path"]).parent / "STOP").exists():
                    break
                # Durable pending directory precedes qsub. An ambiguous qsub is
                # recovered at the same path, never retried with a new directory.
                pending = s.get("pending")
                if pending is None:
                    pending = new_attempt(m, s["phase"], s["arm"])
                    s["pending"] = pending
                    write(Path(m["manifest_path"]).parent / "states.json", state)
                s["job"] = pbs.submit(pending["folder"], pending["job_name"])
                s["pending"] = None
                s["status"] = "SUBMITTED"
                continue
            row = pbs.observe(job)  # validates owner, queue, job name and exact script path
            s["pbs"] = {k: row.get(k) for k in ("job_state", "Exit_status", "comment")}
            s.pop("observation_error", None)
            if not finished(row):
                s["status"] = row["job_state"]
                continue
            result = failure_result(job)
            if str(row.get("Exit_status")) == "0" and result.get("status") == "PASS":
                need(result.get("identity") == m["identity"], "Unbound success receipt")
                if s["phase"] == "qualification":
                    need(result.get("resumed_from") == 4 and result["step"] == 8 and result.get("arm") == s["arm"],
                         "Missing arm-bound fresh-process resume proof")
                elif s["phase"] == "train":
                    need(result["step"] == m["spec"]["updates"] and result["selection"]["identity"] == m["identity"]
                         and result["selection"]["arm"] == s["arm"], "Missing endpoint/selection")
                else:
                    expected = read(Path(job["folder"]) / "request.json")["selection"]
                    test = read(result["test_receipt"])
                    need(result["selection"] == expected and result["step"] == expected["step"]
                         and test["status"] == "PASS" and test["split"] == "test"
                         and test["identity"] == m["identity"] and test["arm"] == s["arm"]
                         and test["step"] == expected["step"]
                         and test["checkpoint_identity"] == expected["checkpoint_identity"], "Unbound final test result")
                proof = dict(result, pbs_job=job, pbs=row, scope=s["phase"], arm=s["arm"], at_utc=now())
                write(run / "receipt.json", proof)
                s["status"] = "DONE"
            elif str(row.get("Exit_status")) == "0" and result.get("status") == "RESUME_REQUIRED":
                need(result.get("identity") == m["identity"] and s["phase"] == "qualification"
                     and result.get("step") == 4 and result.get("arm") == s["arm"], "Unexpected pause")
                s.update(status="READY", previous_job=job, job=None)
            elif str(row.get("Exit_status")) == "0" and result.get("status") == "CONTROL_PAUSED":
                need(result.get("identity") == m["identity"], "Unbound pause receipt")
                s.update(status="READY", previous_job=job, job=None)
            else:
                reason = retry_reason(row, job)
                if reason and retry_allowed(row, result, job, m["identity"]):
                    s.update(status="READY", previous_job=job, job=None, retry_reason=reason,
                             external_interruptions=s.get("external_interruptions", 0)+1)
                else:
                    s.update(status="BLOCKED", error=result.get("error") or "Unclassified terminal job; inspect logs")
        except Unknown as exc:
            s["observation_error"] = str(exc)  # Keep job/pending; no duplicate qsub.
        except Exception as exc:
            s.update(status="BLOCKED", error=str(exc))
        finally:
            s["observed_utc"] = now()
            write(Path(m["manifest_path"]).parent / "states.json", state)
    advance_pipeline(m, state)
    register_final_tests(m, state)
    write(Path(m["manifest_path"]).parent / "states.json", state)
    return state


def guardian(manifest_path):
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    with ExitStack() as lease:
        lease.enter_context(locked(root / "guardian.lock"))
        while True:
            if (root / "STOP").exists():
                with lock(root / "launch.lock", wait=60):
                    if (root / "STOP").exists():
                        lease.close()
                        return
            with lock(root / "states.lock", wait=120):
                states = tick(m, read(root / "states.json"), PBS())
                done = all(s["status"] in ("DONE", "BLOCKED") for s in states.values())
                write(root / "heartbeat.json", dict(at_utc=now(), host=socket.gethostname(), pid=os.getpid(),
                                                     states={k: v["status"] for k, v in states.items()}, all_terminal=done))
            if done:
                # Serialize retirement with registration. A start on another login
                # node cannot see an old guardian lock and then lose its guardian.
                with lock(root / "launch.lock", wait=60):
                    with lock(root / "states.lock", wait=120):
                        current = read(root / "states.json")
                        if all(s["status"] in ("DONE", "BLOCKED") for s in current.values()):
                            lease.close()
                            return
            time.sleep(m["spec"]["poll_seconds"])


def start(manifest_path, phase, arms):
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    need(phase in ("pipeline", "qualification", "train", "test")
         and list(arms) == [a for a in ARMS if a in arms] and arms, "Invalid phase/arms")
    need(not (root / "STOP").exists(), "Experiment is stopped; use resume-control explicitly")
    proof = read(root / "cpu_preflight.json")
    need(proof["status"] == "PASS" and proof["identity"] == m["identity"], "CPU preflight has not passed")
    need(proof.get("native_abi", {}).get("status") == "PASS", "Repeat CPU preflight with the native GLIBC check")
    if phase == "train":
        for arm in arms:
            qualification_passed(m, arm)
    with lock(root / "launch.lock", wait=60):
        with lock(root / "states.lock", wait=120):
            path = root / "states.json"
            state = read(path) if path.exists() else {}
            if phase == "pipeline":
                path_plan = root / "pipeline.json"
                plan = dict(schema="d4-mopd-pipeline-v1", identity=m["identity"], arms=list(arms))
                if path_plan.exists():
                    need(read(path_plan) == plan, "Existing automatic pipeline has another identity/arms")
                else:
                    write(path_plan, plan)
                advance_pipeline(m, state)
            else:
                for arm in arms:
                    key = phase + ":" + arm
                    state.setdefault(key, dict(phase=phase, arm=arm, status="READY", job=None, pending=None))
            write(path, state)
        if not active(root / "guardian.lock"):
            with (root / "guardian.log").open("ab") as log:
                child = subprocess.Popen(entry_command(m) + ["guardian", "--manifest", str(manifest_path)],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=str(root))
            deadline = time.monotonic() + 30
            while not active(root / "guardian.lock") and time.monotonic() < deadline:
                need(child.poll() is None, "Guardian exited during startup; inspect guardian.log")
                time.sleep(0.1)
            need(active(root / "guardian.lock"), "Guardian did not acquire its lease; inspect guardian.log")
    return dict(status="REGISTERED", tasks=list(state), manifest=m["manifest_path"],
                automatic_pipeline=(root / "pipeline.json").exists(),
                note="Use status to confirm the guardian heartbeat and PBS submission")


def quiescent_qualifications(m, state, pbs, *, on_pbs_unknown=None):
    """Environment repair is only supported before qualification has progressed."""
    need(state and all(s["phase"] == "qualification" and s["status"] == "BLOCKED"
                       and s.get("job") and not s.get("pending") for s in state.values()),
         "Repair requires only BLOCKED qualification jobs, with no pending submissions")
    root = Path(m["manifest_path"]).parent

    def require_no_progress():
        for s in state.values():
            run = task_root(m, "qualification", s["arm"])
            need(not (run / "receipt.json").exists()
                 and not any((run / "checkpoints").glob("global_step_*")),
                 "Qualification already produced progress; environment replacement requires a new experiment")

    while True:
        need(not (root / "STOP").exists(), "Experiment STOP requested during PBS observation")
        require_no_progress()
        observations = {}
        for key, s in state.items():
            try:
                observations[key] = pbs.observe(s["job"])
            except Unknown as exc:
                if on_pbs_unknown is None:
                    raise
                on_pbs_unknown(s["job"], exc)
                break
            need(finished(observations[key]), "Qualification PBS job is still active: " + key)
        else:
            need(not (root / "STOP").exists(), "Experiment STOP requested during PBS observation")
            require_no_progress()
            return observations
        # Discard the partial pass: every job must be freshly observed before
        # changing the environment or rearming tasks. Never retry a mutation.


def repair_te(manifest_path, *, lease_fd=None, on_pbs_unknown=None):
    """Rebuild the pinned package without changing the frozen learner/runtime."""
    from .preflight import environment, native_abi, require_native_abi
    from .repair_pipeline import environment_lease
    import importlib.metadata
    import shutil
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    need(Path(sys.executable).absolute() == Path(m["python"]).absolute(), "Use the manifest interpreter")
    def packages():
        return {d.metadata["Name"].lower().replace("_", "-"): d.version
                for d in importlib.metadata.distributions()}
    # Shared locks, no PID signalling or original-login-host requirement.
    with environment_lease(m, lease_fd) as build_lease, lock(root / "launch.lock"), \
            locked(root / "guardian.lock"), lock(root / "states.lock"):
        states = read(root / "states.json") if (root / "states.json").exists() else {}
        observations = quiescent_qualifications(m, states, PBS(), on_pbs_unknown=on_pbs_unknown) if states else {}
        before = packages()
        versions = environment(gpu_packages=True)
        audit = root / "environment_repairs" / uuid.uuid4().hex
        audit.mkdir(parents=True)
        write(audit / "before.json", dict(identity=m["identity"], packages=before, states=states,
                                          pbs=observations, native_abi=native_abi(), at_utc=now()))
        script = Path(__file__).resolve().parents[1] / "scripts/mopd_bootstrap.sh"
        shutil.copy2(script, audit / "build.sh")
        proof = root / "cpu_preflight.json"
        if proof.exists():
            shutil.copy2(proof, audit / "previous_cpu_preflight.json")
        write(proof, dict(status="INVALIDATED", reason="Transformer Engine native rebuild", at_utc=now()))
        try:
            print("Native build log and audit: " + str(audit), flush=True)
            subprocess.run(["bash", str(audit / "build.sh"), m["verl_src"], "--rebuild-te", str(audit)],
                           check=True, pass_fds=(build_lease,))
            after = packages()
            need(all(after.get(k) == v for k, v in before.items()), "An existing dependency version changed")
            need(set(after) - set(before) <= {"setuptools", "wheel", "pybind11", "ninja"}, "Unexpected new dependency")
            need(environment(gpu_packages=True) == versions, "Pinned training environment changed")
            abi = native_abi()
            require_native_abi(abi)
            result = dict(status="PASS", identity=m["identity"], versions=versions, native_abi=abi,
                          build=read(audit / "build_receipt.json"), at_utc=now(),
                          next_step="Run preflight, then retry-qualification; no GPU test has passed yet")
            write(audit / "result.json", result)
            write(root / "environment_repair.json", dict(result, audit=str(audit)))
            return result
        except Exception as exc:
            write(audit / "result.json", dict(status="FAIL", error=str(exc), at_utc=now()))
            raise


def repair_megatron_import(manifest_path, *, lease_fd=None, on_pbs_unknown=None):
    """Apply the audited upstream FA4 guard before any learner progress exists."""
    import importlib.metadata
    import shutil
    from . import megatron_repair
    from .preflight import environment, native_abi, require_native_abi
    from .repair_pipeline import environment_lease
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    need(Path(sys.executable).absolute() == Path(m["python"]).absolute(), "Use the manifest interpreter")

    def packages():
        return {d.metadata["Name"].lower().replace("_", "-"): d.version
                for d in importlib.metadata.distributions()}

    with environment_lease(m, lease_fd), lock(root / "launch.lock"), \
            locked(root / "guardian.lock"), lock(root / "states.lock"):
        need(not (root / "STOP").exists(), "Experiment is stopped")
        states = read(root / "states.json") if (root / "states.json").exists() else {}
        observations = quiescent_qualifications(m, states, PBS(), on_pbs_unknown=on_pbs_unknown) if states else {}
        before, versions, abi = packages(), environment(gpu_packages=True), native_abi()
        require_native_abi(abi)
        audit = root / "environment_repairs" / uuid.uuid4().hex
        audit.mkdir(parents=True)
        write(audit / "before.json", dict(identity=m["identity"], packages=before, states=states,
                                          pbs=observations, native_abi=abi, at_utc=now()))
        proof = root / "cpu_preflight.json"
        if proof.exists():
            shutil.copy2(proof, audit / "previous_cpu_preflight.json")
        write(proof, dict(status="INVALIDATED", reason="MCore optional FA4 import backport", at_utc=now()))
        try:
            print("Megatron import repair audit: " + str(audit), flush=True)
            patch = megatron_repair.apply(audit)
            need(packages() == before and environment(gpu_packages=True) == versions,
                 "A dependency version changed during the import repair")
            after_abi = native_abi()
            require_native_abi(after_abi)
            need(after_abi["files"] == abi["files"], "Native binaries changed during the import repair")
            result = dict(status="PASS", identity=m["identity"], kind="megatron-import", patch=patch,
                          native_abi=after_abi, versions=versions, at_utc=now(),
                          next_step="CPU preflight then both qualifications; GPU tests have not passed yet")
            write(audit / "result.json", result)
            write(root / "megatron_import_repair.json", dict(result, audit=str(audit)))
            return result
        except Exception as exc:
            write(audit / "result.json", dict(status="FAIL", error=str(exc), at_utc=now()))
            raise


def retry_qualification(manifest_path, arms, *, on_pbs_unknown=None):
    """Explicit one-shot rearm after repair; ordinary start never clears BLOCKED."""
    from .preflight import environment, native_abi, require_native_abi
    from .megatron_repair import inspect as inspect_megatron_import
    m = verify(manifest_path)
    from .launcher_repair import active_runtime
    launcher_runtime = active_runtime(m)
    root = Path(manifest_path).parent
    need(list(arms) == [a for a in ARMS if a in arms] and arms, "Invalid arms")
    need(not (root / "STOP").exists(), "Experiment is stopped")
    with lock(root / "launch.lock"), locked(root / "guardian.lock"), lock(root / "states.lock"):
        proof = read(root / "cpu_preflight.json")
        need(proof.get("status") == "PASS" and proof.get("identity") == m["identity"]
             and proof.get("native_abi", {}).get("status") == "PASS", "Repeat the updated CPU preflight first")
        if (root / "launcher_repair.json").exists():
            need(proof.get("launcher_runtime") == str(launcher_runtime),
                 "Repeat CPU preflight after the launcher repair")
        need(environment(gpu_packages=True) == proof["versions"], "Environment changed after preflight")
        abi = native_abi()
        require_native_abi(abi)
        need(abi["files"] == proof["native_abi"]["files"], "Native binaries changed after preflight")
        need(proof.get("megatron_import_guard", {}).get("status") == "PASS",
             "Repeat CPU preflight with the Megatron FA4 import check")
        need(inspect_megatron_import() == proof["megatron_import_guard"],
             "Megatron import guard changed after preflight")
        state = read(root / "states.json")
        observations = quiescent_qualifications(m, state, PBS(), on_pbs_unknown=on_pbs_unknown)
        keys = ["qualification:" + a for a in arms]
        need(all(k in state for k in keys), "Requested qualification is not registered")
        audit = root / "qualification_retries" / (uuid.uuid4().hex + ".json")
        write(audit, dict(identity=m["identity"], before=state, pbs=observations,
                          cpu_preflight_sha256=file_sha(root / "cpu_preflight.json"), at_utc=now()))
        for key in keys:
            s = state[key]
            s.update(status="READY", previous_job=s["job"], job=None, pending=None,
                     previous_error=s.pop("error", None), explicit_retry=str(audit))
            s.pop("pbs", None)
        write(root / "states.json", state)
    # This launches the existing immutable guardian. The repaired binary has the
    # same package version; no manifest/runtime hash or experiment settings change.
    if (root / "pipeline.json").exists():
        return start(manifest_path, "pipeline", read(root / "pipeline.json")["arms"])
    return start(manifest_path, "qualification", arms)
