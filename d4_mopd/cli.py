"""Prepare, inspect and supervise qualification -> train -> test for plain MOPD."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import ARMS
from .assets import prepare, verify
from . import control
from d4_teachers.common import active, need, now, read, write
from d4_training_continuation.common import locked


def execute(request_path):
    from omegaconf import OmegaConf
    import ray
    from verl.trainer.main_ppo import run_ppo
    from .integration import MOPDTaskRunner
    from .config import compose
    from .preflight import environment, validate_native_config
    r = read(request_path)
    m = verify(r["manifest"])
    need(m["identity"] == r["identity"], "Job/manifest identity mismatch")
    folder, root = Path(r["folder"]), Path(r["manifest"]).parent
    os.environ["D4_MOPD_ATTEMPT"] = folder.name
    need(os.environ.get("PBS_JOBID"), "execute requires its registered PBS allocation")
    if (root / "STOP").exists():
        write(folder / "result.json", dict(status="CONTROL_PAUSED", identity=m["identity"]))
        return
    job = dict(job_id=os.environ["PBS_JOBID"], job_name=r["job_name"], folder=str(folder))
    row = control.PBS().observe(job)
    need(row["job_state"] == "R", "Job is not running")
    need(environment(gpu_packages=True) == read(root / "cpu_preflight.json")["versions"], "Environment changed after preflight")
    with locked(Path(r["run_dir"]) / "writer.lock"):
        write(folder / "result.json", dict(status="RUNNING", identity=m["identity"], at_utc=now()))
        try:
            ray.init(address=os.environ["RAY_ADDRESS"], runtime_env={"env_vars": {
                "PYTHONPATH": os.environ["PYTHONPATH"], "D4_MOPD_ATTEMPT": folder.name,
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}})
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) == 8]
                if len(nodes) == 2:
                    break
                time.sleep(2)
            else:
                raise RuntimeError("Both eight-GPU Ray nodes did not register")
            for rank in (0, 1):
                proof = read(folder / f"gpu_node_{rank}.json")
                need(proof["status"] == "PASS" and proof["identity"] == m["identity"], "GPU node preflight missing")
            # Leave incomplete saves as evidence, and make the retry path empty.
            for p in (Path(r["run_dir"]) / "checkpoints").glob("global_step_*"):
                if p.is_dir() and not (p / "mopd_complete.json").exists():
                    dest = folder / "incomplete_checkpoints" / p.name
                    dest.parent.mkdir(exist_ok=True)
                    p.rename(dest)
            cfg = compose(m, r["arm"], r["run_dir"], qualification=r["phase"] == "qualification",
                          test_checkpoint=r.get("selection") if r["phase"] == "test" else None)
            need((r["phase"] == "test") == (cfg.mopd.test_checkpoint is not None), "Test selection missing")
            cfg.trainer.validation_data_dir = str(folder / "validation_generations")
            validate_native_config(cfg)
            OmegaConf.save(cfg, folder / "resolved.yaml")
            run_ppo(cfg, MOPDTaskRunner)
            result = read(folder / "result.json")
            need(result["status"] in ("PASS", "RESUME_REQUIRED", "CONTROL_PAUSED"),
                 "Native driver returned without a result")
        except BaseException as exc:
            write(folder / "result.json", dict(status="FAIL", identity=m["identity"], error=str(exc),
                                                traceback=traceback.format_exc(), at_utc=now()))
            raise
        finally:
            ray.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--teacher-campaign", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--verl-src", required=True)
    create.add_argument("--python", required=True, help="New upstream uv environment; never the teacher environment")
    create.add_argument("--spec")
    for name in ("preflight", "guardian", "status", "stop", "resume-control", "start", "repair-te", "retry-qualification",
                 "repair-and-retry", "repair-worker"):
        q = sub.add_parser(name)
        q.add_argument("--manifest", required=True)
        if name == "start":
            q.add_argument("--phase", choices=("pipeline", "qualification", "train", "test"), required=True)
        if name in ("start", "retry-qualification"):
            q.add_argument("--arm", choices=(*ARMS, "both"), default="both")
        if name == "repair-worker":
            q.add_argument("--repair-folder", required=True)
            q.add_argument("--lease-fd", type=int, required=True)
    for name in ("execute", "gpu-node", "paths"):
        q = sub.add_parser(name)
        q.add_argument("--request", required=True)
        if name == "gpu-node":
            q.add_argument("--rank", type=int, required=True)
    a = p.parse_args()
    if a.command == "prepare":
        result = prepare(a.teacher_campaign, a.output, a.verl_src, a.python, a.spec)
    elif a.command == "paths":
        r = read(a.request)
        m = read(r["manifest"])
        print(str(Path(r["manifest"]).parent / "runtime") + os.pathsep + m["verl_src"])
        return
    elif a.command == "execute":
        execute(a.request)
        return
    elif a.command == "gpu-node":
        from .preflight import gpu_node
        r = read(a.request)
        result = gpu_node(r["manifest"], r["folder"], a.rank)
    else:
        path = Path(a.manifest).resolve()
        root = path.parent
        if a.command == "preflight":
            # Prefer the exact pinned checkout even if an unrelated verl is installed.
            sys.path.insert(0, read(path)["verl_src"])
            from .preflight import cpu
            result = cpu(path)
        elif a.command == "repair-te":
            result = control.repair_te(path)
        elif a.command == "repair-and-retry":
            from .repair_pipeline import start
            result = start(path)
        elif a.command == "repair-worker":
            from .repair_pipeline import worker
            worker(path, a.repair_folder, a.lease_fd)
            return
        elif a.command == "retry-qualification":
            result = control.retry_qualification(path, ARMS if a.arm == "both" else (a.arm,))
        elif a.command == "guardian":
            control.guardian(path)
            return
        elif a.command == "start":
            arms = ARMS if a.arm == "both" else (a.arm,)
            result = control.start(path, a.phase, arms)
        elif a.command == "stop":
            write(root / "STOP", dict(at_utc=now(), mode="stop-new-submissions-and-next-update"))
            result = {"status": "STOP_REQUESTED", "existing_allocations": "not cancelled; observe status"}
        elif a.command == "resume-control":
            m = verify(path)
            (root / "STOP").unlink(missing_ok=True)
            if (root / "pipeline.json").exists():
                control.start(path, "pipeline", read(root / "pipeline.json")["arms"])
            else:
                states = read(root / "states.json")
                for phase in ("qualification", "train", "test"):
                    arms = [a for a in ARMS if phase+":"+a in states]
                    if arms:
                        control.start(path, phase, arms)
            result = {"status": "CONTROL_RESUMED", "blocked_jobs": "remain blocked for inspection"}
        else:
            from .repair_pipeline import status as repair_status
            result = {"manifest": str(path), "guardian_lock_held": active(root / "guardian.lock"),
                      "stopped": (root / "STOP").exists(),
                      "repair_pipeline": repair_status(path),
                      "pipeline": read(root / "pipeline.json") if (root / "pipeline.json").exists() else None,
                      "checkpoint_retention": {arm: read(root / "runs" / arm / "retention.json")
                          for arm in ARMS if (root / "runs" / arm / "retention.json").exists()},
                      "heartbeat": read(root / "heartbeat.json") if (root / "heartbeat.json").exists() else None,
                      "states": read(root / "states.json") if (root / "states.json").exists() else {}}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
