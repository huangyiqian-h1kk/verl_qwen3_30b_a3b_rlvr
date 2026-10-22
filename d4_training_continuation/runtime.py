"""One PBS allocation: five fresh-process qualifications, then formal training."""
import argparse
from collections import Counter
import copy
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
from . import PHASES, TOTAL_STEPS
from .common import digest, expired, locked, need, now, read, write
from .source import topology_config, validate_request
from .scheduler import PBS
from . import checkpoints as ck
from . import evaluation
from .controller import authoritative_checkpoint


def allocation(path):
    r = validate_request(path)
    folder = Path(r["job_dir"])
    need(not (folder / "allocation.json").exists(), "PBS rerun of the same candidate is forbidden")
    hosts = [h.strip().split('.')[0] for h in (folder / "hosts").read_text().splitlines() if h.strip()]
    need(len(set(hosts)) == len(hosts) == r["expected_nodes"], "Wrong PBS allocation size")
    job = dict(job_id=os.environ["PBS_JOBID"], job_name=r["job_name"], folder=str(folder))
    row = PBS().observe(job)
    need(row["job_state"] == "R" and row.get("run_count") and row.get("stime"), "Cannot bind PBS execution")
    write(folder / "allocation.json", dict(job_id=job["job_id"], hosts=hosts, execution_id=uuid.uuid4().hex,
          run_count=row["run_count"], stime=row["stime"], at_utc=now()))


def wait_gpus(nodes):
    import ray
    ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
    try:
        until = time.monotonic() + 300
        while ray.available_resources().get("GPU", 0) < nodes * 8:
            need(time.monotonic() < until, "Previous process did not release all allocated GPUs")
            time.sleep(2)
    finally:
        ray.shutdown()


def execute(path):
    r = validate_request(path)
    folder = Path(r["job_dir"])
    report = dict(status="RUNNING", job_id=os.environ["PBS_JOBID"], identity=r["identity"], started_utc=now())
    with locked(Path(r["run_dir"]) / "stage.lock"):
        write(folder / "result.json", report)
        try:
            from multidomain.preflight import check
            from d4_two_node_resumefix.core import check_data_counts
            # The immutable data manifest is bound to the original single-node
            # build config, not the derived allocation/phase topology config.
            check(r["contract"]["base_config"], Path(r["data_dir"]), models=True)
            check_data_counts(r["data_dir"])
            subprocess.run([sys.executable, "-m", "multidomain.check_infra", "--expected-nodes",
                            str(r["expected_nodes"]), "--output", str(folder / "preflight/infrastructure.json")],
                           cwd=r["root"], check=True)
            names = [x[0] for x in PHASES] if r["role"] == "candidate" else ["train"]
            for name in names:
                wait_gpus(r["expected_nodes"])
                subprocess.run([sys.executable, "-I", str(folder / "entry.py"), "phase", "--request", str(path),
                                "--phase-name", name], cwd=r["root"], check=True)
                if name != "train":
                    need(read(folder / name / "qualification.json")["status"] == "PASS", "Incomplete phase")
                ck.remove_pin(folder / name / "source")
            report["status"] = "PASS"
        except BaseException as e:
            report.update(status="FAIL", error=type(e).__name__ + ": " + str(e))
            raise
        finally:
            report["finished_utc"] = now()
            write(folder / "result.json", report)


def select_source(r, name):
    folder = Path(r["job_dir"])
    if name == "A":
        state = read(Path(r["control_dir"]) / "state.json")
        from .opportunities import authorized
        admitted = authorized(state, r) if r.get("opportunity_id") else not state.get("promoted")
        need(state["operation"] == r["contract"]["operation"] and admitted
             and not state.get("handoff_failed") and not expired(state["deadline_utc"]), "Candidate is no longer authorized")
        # Current formal single, captured at actual startup. A queued candidate
        # must not resume the stale checkpoint that existed at submission time.
        source = authoritative_checkpoint(state["authority"], r["kind"])
        need(source["world_size"] == 8 and source["step"] + 5 <= TOTAL_STEPS, "Five qualification steps unavailable")
    elif name == "train":
        need(r["role"] == "formal", "Wrong formal execution role")
        source = authoritative_checkpoint(r["source"], r["kind"])
    else:
        names = [x[0] for x in PHASES]
        previous = names[names.index(name) - 1]
        proof = read(folder / previous / "qualification.json")
        need(proof["status"] == "PASS" and proof["identity"] == r["identity"], "Previous phase failed")
        source = proof["checkpoint"]
    return ck.pin(source, folder / name / "source")


def phase(path, name):
    r = validate_request(path)
    folder = Path(r["job_dir"])
    expected = [x[0] for x in PHASES] if r["role"] == "candidate" else ["train"]
    need(name in expected, "Unrequested phase")
    phase_dir = folder / name
    phase_dir.mkdir()
    # A retention race is safe to retry BEFORE constructing any model.
    for attempt in range(5):
        try:
            resume = select_source(r, name)
            break
        except FileNotFoundError:
            if attempt == 4:
                raise
            time.sleep(2)
    target_world = r["expected_nodes"] * 8 if name == "train" else dict((n, w) for n, _, w in PHASES)[name]
    if name != "train":
        need(resume["world_size"] == dict((n, w) for n, w, _ in PHASES)[name], "Wrong source world for phase")
    need(0 < resume["step"] < TOTAL_STEPS, "No remaining training steps")
    execution = dict(phase=name, token=uuid.uuid4().hex, identity=r["identity"], driver_pid=os.getpid(),
                     execution_id=read(folder / "allocation.json")["execution_id"],
                     source_world=resume["world_size"], target_world=target_world, source_step=resume["step"])
    write(phase_dir / "phase.json", execution)
    write(phase_dir / "resume_source.json", resume)
    from omegaconf import OmegaConf
    from multidomain.runtime_config import compose
    cfg_source = topology_config(r["config"], target_world // 8)
    cfg = compose(cfg_source, Path(r["data_dir"]), Path(r["run_dir"]), "train")
    if r.get("training_policy"):
        from .save_policy import POLICY
        need(cfg.trainer.save_freq == POLICY["save_freq"] and cfg.trainer.test_freq == POLICY["test_freq"],
             "Composed runtime lost the save5/eval25 policy")
    cfg.trainer.val_only = False
    cfg.trainer.val_before_train = bool(r.get("training_policy") and
                                       evaluation.due(resume["step"], cfg.trainer.test_freq))
    cfg.trainer.max_actor_ckpt_to_keep = None
    cfg.trainer.del_local_ckpt_after_load = False
    cfg.trainer.resume_mode = "resume_path"
    cfg.trainer.resume_from_path = resume["path"]
    # Replay/reuse a scheduled source evaluation before the next training update.
    # Checkpoint persistence remains 5; validation remains 25.
    if r["kind"] == "domain":
        cfg.data.shuffle = False
    cfg.reward.custom_reward_function.path = r["reward"]["import_path"]
    cfg.reward.custom_reward_function.name = "compute_score"
    env = dict(PYTHONPATH=os.pathsep.join([str(folder / "runtime"), str(folder), r["root"], os.environ.get("PYTHONPATH", "")]),
               CONT_JOB=str(folder), CONT_PHASE_DIR=str(phase_dir), MD2_PHASE_DIR=str(phase_dir),
               MD2_AUDIT_STATE="0", MD_RUN_DIR=r["run_dir"], MD_STAGE="train")
    os.environ.update(env)
    for key, value in env.items():
        OmegaConf.update(cfg, "ray_kwargs.ray_init.runtime_env.env_vars." + key, value, force_add=True)
    from .optimizer_repair import configure
    configure(cfg)
    OmegaConf.save(cfg, phase_dir / "resolved.yaml", resolve=True)
    run_trainer(cfg, r, phase_dir, resume, target_world)


def run_trainer(config, request, phase_dir, resume, world):
    import ray
    from omegaconf import OmegaConf
    from torch.utils.data import SequentialSampler
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.main_ppo_v0 import BaseTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
    from verl.trainer.ppo.utils import create_rl_sampler
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from multidomain.dataset import MultiDomainDataset
    from .worker import ContinuationActor
    from .trainer import trainer_class, PhaseComplete
    from .evidence import topology

    @ray.remote
    class ContinuationRunner(BaseTaskRunner):
        def run(self, config):
            OmegaConf.resolve(config)
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(ContinuationActor)
            self.mapping[Role.ActorRollout] = "global_pool"
            self.add_reward_model_resource_pool(config)
            tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
            original = MultiDomainDataset(config.data.train_files, tokenizer, config.data)
            plan, rows = None, None
            if request["kind"] == "domain":
                from d4_domain_batch.sampling import PlannedDataset, build_plan, validate_plan
                rows = original.rows
                need(Counter(r["domain"] for r in rows) == Counter({d: 10000 for d in (
                    "math", "if", "conversational_pivot", "logic_algorithmic")}), "Domain prompt counts changed")
                plan = build_plan(rows, seed=request["config"]["seed"])
                need(validate_plan(plan, rows) == request["sampling"] == read(Path(resume["path"]) / "sampling.json"),
                     "Domain sampling plan changed")
                train_ds = PlannedDataset(original, plan)
                sampler = SequentialSampler(train_ds)
            else:
                train_ds = original
                sampler = create_rl_sampler(config.data, train_ds)
            val_ds = MultiDomainDataset(config.data.val_files, tokenizer, config.data)
            trainer = trainer_class(RayPPOTrainer)(config=config, tokenizer=tokenizer, processor=None,
                role_worker_mapping=self.role_worker_mapping, resource_pool_manager=self.init_resource_pool_mgr(config),
                ray_worker_group_cls=RayWorkerGroup, train_dataset=train_ds, val_dataset=val_ds,
                collate_fn=collate_fn, train_sampler=sampler)
            trainer.init_workers()
            trainer.actor_rollout_wg.configure_checkpoints(trainer.total_training_steps)
            actors = topology(trainer.actor_rollout_wg.continuation_topology(), world)
            replicas = len(trainer.llm_server_manager.get_replicas())
            need(replicas == world // 4, "Wrong TP4 rollout replica count")
            write(phase_dir / "topology.json", dict(actor_ranks=actors, rollout_replicas=replicas,
                  runner_pid=os.getpid(), runner_host=socket.gethostname().split('.')[0]))
            trainer.configure_continuation(request, phase_dir, resume, plan, rows)
            try:
                try:
                    trainer.fit()
                except PhaseComplete:
                    need(phase_dir.name in ("A", "B", "C", "D")
                         and read(phase_dir / "qualification.json")["status"] == "PASS", "Premature phase exit")
                    return
            finally:
                trainer._shutdown_dump_executor()
            need((phase_dir / "step_468.json").is_file(), "Training ended before the full step budget")
            write(phase_dir / "result.json", dict(status="PASS", completed_step=468))
    run_ppo(config, ContinuationRunner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("allocation", "execute", "phase"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--phase-name")
    a = parser.parse_args(argv)
    if a.action == "phase":
        phase(a.request, a.phase_name)
    else:
        globals()[a.action](a.request)
