"""One physical eight-GPU node per train/export/test allocation."""

import argparse
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import uuid
from . import DOMAINS
from .common import below, is_fenced, lock, model_inventory, need, now, read, write
from .source import validate_request, load_campaign
from . import selection
from d4_training_continuation import checkpoints as ck
from d4_training_continuation.scheduler import PBS
from d4_training_continuation.runtime import wait_gpus


def allowed(r):
    need(not is_fenced(r["campaign_dir"], r["domain"]), "Teacher is durably stopped")


def fixed_recipe(cfg, r):
    from omegaconf import OmegaConf

    cfg.data.shuffle = False
    for key, value in {
        "data.validation_shuffle": False,
        "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.2,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.optim.lr_decay_style": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.megatron.seed": r["policy"]["seed"],
        "actor_rollout_ref.ref.megatron.seed": r["policy"]["seed"],
        "actor_rollout_ref.rollout.seed": r["policy"]["seed"],
        "trainer.del_local_ckpt_after_load": False,
    }.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    cfg.trainer.max_actor_ckpt_to_keep = None


def allocation(path):
    r = validate_request(path)
    allowed(r)
    folder = Path(r["job_dir"])
    need(
        not (folder / "allocation.json").exists(),
        "Re-running the same PBS allocation is forbidden",
    )
    hosts = (folder / "hosts").read_text().split()
    need(len(hosts) == 1, "Teacher needs exactly one physical node")
    row = PBS().observe(
        dict(folder=str(folder), job_id=os.environ["PBS_JOBID"], job_name=r["job_name"])
    )
    need(
        row["job_state"] == "R" and row.get("run_count") and row.get("stime"),
        "Allocation cannot be bound",
    )
    write(
        folder / "allocation.json",
        dict(
            job_id=os.environ["PBS_JOBID"],
            hosts=hosts,
            execution_id=uuid.uuid4().hex,
            run_count=row["run_count"],
            stime=row["stime"],
            at_utc=now(),
        ),
    )


def cpu_checks(path):
    r = validate_request(path)
    from .sampling import loader_restart_check
    from d4_training_continuation.validation_journal import check_source
    from d4_training_continuation.optimizer_repair import verify_manager_source

    src = Path(r["config"]["verl_src"])
    verify_manager_source(
        src / "verl/utils/checkpoint/megatron_checkpoint_manager.py",
        "Installed manager changed",
    )
    check_source(src / "verl/trainer/ppo/ray_trainer.py")
    c = load_campaign(r["campaign_dir"], verify=False)
    parent_data = Path(c["policy"]["parent_root"]) / c["policy"]["parent_data"]
    need(
        read(parent_data / "manifest.json") == c["contract"]["parent_data_manifest"],
        "Parent manifest changed",
    )
    from multidomain.preflight import check

    check(c["contract"]["parent_data_config"], parent_data, models=True)
    need(
        model_inventory(r["base_model"]) == c["contract"]["base_inventory"],
        "Frozen base HF model changed",
    )
    sampler = loader_restart_check()
    from d4_db_mathfix.selfcheck import main as check_math

    check_math()  # Real isolated backend; CPU, no model weights/GPU allocation.
    from omegaconf import OmegaConf
    from multidomain.runtime_config import compose

    cfg = compose(r["config"], Path(r["data_dir"]), Path(r["run_dir"]), "train")
    fixed_recipe(cfg, r)
    OmegaConf.save(cfg, Path(r["job_dir"]) / "cpu_resolved.yaml", resolve=True)
    need(
        cfg.trainer.total_training_steps == r["policy"]["max_steps"]
        and cfg.trainer.test_freq == 10
        and cfg.trainer.save_freq == 5,
        "Composed teacher schedule changed",
    )
    report = dict(
        status="PASS",
        campaign_identity=r["campaign_identity"],
        sampler=sampler,
        verifier_identity=r["verifier_identity"],
        model_weights_loaded=False,
        at_utc=now(),
    )
    write(Path(r["job_dir"]) / "cpu_preflight.json", report)
    return report


def execute(path):
    r = validate_request(path)
    allowed(r)
    folder = Path(r["job_dir"])
    report = dict(
        status="RUNNING", identity=r["identity"], role=r["role"], started_utc=now()
    )
    with lock(Path(r["run_dir"]) / "writer.lock"):
        write(folder / "result.json", report)
        try:
            cleanup_interrupted(r)
            cpu_checks(path)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "multidomain.check_infra",
                    "--expected-nodes",
                    "1",
                    "--output",
                    str(folder / "infrastructure.json"),
                ],
                cwd=r["root"],
                check=True,
            )
            if r["role"] == "export":
                selected = selection.validate_best(
                    read(Path(r["run_dir"]) / "best.json")
                )
                need(
                    selected == r["selected"], "Selected teacher changed before export"
                )
                if (Path(r["run_dir"]) / "export.json").exists():
                    from .exporting import validate_export, retire_native_best

                    exported = validate_export(read(Path(r["run_dir"]) / "export.json"))
                    need(
                        exported["selected"] == selected,
                        "Export belongs to another selection",
                    )
                    retire_native_best(r["run_dir"], exported)
                    report["status"] = "PASS"
                    return
                if selected["step"] == 0:
                    write(
                        folder / "export_candidate.json",
                        dict(path=r["base_model"], selected=selected),
                    )
                else:
                    wait_gpus(1)
                    subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            str(folder / "entry.py"),
                            "phase",
                            "--request",
                            str(path),
                        ],
                        check=True,
                    )
                wait_gpus(1)
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(folder / "entry.py"),
                        "load-check",
                        "--request",
                        str(path),
                    ],
                    check=True,
                )
            else:
                wait_gpus(1)
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(folder / "entry.py"),
                        "phase",
                        "--request",
                        str(path),
                    ],
                    check=True,
                )
            allowed(r)
            report["status"] = "PASS"
        except BaseException as e:
            report.update(status="FAIL", error=type(e).__name__ + ": " + str(e))
            raise
        finally:
            report["finished_utc"] = now()
            write(folder / "result.json", report)


def cleanup_interrupted(r):
    """Called only under the exclusive run writer lease, before worker creation."""
    run = Path(r["run_dir"])
    for parent, pattern in (
        (run / "checkpoints", ".staging_*"),
        (run / "best_weights", ".pin_*"),
    ):
        for path in parent.glob(pattern):
            shutil.rmtree(below(path, parent))
    for path in (run / "jobs").glob("export_*/export_candidate"):
        if path.parent != Path(r["job_dir"]):
            shutil.rmtree(below(path, run / "jobs"))


def phase(path):
    r = validate_request(path)
    allowed(r)
    folder, run = Path(r["job_dir"]), Path(r["run_dir"])
    phase_dir = folder / "train"  # Required by the pinned full-restore/audit context.
    phase_dir.mkdir()
    resume = ck.latest(run, r["lineage"]) if r["role"] == "train" else None
    resume = resume or dict(
        step=0, files={}, world_size=8, path=r["base_model"], lineage=None
    )
    if r["role"] == "test":
        resume["step"] = r["evaluation_model"]["step"]
    write(phase_dir / "resume_source.json", resume)
    write(
        phase_dir / "phase.json",
        dict(
            phase="train",
            identity=r["identity"],
            token=uuid.uuid4().hex,
            source_world=8,
            target_world=8,
            source_step=resume["step"],
            driver_pid=os.getpid(),
            execution_id=read(folder / "allocation.json")["execution_id"],
        ),
    )
    from omegaconf import OmegaConf
    from multidomain.runtime_config import compose

    model = r["evaluation_model"]["path"] if r["role"] == "test" else None
    cfg = compose(r["config"], Path(r["data_dir"]), run, "train", model_path=model)
    fixed_recipe(cfg, r)
    cfg.trainer.resume_mode = (
        "resume_path" if r["role"] == "train" and resume["step"] else "disable"
    )
    cfg.trainer.resume_from_path = (
        resume["path"] if cfg.trainer.resume_mode == "resume_path" else None
    )
    stopped = (
        read(run / "validation_state.json").get("early_stop", False)
        if (run / "validation_state.json").exists()
        else False
    )
    cfg.trainer.val_only = r["role"] == "test" or (
        r["role"] == "train" and (stopped or resume["step"] >= r["policy"]["max_steps"])
    )
    cfg.trainer.val_before_train = r["role"] == "test" or (
        r["role"] == "train" and resume["step"] % 10 == 0
    )
    cfg.trainer.experiment_name = r["policy"]["campaign"] + "_" + r["domain"]
    if r["role"] != "train":
        cfg.trainer.logger = ["console"]
    if r["role"] == "test":
        cfg.data.val_files = [
            str(Path(r["campaign_dir"]) / "data" / d / "test.parquet") for d in DOMAINS
        ]
    cfg.reward.custom_reward_function.path = r["reward"]["import_path"]
    cfg.reward.custom_reward_function.name = "compute_score"
    env = dict(
        PYTHONPATH=os.pathsep.join(
            [
                str(folder / "runtime"),
                str(folder),
                str(folder / "runtime/compat"),
                r["config"]["verl_src"],
            ]
        ),
        CONT_JOB=str(folder),
        CONT_PHASE_DIR=str(phase_dir),
        MD2_PHASE_DIR=str(phase_dir),
        MD2_AUDIT_STATE="0",
        MD_RUN_DIR=str(run),
        MD_STAGE="evaluate" if r["role"] == "test" else "train",
    )
    os.environ.update(env)
    for key, value in env.items():
        OmegaConf.update(
            cfg,
            "ray_kwargs.ray_init.runtime_env.env_vars." + key,
            value,
            force_add=True,
        )
    from d4_training_continuation.optimizer_repair import configure

    configure(cfg)
    OmegaConf.save(cfg, phase_dir / "resolved.yaml", resolve=True)
    run_trainer(cfg, r, phase_dir, resume)


def run_trainer(config, request, phase_dir, resume):
    import ray
    from omegaconf import OmegaConf
    from torch.utils.data import SequentialSampler
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.main_ppo_v0 import BaseTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from .dataset import TeacherDataset
    from .sampling import build_plan, PlannedDataset
    from .worker import TeacherActor
    from .trainer import trainer_class, TeacherComplete
    from d4_training_continuation.evidence import topology

    @ray.remote
    class TeacherRunner(BaseTaskRunner):
        def run(self, config):
            OmegaConf.resolve(config)
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(TeacherActor)
            self.mapping[Role.ActorRollout] = "global_pool"
            self.add_reward_model_resource_pool(config)
            tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
            original = TeacherDataset(config.data.train_files, tokenizer, config.data)
            domain = request["domain"] if request["domain"] != "base" else "math"
            plan = build_plan(
                original.rows,
                domain,
                seed=request["policy"]["seed"],
                total_steps=request["policy"]["max_steps"],
            )
            ds = PlannedDataset(original, plan)
            val = TeacherDataset(config.data.val_files, tokenizer, config.data)
            split = "test" if request["role"] == "test" else "validation"
            domains = DOMAINS if request["role"] == "test" else (domain,)
            expected_ids = [
                sid
                for d in domains
                for sid in request["data_manifest"]["domains"][d][split]["ids"]
            ]
            need(
                [row["sample_id"] for row in val.rows] == expected_ids,
                "Evaluation split differs from frozen manifest",
            )
            need(
                [row["sample_id"] for row in original.rows]
                == request["data_manifest"]["domains"][domain]["train"]["ids"],
                "Training rows differ from the frozen domain slice",
            )
            trainer = trainer_class(RayPPOTrainer)(
                config=config,
                tokenizer=tokenizer,
                processor=None,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.init_resource_pool_mgr(config),
                ray_worker_group_cls=RayWorkerGroup,
                train_dataset=ds,
                val_dataset=val,
                collate_fn=collate_fn,
                train_sampler=SequentialSampler(ds),
            )
            trainer.init_workers()
            trainer.actor_rollout_wg.configure_checkpoints(
                -1
            )  # Export only the selected best, in its own process.
            actors = topology(trainer.actor_rollout_wg.continuation_topology(), 8)
            need(
                len(trainer.llm_server_manager.get_replicas()) == 2,
                "Teacher requires two TP4 rollout replicas",
            )
            write(
                phase_dir / "topology.json",
                dict(
                    actor_ranks=actors,
                    rollout_replicas=2,
                    runner_host=socket.gethostname(),
                    sampling=plan.metadata(),
                ),
            )
            trainer.configure_teacher(request, phase_dir, resume, plan)
            try:
                if request["role"] == "export":
                    selected = selection.validate_best(request["selected"])
                    destination = Path(request["job_dir"]) / "export_candidate"
                    trainer.checkpoint_manager.sleep_replicas()
                    results = trainer.actor_rollout_wg.export_selected_teacher(
                        selected, str(destination)
                    )
                    need(
                        len(results) == 8
                        and all(x["status"] == "PASS" for x in results),
                        "Export ranks incomplete",
                    )
                    write(
                        Path(request["job_dir"]) / "export_candidate.json",
                        dict(
                            path=str(destination / "model/huggingface"),
                            selected=selected,
                        ),
                    )
                    return
                try:
                    trainer.fit()
                except TeacherComplete:
                    need(trainer.teacher_stop, "Premature teacher exit")
                if request["role"] == "train":
                    val_state = read(Path(request["run_dir"]) / "validation_state.json")
                    cp = ck.latest(request["run_dir"], request["lineage"])
                    need(
                        cp
                        and cp["step"] == val_state["step"]
                        and (
                            val_state["early_stop"]
                            or cp["step"] == request["policy"]["max_steps"]
                        ),
                        "Teacher ended without final validation",
                    )
                    selection.validate_best(
                        read(Path(request["run_dir"]) / "best.json")
                    )
                    write(
                        Path(request["run_dir"]) / "training_complete.json",
                        dict(
                            status="PASS",
                            completed_step=cp["step"],
                            early_stop=val_state["early_stop"],
                            lineage=request["lineage"],
                            validation=val_state,
                            at_utc=now(),
                        ),
                    )
                    ck.retain(request["run_dir"], request["lineage"], keep=1)
            finally:
                trainer._shutdown_dump_executor()

    run_ppo(config, TeacherRunner)


def load_check(path):
    """A fresh process loads every selected HF shard and executes a finite forward."""
    r = validate_request(path)
    allowed(r)
    need(r["role"] == "export", "Fresh load check requires an export job")
    from .exporting import finish_export

    finish_export(r)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("allocation", "execute", "phase", "cpu-checks", "load-check")
    )
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    globals()[args.action.replace("-", "_")](args.request)
