"""Pinned Megatron loader: rebuild live-parameter descriptors, retain all state."""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys
from .common import need, read, sha, verify_files, write
from .source import MANAGER_SHA
VERSION = "continuation-optimizer-v1"
TARGET = "verl.utils.checkpoint.megatron_checkpoint_manager"
HOOK = "d4_training_continuation.optimizer_repair.worker_setup"

def verify_manager_source(path, message):
    need(sha(path) == MANAGER_SHA, message + ": " + str(path))


def context():
    job = Path(os.environ["CONT_JOB"]).resolve()
    phase = Path(os.environ["CONT_PHASE_DIR"]).resolve()
    need(phase.parent == job and phase.name in ("A", "B", "C", "D", "E", "train"), "Wrong worker phase")
    need(Path(__file__).resolve().parent == job / "runtime/d4_training_continuation", "Worker imported another snapshot")
    request, execution = read(job / "request.json"), read(phase / "phase.json")
    need(execution["identity"] == request["identity"] and execution["phase"] == phase.name,
         "Phase contract mismatch")
    expected = {n: h for n, h in request["contract"]["snapshot_files"].items()
                if n.startswith("runtime/d4_training_continuation/") or n.startswith("audit_source/")}
    verify_files(job, expected)
    return job, phase, request, execution


def _wrap_load(original):
    @functools.wraps(original)
    def load(self, local_path, hdfs_path=None, del_local_after_load=False):
        job, phase, req, execution = context()
        world = execution["target_world"]
        need(world in (8, 16) and self.world_size == world and 0 <= self.rank < world, "Manager world mismatch")
        need(self.should_load_dist_ckpt_model and self.should_load_optimizer and self.should_load_extra,
             "Full model/optimizer/extra restore is mandatory")
        need(self.use_checkpoint_opt_param_scheduler and self.lr_scheduler is not None and not del_local_after_load,
             "Full LR restore and retained source are mandatory")
        selected = read(phase / "resume_source.json")
        path = Path(local_path).resolve()
        need(path == Path(selected["path"]).resolve() / "actor", "Worker source differs from driver selection")
        spec = read(path / "ckpt_contents.json")
        need(spec["world_size"] == selected["world_size"] == execution["source_world"]
             and spec["global_step"] == selected["step"], "Selected checkpoint metadata changed")
        result = original(self, local_path, hdfs_path, del_local_after_load)
        write(phase / f"optimizer_load_returned_rank_{self.rank}.json",
              dict(status="LOAD_RETURNED", version=VERSION, rank=self.rank, world_size=world,
                   token=execution["token"], checkpoint=str(path),
                   components=["model", "optimizer", "lr_scheduler", "rng_state"]))
        return result
    load._0390_full_restore_repair = VERSION
    return load


def configure(cfg):
    from omegaconf import OmegaConf
    _, _, _, execution = context()
    need(int(cfg.trainer.nnodes) * 8 == execution["target_world"], "Driver topology mismatch")
    need(list(cfg.actor_rollout_ref.actor.checkpoint.load_contents) == ["model", "optimizer", "extra"], "Partial restore refused")
    key = "ray_kwargs.ray_init.runtime_env.worker_process_setup_hook"
    need(OmegaConf.select(cfg, key) in (None, HOOK), "Existing Ray setup hook requires integration")
    OmegaConf.update(cfg, key, HOOK, force_add=True)
    OmegaConf.update(cfg, "actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler", True, force_add=True)
    OmegaConf.update(cfg, "trainer.del_local_ckpt_after_load", False, force_add=True)


def worker_setup():
    context()
    if TARGET in sys.modules:
        install(sys.modules[TARGET])
    elif not any(isinstance(x, _Finder) for x in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def worker_check(engine):
    context()
    cls = type(engine.checkpoint_mananager)
    need(getattr(cls._build_optimizer_state_dict, "_0390_optimizer_repair", None) == VERSION
         and getattr(cls.load_checkpoint, "_0390_full_restore_repair", None) == VERSION, "Worker loader hook missing")
    return VERSION

def _wrap_builder(original):
    @functools.wraps(original)
    def build(self, model_sharded_state_dict, metadata, is_loading=False):
        if is_loading and not self.use_megatron_fsdp:
            # load_dist_checkpointing can mutate replica_id and replace .data
            # with a detached tensor. Rebuild descriptors from the live model;
            # do not deep-copy Parameters or invent PP/TP/DP replica coordinates.
            model_sharded_state_dict = self._build_model_sharded_state_dict(metadata)
        return original(self, model_sharded_state_dict, metadata, is_loading=is_loading)
    build._0390_optimizer_repair = VERSION
    return build


def install(module):
    verify_manager_source(module.__file__, "Worker checkpoint manager source changed")
    need(module.megatron.core.__version__ == "0.13.1", "Worker Megatron version changed")
    cls = module.MegatronCheckpointManager
    current = cls._build_optimizer_state_dict
    marker = getattr(current, "_0390_optimizer_repair", None)
    if marker is not None:
        need(marker == VERSION, "Another optimizer repair is already installed")
        return
    cls._build_optimizer_state_dict = _wrap_builder(current)
    cls.load_checkpoint = _wrap_load(cls.load_checkpoint)


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        need(spec and spec.loader and spec.origin, "Cannot find worker checkpoint manager")
        verify_manager_source(spec.origin, "Worker checkpoint source differs from tested source")
        spec.loader = _Loader(spec.loader)
        return spec

