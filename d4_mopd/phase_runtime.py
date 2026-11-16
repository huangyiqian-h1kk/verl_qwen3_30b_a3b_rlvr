"""Whole-batch student/teacher multiplexing for the qualified single-node profile.

Importing this module needs only the standard library. Native adapters are
resolved lazily so CPU contract tests do not need Ray, CUDA, or Megatron. The
adapters retain native teacher scoring/padding and the entire native actor path.
"""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import struct
import threading
import time

from . import DOMAINS


PROFILE = "single_node"
TEACHER_RPC_LIMIT_PER_WORKER = 16
FIELDS = ("_mopd_phase_uid", "_mopd_phase_domain", "_mopd_phase_sha",
          "_mopd_phase_prompt_length", "_mopd_phase_response_length")
UPSTREAM_SHA256 = {
    "experimental/agent_loop/agent_loop.py": "99a131b86cb16d5d8aa9a773a644dcd628948941cbebcd007cd70d78e47f1e33",
    "experimental/teacher_loop/teacher_model.py": "371cf5075aa9b008c6e7084355ab01971328f9eba6e598b8ea5748d398edb729",
    "experimental/teacher_loop/teacher_manager.py": "485883f75919fb5b9eec54b509b87765d19482bd56975ab2454601d9af16bfc4",
    "checkpoint_engine/base.py": "504a3dd30f6187fde200fd3bc444d8849975c0b8675cecc7797ffa0373a2b7a5",
    "trainer/ppo/ray_trainer.py": "eb41f99ccc66976eab14ebe35aa1aa5b7318e02469240ef1569ac4c884c94312",
    "workers/rollout/replica.py": "d0896c19a4b22d052c93dcb9f48c86b5e010c3edcc76b2a2a58899f28525cdf9",
    "workers/rollout/vllm_rollout/vllm_async_server.py": "002754234f87a12c868e4195d242ea9b013c82597117e0984640a1b5a6468c1d",
    "single_controller/ray/base.py": "95aa631359ba10852d519a420ee82d416adcb13f07a38388a0986c5cb8f96a66",
}
_INIT_OBSERVER = ContextVar("mopd_phase_init_observer", default=None)
_INIT_LOCK = threading.RLock()


class PhaseContractError(RuntimeError):
    """Refuse execution when the reviewed phase or data contract is violated."""


def _need(condition, message):
    if not condition:
        raise PhaseContractError(message)


def _get(value, key, default=None):
    if hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)


def is_phased_config(config):
    return _get(_get(config, "mopd"), "execution_profile") == PROFILE


def verify_upstream_contract(root=None):
    """Require the exact reviewed upstream files before installing any hook."""
    if root is None:
        root = Path(importlib.import_module("verl").__file__).resolve().parent
    root = Path(root)
    for relative, expected in UPSTREAM_SHA256.items():
        path = root / relative
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise PhaseContractError("Missing pinned upstream source: {}".format(path)) from error
        _need(actual == expected, "Pinned upstream source changed: {}".format(path))
    return dict(UPSTREAM_SHA256)


def sequence_digest(tokens):
    values = [int(token) for token in tokens]
    _need(all(0 <= value < 2 ** 31 for value in values), "Invalid text token ID")
    return hashlib.sha256(struct.pack("<{}i".format(len(values)), *values)).hexdigest()


def reconstruct_sequence(prompts, responses, attention_mask, prompt_length, response_length):
    """Remove padding only; never remove a real token using the loss mask."""
    p, r = len(prompts), len(responses)
    _need(len(attention_mask) == p + r, "Attention mask width changed")
    pl, rl = int(prompt_length), int(response_length)
    _need(0 < pl <= p and 0 < rl <= r and pl + rl <= 65536,
          "Original prompt/response lengths violate the complete 64K budget")
    expected = [0] * (p - pl) + [1] * pl + [1] * rl + [0] * (r - rl)
    _need(list(attention_mask) == expected, "Unexpected padding or noncontiguous attention mask")
    return list(prompts[p - pl:]) + list(responses[:rl])


def require_same_order(expected, actual):
    _need(list(expected) == list(actual), "Teacher batch rows were dropped, duplicated, or reordered")


async def await_all(calls):
    """Wait for every in-flight operation before reporting any individual error."""
    results = await asyncio.gather(*calls, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


class StageRecorder:
    """Timing and optional physical-GPU observations, never mislabeled as peaks.

    observer(event) may attach a hardware snapshot or an independently sampled
    peak record. No torch allocation metric is substituted for physical GPU use.
    An observer failure is fatal, preserving a fail-closed qualification trace.
    """

    def __init__(self, path=None, observer=None, emit=None):
        self.path = Path(path) if path is not None else None
        self.observer = observer
        self.emit = emit

    def event(self, stage, event, **details):
        record = dict(stage=stage, event=event, utc=datetime.now(timezone.utc).isoformat(),
                      monotonic_ns=time.monotonic_ns(), **details)
        if self.observer is not None:
            record["physical_gpu_observation"] = self.observer(dict(record))
        if self.emit is not None:
            self.emit(dict(record))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
        return record

    @contextmanager
    def stage(self, name, **details):
        started = time.monotonic()
        self.event(name, "START", **details)
        try:
            yield
        except BaseException as error:
            self.event(name, "ERROR", elapsed_seconds=time.monotonic() - started,
                       error_type=type(error).__name__, error=str(error), **details)
            raise
        else:
            self.event(name, "FINISH", elapsed_seconds=time.monotonic() - started, **details)


def _recorder(config, observer=None):
    run_dir = _get(_get(config, "mopd"), "run_dir")
    attempt = os.environ.get("D4_MOPD_ATTEMPT")
    _need(run_dir and attempt and Path(attempt).name == attempt, "Missing explicit attempt for phase trace")
    return StageRecorder(Path(run_dir) / "attempts" / attempt / "phase_events.jsonl", observer=observer)


class PhaseRuntime:
    """Pure-Python controller; native components supply awaited operations."""

    def __init__(self, checkpoint, teachers, recorder=None):
        self.checkpoint = checkpoint
        self.teachers = teachers
        self.recorder = recorder or StageRecorder()
        self.state = "IDLE"
        _need(teachers.phase_state == "ASLEEP", "Teachers were not parked during initialization")

    def assert_teachers_asleep(self):
        _need(self.state != "FAILED", "Phase runtime failed; start a new attempt from a complete checkpoint")
        _need(self.teachers.phase_state == "ASLEEP", "Teacher allocation is not confirmed asleep")

    def assert_training_ready(self):
        self.assert_teachers_asleep()
        _need(self.checkpoint.student_asleep, "Student rollout must be asleep before native actor work")
        _need(self.state == "IDLE", "Actor work attempted during an active phase")

    async def generate(self, generation, scoring, *, validate, expected_count):
        self.assert_teachers_asleep()
        _need(self.state == "IDLE", "Concurrent phase entry is forbidden")
        _need(not self.checkpoint.student_asleep, "Student weights/KV have not been restored")
        try:
            self.state = "GENERATING"
            with self.recorder.stage("validation_generation" if validate else "student_generation"):
                batch = await generation()
                _need(len(batch) == expected_count, "Generation changed the full batch size")
            if validate:
                self.state = "IDLE"
                return batch
            self.state = "STUDENT_SLEEPING"
            with self.recorder.stage("student_sleep"):
                await self.checkpoint.sleep_replicas()
            _need(self.checkpoint.student_asleep, "Student sleep was not confirmed")
            self.state = "TEACHERS_WAKING"
            with self.recorder.stage("teachers_wake"):
                await self.teachers.wake_teachers()
            _need(self.teachers.phase_state == "AWAKE", "Teacher wake was not confirmed")
            self.state = "SCORING"
            with self.recorder.stage("teacher_scoring", rpc_limit_per_worker=TEACHER_RPC_LIMIT_PER_WORKER):
                batch = await scoring(batch)
                _need(len(batch) == expected_count, "Teacher scoring changed the full batch size")
            self.state = "TEACHERS_SLEEPING"
            with self.recorder.stage("teachers_sleep"):
                await self.teachers.sleep_teachers()
            _need(self.teachers.phase_state == "ASLEEP", "Teacher sleep was not confirmed")
            self.state = "IDLE"
            self.assert_training_ready()
            return batch
        except BaseException:
            self.state = "FAILED"
            # No partial batch, automatic wake, retry, or optimizer update.
            raise


def shared_resource_pool_manager(config, mapping):
    _need(is_phased_config(config), "Shared pool requires explicit single_node profile")
    _need(_get(config.trainer, "nnodes") == 1 and _get(config.trainer, "n_gpus_per_node") == 8,
          "Single profile requires exactly eight student GPUs")
    from verl.single_controller.ray.base import ResourcePoolManager
    from verl.trainer.ppo.utils import Role
    allowed = {Role.ActorRollout, Role.ActorRolloutRef, Role.TeacherModel}
    _need(set(mapping).issubset(allowed) and Role.TeacherModel in mapping,
          "Unexpected model roles in the shared eight-GPU pool")
    return ResourcePoolManager(resource_pool_spec={"global_pool": [8]},
                               mapping={role: "global_pool" for role in mapping}, max_colocate_count=3)


@contextmanager
def teacher_initialization_hook(config, observer=None):
    """Scoped export substitution required by native init_workers' local import."""
    if not is_phased_config(config):
        yield
        return
    verify_upstream_contract()
    adapters = _load_adapters()
    package = importlib.import_module("verl.experimental.teacher_loop")
    native = importlib.import_module("verl.experimental.teacher_loop.teacher_model").MultiTeacherModelManager
    replacement = adapters["ParkedMultiTeacherModelManager"]
    with _INIT_LOCK:
        _need(package.MultiTeacherModelManager is native, "Teacher initialization export already patched")
        token = _INIT_OBSERVER.set(observer)
        package.MultiTeacherModelManager = replacement
        try:
            yield
        finally:
            changed = package.MultiTeacherModelManager is not replacement
            package.MultiTeacherModelManager = native
            _INIT_OBSERVER.reset(token)
            _need(not changed, "Teacher initialization export changed inside scoped hook")


def bind_phase_runtime(trainer, observer=None):
    _need(is_phased_config(trainer.config), "Phase binding requires single_node profile")
    adapters = _load_adapters()
    _need(isinstance(trainer.teacher_model_manager, adapters["ParkedMultiTeacherModelManager"]),
          "Native teachers initialized without the parking hook")
    _need(isinstance(trainer.checkpoint_manager, adapters["PhasedCheckpointEngineManager"]),
          "Missing phase-aware checkpoint manager")
    _need(isinstance(trainer.async_rollout_manager, adapters["PhasedAgentLoopManager"]),
          "Missing phase-aware agent loop manager")
    recorder = trainer.teacher_model_manager.phase_recorder
    if observer is not None:
        recorder.observer = observer
    runtime = PhaseRuntime(trainer.checkpoint_manager, trainer.teacher_model_manager, recorder)
    _need(trainer.checkpoint_manager.student_asleep, "Initialization did not sleep student replicas")
    trainer.checkpoint_manager.phase_runtime = runtime
    trainer.async_rollout_manager.phase_runtime = runtime
    trainer.phase_runtime = runtime
    return runtime


def assert_training_ready(trainer):
    if is_phased_config(trainer.config):
        _need(hasattr(trainer, "phase_runtime"), "Single-node runtime is not bound")
        trainer.phase_runtime.assert_training_ready()


_ADAPTER_NAMES = {"PhasedAgentLoopManager", "PhasedAgentLoopWorker",
                  "PhasedCheckpointEngineManager", "ParkedMultiTeacherModelManager"}


def _load_adapters():
    if all(name in globals() for name in _ADAPTER_NAMES):
        return {name: globals()[name] for name in _ADAPTER_NAMES}
    verify_upstream_contract()
    import ray
    import torch
    from tensordict import TensorDict
    from verl.protocol import DataProto
    from verl.utils.ray_utils import auto_await
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
    from verl.experimental.teacher_loop import teacher_model as tm
    from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs
    from verl.checkpoint_engine import CheckpointEngineManager

    expected = {"output", "prompt_ids", "response_ids", "validate", "sample_kwargs"}
    _need(expected.issubset(inspect.signature(AgentLoopWorker._compute_teacher_logprobs).parameters),
          "Native per-sample teacher hook signature changed")

    class ParkedMultiTeacherModelManager(tm.MultiTeacherModelManager):
        def __init__(self, config, resource_pool):
            _need(is_phased_config(config), "Parked teachers require single_node profile")
            self.phase_recorder = _recorder(config, _INIT_OBSERVER.get())
            self.phase_state = "INITIALIZING"
            super().__init__(config, resource_pool)

        def _initialize_teacher_model_managers(self):
            models = self.distillation_config.teacher_models
            _need(set(models) == set(DOMAINS) and self.resource_pool.world_size == 8,
                  "Expected four teachers on the shared eight-GPU pool")
            for teacher in models.values():
                _need(teacher.num_replicas == 1 and teacher.per_replica_world_size == 2,
                      "Each persistent teacher must occupy one fixed TP2 pair")
                _need(teacher.inference.free_cache_engine and teacher.inference.enable_sleep_mode,
                      "Teacher sleep mode must remain enabled")
            pools = tm.split_resource_pool(self.resource_pool, split_size=[models[key].world_size for key in models])
            try:
                for (key, config), pool in zip(models.items(), pools):
                    with self.phase_recorder.stage("teacher_init", teacher=key):
                        manager = tm.TeacherModelManager(self.distillation_config, config, pool)
                    self.teacher_model_managers[key] = manager
                    self.server_addresses[key] = manager.server_addresses
                    self.server_handles[key] = manager.server_handles
                    self.load_balancer_handle[key] = manager.load_balancer_handle
                    with self.phase_recorder.stage("teacher_initial_sleep", teacher=key):
                        result = tm._run_all([replica.sleep() for replica in manager.rollout_replicas])
                        _need(not inspect.isawaitable(result), "Teacher initialization must run synchronously")
                self.phase_state = "ASLEEP"
            except BaseException:
                self.phase_state = "FAILED"
                raise

        async def wake_teachers(self):
            _need(self.phase_state == "ASLEEP", "Teacher wake requires a completely parked set")
            self.phase_state = "WAKING"
            try:
                await await_all([replica.wake_up() for manager in self.teacher_model_managers.values()
                                 for replica in manager.rollout_replicas])
                self.phase_state = "AWAKE"
            except BaseException:
                self.phase_state = "FAILED"
                raise

        async def sleep_teachers(self):
            _need(self.phase_state == "AWAKE", "Teacher sleep requires a completely awake set")
            self.phase_state = "SLEEPING"
            try:
                await await_all([replica.sleep() for manager in self.teacher_model_managers.values()
                                 for replica in manager.rollout_replicas])
                self.phase_state = "ASLEEP"
            except BaseException:
                self.phase_state = "FAILED"
                raise

    class PhasedCheckpointEngineManager(CheckpointEngineManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _need(self.backend == "naive" and len(self.replicas) == 2,
                  "Single profile requires native naive sync and two student TP4 replicas")
            self.student_asleep = False
            self.phase_runtime = None

        @auto_await
        async def sleep_replicas(self):
            if self.phase_runtime is not None:
                self.phase_runtime.assert_teachers_asleep()
            if self.student_asleep:
                return
            try:
                await super().sleep_replicas()
                self.student_asleep = True
            except BaseException:
                if self.phase_runtime is not None:
                    self.phase_runtime.state = "FAILED"
                raise

        @auto_await
        async def update_weights(self, global_steps=None):
            if self.phase_runtime is not None:
                self.phase_runtime.assert_training_ready()
            _need(self.student_asleep, "Native naive synchronization requires sleeping student replicas")
            recorder = self.phase_runtime.recorder if self.phase_runtime else StageRecorder()
            try:
                with recorder.stage("student_weight_sync", step=global_steps):
                    result = await super().update_weights(global_steps=global_steps)
                self.student_asleep = False
                return result
            except BaseException:
                if self.phase_runtime is not None:
                    self.phase_runtime.state = "FAILED"
                raise

    class PhasedAgentLoopWorker(AgentLoopWorker):
        async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
            if validate:
                return await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
            _need(is_phased_config(self.config) and self.distillation_enabled, "Distillation must stay enabled")
            _need(not output.multi_modal_data, "This qualification covers text-only MOPD")
            kwargs = sample_kwargs or {}
            domain, uid = kwargs.get(self.teacher_key), kwargs.get("uid")
            domain = domain.item() if hasattr(domain, "item") else domain
            uid = uid.item() if hasattr(uid, "item") else uid
            _need(domain in DOMAINS and isinstance(uid, str) and uid, "Missing original domain or UID")
            _need(not any(key in output.extra_fields for key in FIELDS), "Reserved phase metadata collision")
            values = (uid, domain, sequence_digest(prompt_ids + response_ids), len(prompt_ids), len(response_ids))
            output.extra_fields.update(zip(FIELDS, values))

        async def score_teacher_batch(self, batch):
            _need(is_phased_config(self.config) and self.distillation_enabled, "Invalid deferred scoring worker")
            _need(all(key in batch.non_tensor_batch for key in FIELDS), "Missing original generation lineage")
            _need("teacher_ids" not in batch.batch and "teacher_logprobs" not in batch.batch,
                  "Teacher scoring was already performed")
            p, r = batch.batch["prompts"].shape[1], batch.batch["responses"].shape[1]

            async def score(index):
                row = batch.batch[index]
                meta = {key: batch.non_tensor_batch[key][index] for key in FIELDS}
                sequence = reconstruct_sequence(row["prompts"].tolist(), row["responses"].tolist(),
                                                row["attention_mask"].tolist(), meta[FIELDS[3]], meta[FIELDS[4]])
                _need(sequence_digest(sequence) == meta[FIELDS[2]], "Student prefix changed before teacher scoring")
                _need(meta[FIELDS[1]] in DOMAINS, "Unknown teacher routing key")
                ids, logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                    sequence_ids=sequence, routing_key=meta[FIELDS[1]])
                ids, logprobs = _pad_teacher_outputs(ids, logprobs, p, r, int(meta[FIELDS[3]]),
                                                     int(meta[FIELDS[4]]), self.tokenizer.pad_token_id)
                mask = row["response_mask"].bool()
                _need(ids.shape == logprobs.shape and ids.shape == (1, p + r, 1), "Native K1 teacher shape changed")
                _need(bool(torch.isfinite(logprobs[0, p - 1:p + r - 1, 0][mask]).all()), "Nonfinite teacher signal")
                _need(bool((ids[0, p - 1:p + r - 1, 0][mask] == row["responses"][mask]).all()),
                      "Native teacher next-token alignment changed")
                return ids, logprobs

            # Bound host-side unpadded prefix/RPC materialization. This changes
            # neither the full optimizer batch nor vLLM's token/sequence budget.
            semaphore = asyncio.Semaphore(TEACHER_RPC_LIMIT_PER_WORKER)

            async def bounded_score(index):
                async with semaphore:
                    return await score(index)

            results = await await_all([bounded_score(index) for index in range(len(batch))])
            return DataProto(batch=TensorDict({"teacher_ids": torch.cat([v[0] for v in results]),
                                               "teacher_logprobs": torch.cat([v[1] for v in results])},
                                              batch_size=len(batch)),
                             non_tensor_batch={key: batch.non_tensor_batch[key].copy() for key in FIELDS})

    class PhasedAgentLoopManager(AgentLoopManager):
        def __init__(self, config, *args, **kwargs):
            _need(is_phased_config(config), "Phased agent loop requires explicit single_node profile")
            _need(config.data.train_batch_size == 2048 and config.actor_rollout_ref.rollout.n == 1,
                  "Single-node qualification must retain B2048 and R1")
            _need(config.actor_rollout_ref.rollout.max_model_len == 65536,
                  "Single-node qualification must retain the real 64K budget")
            self.agent_loop_workers_class = ray.remote(PhasedAgentLoopWorker)
            self.phase_runtime = None
            super().__init__(config, *args, **kwargs)

        @auto_await
        async def generate_sequences(self, prompts):
            _need(self.phase_runtime is not None, "Phase runtime must be bound before generation")
            validate = bool(prompts.meta_info.get("validate", False))
            if not validate:
                _need(len(prompts) == 2048, "A training update must contain the complete B2048 batch")
            original_uids = list(prompts.non_tensor_batch.get("uid", []))
            original_domains = list(prompts.non_tensor_batch.get("data_source", []))

            async def generate():
                output = await super(PhasedAgentLoopManager, self).generate_sequences(prompts)
                if not validate:
                    require_same_order(original_uids, output.non_tensor_batch[FIELDS[0]])
                    _need(original_domains == list(output.non_tensor_batch[FIELDS[1]]),
                          "Teacher route changed from the original per-sample domain")
                return output

            async def score(batch):
                chunks = batch.chunk(len(self.agent_loop_workers))
                _need(len(chunks) == len(self.agent_loop_workers), "Teacher chunk/worker count changed")
                pieces = await await_all([worker.score_teacher_batch.remote(chunk)
                                          for worker, chunk in zip(self.agent_loop_workers, chunks)])
                scored = DataProto.concat(pieces)
                for key in FIELDS:
                    require_same_order(batch.non_tensor_batch[key], scored.non_tensor_batch[key])
                _need(set(scored.batch.keys()) == {"teacher_ids", "teacher_logprobs"}, "Unexpected scoring output")
                batch.batch["teacher_ids"] = scored.batch["teacher_ids"]
                batch.batch["teacher_logprobs"] = scored.batch["teacher_logprobs"]
                return batch

            return await self.phase_runtime.generate(generate, score, validate=validate, expected_count=len(prompts))

    adapters = {name: value for name, value in locals().items() if name in _ADAPTER_NAMES}
    for name, value in adapters.items():
        value.__module__ = __name__
        value.__qualname__ = name
        globals()[name] = value
    return adapters


def __getattr__(name):
    if name in _ADAPTER_NAMES:
        return _load_adapters()[name]
    raise AttributeError(name)
