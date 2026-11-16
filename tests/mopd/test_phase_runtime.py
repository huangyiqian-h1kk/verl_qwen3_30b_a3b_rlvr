"""CPU checks for whole-batch staging; no Ray, CUDA, network, or model weights."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from d4_mopd import phase_runtime as phase


class Checkpoint:
    def __init__(self, events):
        self.student_asleep = False
        self.events = events

    async def sleep_replicas(self):
        self.events.append("student_sleep")
        self.student_asleep = True


class Teachers:
    def __init__(self, events, failure=None):
        self.phase_state = "ASLEEP"
        self.events, self.failure = events, failure

    async def wake_teachers(self):
        self.events.append("teachers_wake")
        if self.failure == "wake":
            self.phase_state = "FAILED"
            raise RuntimeError("wake failed")
        self.phase_state = "AWAKE"

    async def sleep_teachers(self):
        self.events.append("teachers_sleep")
        if self.failure == "sleep":
            self.phase_state = "FAILED"
            raise RuntimeError("sleep failed")
        self.phase_state = "ASLEEP"


class PhaseOrderingTests(unittest.IsolatedAsyncioTestCase):
    def setup_runtime(self, failure=None):
        events, records = [], []
        checkpoint, teachers = Checkpoint(events), Teachers(events, failure)
        runtime = phase.PhaseRuntime(checkpoint, teachers, phase.StageRecorder(emit=records.append))
        return runtime, events, records

    async def test_full_batch_order_preserves_object_and_one_native_update(self):
        runtime, events, records = self.setup_runtime()
        batch = list(range(2048))

        async def generate():
            self.assertEqual(runtime.teachers.phase_state, "ASLEEP")
            events.append("generate_complete_2048")
            return batch

        async def score(value):
            self.assertIs(value, batch)
            self.assertTrue(runtime.checkpoint.student_asleep)
            self.assertEqual(runtime.teachers.phase_state, "AWAKE")
            events.append("score_complete_2048")
            return value

        output = await runtime.generate(generate, score, validate=False, expected_count=2048)
        runtime.assert_training_ready()
        events.append("native_update_once")
        self.assertIs(output, batch)
        self.assertEqual(events, ["generate_complete_2048", "student_sleep", "teachers_wake",
                                  "score_complete_2048", "teachers_sleep", "native_update_once"])
        self.assertEqual([r["stage"] for r in records if r["event"] == "FINISH"],
                         ["student_generation", "student_sleep", "teachers_wake", "teacher_scoring", "teachers_sleep"])
        self.assertTrue(all(r["elapsed_seconds"] >= 0 for r in records if r["event"] == "FINISH"))

    async def test_real_validation_does_not_wake_teachers_or_sleep_student(self):
        runtime, events, records = self.setup_runtime()

        async def generate():
            return ["validation"] * 8

        async def forbidden(_):
            self.fail("Teacher scoring is forbidden during native validation")

        output = await runtime.generate(generate, forbidden, validate=True, expected_count=8)
        self.assertEqual(len(output), 8)
        self.assertEqual(events, [])
        self.assertFalse(runtime.checkpoint.student_asleep)
        self.assertEqual(runtime.teachers.phase_state, "ASLEEP")
        self.assertEqual(records[0]["stage"], "validation_generation")

    async def test_generation_cannot_drop_sample(self):
        runtime, events, _ = self.setup_runtime()

        async def generate():
            return list(range(2047))

        async def forbidden(_):
            self.fail("Must not score an incomplete batch")

        with self.assertRaises(phase.PhaseContractError):
            await runtime.generate(generate, forbidden, validate=False, expected_count=2048)
        self.assertEqual(events, [])
        self.assertEqual(runtime.state, "FAILED")

    async def test_wake_failure_never_scores_or_allows_actor_work(self):
        runtime, events, _ = self.setup_runtime("wake")

        async def generate():
            return [0] * 2048

        async def forbidden(_):
            self.fail("Wake failure must not score")

        with self.assertRaisesRegex(RuntimeError, "wake failed"):
            await runtime.generate(generate, forbidden, validate=False, expected_count=2048)
        self.assertEqual(events, ["student_sleep", "teachers_wake"])
        with self.assertRaises(phase.PhaseContractError):
            runtime.assert_training_ready()

    async def test_scoring_failure_does_not_retry_or_wake_student(self):
        runtime, events, records = self.setup_runtime()
        calls = []

        async def generate():
            return [0] * 2048

        async def fail(_):
            calls.append(1)
            raise RuntimeError("one scoring RPC failed")

        with self.assertRaisesRegex(RuntimeError, "scoring RPC"):
            await runtime.generate(generate, fail, validate=False, expected_count=2048)
        self.assertEqual(calls, [1])
        self.assertTrue(runtime.checkpoint.student_asleep)
        self.assertEqual(runtime.state, "FAILED")
        self.assertEqual(records[-1]["event"], "ERROR")
        with self.assertRaises(phase.PhaseContractError):
            await runtime.generate(generate, fail, validate=False, expected_count=2048)

    async def test_sleep_failure_prevents_actor_entry(self):
        runtime, events, _ = self.setup_runtime("sleep")

        async def generate():
            return [0] * 2048

        async def score(batch):
            return batch

        with self.assertRaisesRegex(RuntimeError, "sleep failed"):
            await runtime.generate(generate, score, validate=False, expected_count=2048)
        with self.assertRaises(phase.PhaseContractError):
            runtime.assert_training_ready()
        self.assertEqual(events[-1], "teachers_sleep")

    async def test_all_inflight_calls_settle_before_error_is_returned(self):
        completed = []

        async def fail():
            raise RuntimeError("first")

        async def finish():
            await asyncio.sleep(0)
            completed.append("settled")
            return 7

        with self.assertRaisesRegex(RuntimeError, "first"):
            await phase.await_all([fail(), finish()])
        self.assertEqual(completed, ["settled"])

    async def test_no_concurrent_entry(self):
        runtime, _, _ = self.setup_runtime()
        runtime.state = "SCORING"
        with self.assertRaises(phase.PhaseContractError):
            await runtime.generate(None, None, validate=False, expected_count=2048)


class TokenContractTests(unittest.TestCase):
    def test_padding_removal_preserves_all_real_tokens_and_real_zero_token(self):
        prompt, response = [999, 0, 11], [12, 0, 13, 999]
        sequence = phase.reconstruct_sequence(prompt, response, [0, 1, 1, 1, 1, 1, 0], 2, 3)
        self.assertEqual(sequence, [0, 11, 12, 0, 13])

    def test_full_64k_is_not_shortened(self):
        prompt, response = [42], [43] * 65535
        sequence = phase.reconstruct_sequence(prompt, response, [1] * 65536, 1, 65535)
        self.assertEqual(len(sequence), 65536)
        self.assertEqual(phase.sequence_digest(sequence), phase.sequence_digest(prompt + response))

    def test_internal_mask_hole_and_overlength_fail(self):
        with self.assertRaises(phase.PhaseContractError):
            phase.reconstruct_sequence([7, 8], [9, 10], [1, 0, 1, 1], 1, 2)
        with self.assertRaises(phase.PhaseContractError):
            phase.reconstruct_sequence([1], [2] * 65536, [1] * 65537, 1, 65536)

    def test_order_check_rejects_reordered_dropped_and_duplicate_rows(self):
        expected = ["a", "b", "c"]
        phase.require_same_order(expected, expected.copy())
        for changed in (["b", "a", "c"], ["a", "b"], ["a", "b", "b"]):
            with self.assertRaises(phase.PhaseContractError):
                phase.require_same_order(expected, changed)

    def test_digest_changes_with_token_order_not_padding_width(self):
        self.assertNotEqual(phase.sequence_digest([1, 2]), phase.sequence_digest([2, 1]))
        with self.assertRaises(phase.PhaseContractError):
            phase.sequence_digest([-1])


class HookAndEvidenceTests(unittest.TestCase):
    def test_canonical_project_domains(self):
        from d4_mopd import DOMAINS
        self.assertIs(phase.DOMAINS, DOMAINS)
        self.assertIn("conversational_pivot", phase.DOMAINS)
        self.assertIn("logic_algorithmic", phase.DOMAINS)

    def test_profile_is_outside_scientific_spec(self):
        self.assertTrue(phase.is_phased_config({"mopd": {"execution_profile": "single_node"}}))
        self.assertFalse(phase.is_phased_config({"mopd": {"effective_spec": {"execution_profile": "single_node"}}}))
        self.assertFalse(phase.is_phased_config({"mopd": {"execution_profile": "dual_node"}}))

    def test_exact_source_contract_fails_on_drift_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "native.py"
            source = b"pinned source\n"
            path.write_bytes(source)
            with mock.patch.object(phase, "UPSTREAM_SHA256", {"native.py": hashlib.sha256(source).hexdigest()}):
                phase.verify_upstream_contract(root)
                path.write_bytes(source + b"changed")
                with self.assertRaisesRegex(phase.PhaseContractError, "changed"):
                    phase.verify_upstream_contract(root)
                path.unlink()
                with self.assertRaisesRegex(phase.PhaseContractError, "Missing"):
                    phase.verify_upstream_contract(root)

    def test_initialization_export_is_restored_even_on_failure(self):
        package, model = ModuleType("teacher_loop"), ModuleType("teacher_model")
        native, parked = type("Native", (), {}), type("Parked", (), {})
        package.MultiTeacherModelManager = model.MultiTeacherModelManager = native
        modules = {"verl.experimental.teacher_loop": package,
                   "verl.experimental.teacher_loop.teacher_model": model}
        with mock.patch.object(phase, "verify_upstream_contract"), \
             mock.patch.object(phase, "_load_adapters", return_value={"ParkedMultiTeacherModelManager": parked}), \
             mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                with phase.teacher_initialization_hook({"mopd": {"execution_profile": "single_node"}}):
                    self.assertIs(package.MultiTeacherModelManager, parked)
                    self.assertIs(model.MultiTeacherModelManager, native)
                    raise RuntimeError("initialization failed")
            self.assertIs(package.MultiTeacherModelManager, native)
            self.assertIsNone(phase._INIT_OBSERVER.get())

    def test_dual_profile_never_imports_or_patches_native_classes(self):
        with mock.patch.object(phase, "_load_adapters", side_effect=AssertionError("unwanted import")):
            with phase.teacher_initialization_hook({"mopd": {"execution_profile": "dual_node"}}):
                pass

    def test_trace_records_timing_and_hardware_observer_without_inventing_peak(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "phase.jsonl"
            observer = mock.Mock(return_value={"kind": "boundary_snapshot", "physical_gpu_count": 8})
            recorder = phase.StageRecorder(path, observer=observer)
            with recorder.stage("native_actor_update", step=1):
                pass
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["event"] for r in rows], ["START", "FINISH"])
            self.assertEqual(rows[1]["physical_gpu_observation"]["kind"], "boundary_snapshot")
            self.assertGreaterEqual(rows[1]["elapsed_seconds"], 0)
            self.assertEqual(observer.call_count, 2)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("verl"),
                     "requires the pinned native CPU dependency environment")
class NativeAdapterTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = phase._load_adapters()

    def make_worker(self):
        from d4_mopd.compat import BasicPGConfig
        from verl.workers.config import DistillationConfig, DistillationTeacherModelConfig, RolloutConfig
        from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
        teachers = {"teacher_model": DistillationTeacherModelConfig()}
        for domain in phase.DOMAINS:
            teachers[domain] = DistillationTeacherModelConfig(
                key=domain, model_path="/unused/" + domain, num_replicas=1,
                inference=RolloutConfig(name="vllm", tensor_model_parallel_size=2,
                                       prompt_length=20665, response_length=65535, max_model_len=86201,
                                       max_num_seqs=16, max_num_batched_tokens=8192))
        distillation = DistillationConfig(
            enabled=True, n_gpus_per_node=8, nnodes=1, teacher_models=teachers,
            distillation_loss=BasicPGConfig(loss_mode="k1", use_policy_gradient=True,
                                           policy_loss_mode="gpg", use_task_rewards=False,
                                           loss_max_clamp=5.0, log_prob_min_clamp=None))
        self.assertEqual(set(distillation.teacher_models), set(phase.DOMAINS))
        calls = []

        class Client:
            def __init__(self, domain):
                self.domain = domain

            async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
                if len(calls) % 7 == 0:
                    await asyncio.sleep(0)
                self_params = dict(sampling_params)
                calls.append((self.domain, list(prompt_ids), self_params))
                next_tokens = prompt_ids[1:] + [99]
                offset = phase.DOMAINS.index(self.domain) + 1
                return SimpleNamespace(extra_fields={"prompt_ids": [[value] for value in next_tokens],
                                                     "prompt_logprobs": [[-offset / 10] for _ in next_tokens]})

        config = SimpleNamespace(mopd=SimpleNamespace(execution_profile="single_node"), distillation=distillation)
        worker = self.adapters["PhasedAgentLoopWorker"].__new__(self.adapters["PhasedAgentLoopWorker"])
        worker.config, worker.distillation_enabled = config, True
        worker.teacher_key = "data_source"
        worker.tokenizer = SimpleNamespace(pad_token_id=999)
        worker.teacher_server_manager = AsyncTeacherLLMServerManager(config, {d: Client(d) for d in phase.DOMAINS})
        return worker, calls

    async def test_native_resource_mapping_reserves_eight_not_sixteen_gpus(self):
        from verl.trainer.ppo.utils import Role
        config = SimpleNamespace(mopd=SimpleNamespace(execution_profile="single_node"),
                                 trainer=SimpleNamespace(nnodes=1, n_gpus_per_node=8))
        original = {Role.ActorRollout: "global_pool", Role.TeacherModel: "teacher_pool"}
        manager = phase.shared_resource_pool_manager(config, original)
        self.assertEqual(manager.resource_pool_spec, {"global_pool": [8]})
        self.assertEqual(manager.get_n_gpus(), 8)
        self.assertEqual(set(manager.mapping.values()), {"global_pool"})
        self.assertEqual(original[Role.TeacherModel], "teacher_pool")
        self.assertEqual(manager.max_colocate_count, 3)

    def make_batch(self, count, long=False):
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl.protocol import DataProto
        rows, metadata = [], {key: [] for key in phase.FIELDS}
        for index in range(count):
            if long:
                prompt, response, pl, rl = [41], [42] * 65535, 1, 65535
            else:
                pl, rl = 1 + index % 2, 2 + index % 2
                prompt = [999] * (3 - pl) + [10 + index % 5] * pl
                response = [20 + index % 5] * rl + [999] * (4 - rl)
            attention = [0] * (len(prompt) - pl) + [1] * (pl + rl) + [0] * (len(response) - rl)
            sequence = prompt[-pl:] + response[:rl]
            rows.append((prompt, response, attention, [1] * rl + [0] * (len(response) - rl)))
            values = ("uid-{}".format(index), phase.DOMAINS[index % 4], phase.sequence_digest(sequence), pl, rl)
            for key, value in zip(phase.FIELDS, values):
                metadata[key].append(value)
        tensors = {key: torch.tensor([row[col] for row in rows], dtype=torch.long)
                   for col, key in enumerate(("prompts", "responses", "attention_mask", "response_mask"))}
        return DataProto(batch=TensorDict(tensors, batch_size=count),
                         non_tensor_batch={key: np.array(values, dtype=object) for key, values in metadata.items()})

    async def test_native_teacher_manager_padding_full_b2048_and_four_real_keys(self):
        from collections import Counter
        import torch
        from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs
        worker, calls = self.make_worker()
        batch = self.make_batch(2048)
        result = await worker.score_teacher_batch(batch)
        self.assertEqual(len(result), 2048)
        self.assertEqual(Counter(domain for domain, _, _ in calls), {d: 512 for d in phase.DOMAINS})
        self.assertTrue(all(params["max_tokens"] == 1 and params["prompt_logprobs"] == 0 for _, _, params in calls))
        phase.require_same_order(batch.non_tensor_batch[phase.FIELDS[0]], result.non_tensor_batch[phase.FIELDS[0]])
        mask = batch.batch["response_mask"].bool()
        self.assertTrue(torch.equal(result.batch["teacher_ids"][:, 2:6, 0][mask], batch.batch["responses"][mask]))
        for index in range(2048):
            expected = -(index % 4 + 1) / 10
            self.assertTrue(torch.allclose(result.batch["teacher_logprobs"][index, 2:6, 0][mask[index]],
                                           torch.full((int(mask[index].sum()),), expected)))
            pl, rl = int(batch.non_tensor_batch[phase.FIELDS[3]][index]), int(batch.non_tensor_batch[phase.FIELDS[4]][index])
            sequence = batch.batch["prompts"][index, -pl:].tolist() + batch.batch["responses"][index, :rl].tolist()
            expected_ids, expected_lp = _pad_teacher_outputs(
                torch.tensor(sequence[1:] + [99], dtype=torch.int32).unsqueeze(-1),
                torch.full((len(sequence), 1), expected), 3, 4, pl, rl, 999)
            self.assertTrue(torch.equal(result.batch["teacher_ids"][index:index + 1], expected_ids))
            self.assertTrue(torch.equal(result.batch["teacher_logprobs"][index:index + 1], expected_lp))

    async def test_native_teacher_scoring_preserves_full_64k_prefix(self):
        worker, calls = self.make_worker()
        batch = self.make_batch(1, long=True)
        result = await worker.score_teacher_batch(batch)
        self.assertEqual(len(calls[0][1]), 65536)
        self.assertEqual(tuple(result.batch["teacher_ids"].shape), (1, 65536, 1))

    async def test_native_score_rejects_modified_prefix_before_any_teacher_call(self):
        worker, calls = self.make_worker()
        batch = self.make_batch(1)
        batch.batch["responses"][0, 0] += 1
        with self.assertRaisesRegex(phase.PhaseContractError, "prefix changed"):
            await worker.score_teacher_batch(batch)
        self.assertEqual(calls, [])

    async def test_eos_and_loss_mask_hole_remain_in_teacher_context(self):
        worker, calls = self.make_worker()
        batch = self.make_batch(1)
        batch.batch["responses"][0] = __import__("torch").tensor([12, 151645, 13, 999])
        batch.batch["response_mask"][0] = __import__("torch").tensor([1, 0, 1, 0])
        batch.batch["attention_mask"][0] = __import__("torch").tensor([0, 0, 1, 1, 1, 1, 0])
        batch.non_tensor_batch[phase.FIELDS[4]][0] = 3
        expected = [10, 12, 151645, 13]
        batch.non_tensor_batch[phase.FIELDS[2]][0] = phase.sequence_digest(expected)
        await worker.score_teacher_batch(batch)
        self.assertEqual(calls[0][1], expected)

    async def test_deferred_hook_records_native_lineage_without_calling_teacher(self):
        worker, calls = self.make_worker()
        output = SimpleNamespace(extra_fields={}, multi_modal_data={})
        await worker._compute_teacher_logprobs(output, [8, 9], [10, 151645], False,
                                               {"data_source": "conversational_pivot", "uid": "row-original"})
        self.assertEqual(calls, [])
        self.assertEqual(output.extra_fields[phase.FIELDS[0]], "row-original")
        self.assertEqual(output.extra_fields[phase.FIELDS[1]], "conversational_pivot")
        self.assertEqual(output.extra_fields[phase.FIELDS[2]], phase.sequence_digest([8, 9, 10, 151645]))
        self.assertNotIn("teacher_logprobs", output.extra_fields)

    async def test_native_teacher_initialization_parks_each_pair_before_next(self):
        from verl.experimental.teacher_loop import teacher_model as tm
        worker, _ = self.make_worker()
        events = []

        class Replica:
            def __init__(self, key):
                self.key = key

            async def sleep(self):
                events.append(("sleep", self.key))

            async def wake_up(self):
                events.append(("wake", self.key))

        class Manager:
            def __init__(self, distillation, config, pool):
                events.append(("init", config.key))
                self.rollout_replicas = [Replica(config.key)]
                self.server_addresses, self.server_handles, self.load_balancer_handle = [], [], object()

        with mock.patch.object(phase, "_recorder", return_value=phase.StageRecorder()), \
             mock.patch.object(tm, "split_resource_pool", return_value=[object() for _ in range(4)]), \
             mock.patch.object(tm, "TeacherModelManager", Manager):
            manager = self.adapters["ParkedMultiTeacherModelManager"](worker.config, SimpleNamespace(world_size=8))
        self.assertEqual(events, [(action, key) for key in phase.DOMAINS for action in ("init", "sleep")])
        self.assertEqual(manager.phase_state, "ASLEEP")
        await manager.wake_teachers()
        self.assertEqual(manager.phase_state, "AWAKE")
        await manager.sleep_teachers()
        self.assertEqual(manager.phase_state, "ASLEEP")

    async def test_native_checkpoint_sleep_is_idempotent_and_sync_restores_awake(self):
        from verl.workers.config import CheckpointEngineConfig
        events = []

        class Replica:
            async def sleep(self):
                events.append("sleep")

        class Actor:
            def update_weights(self, **kwargs):
                async def synchronize():
                    events.append("native_sync")
                return [synchronize()]

        manager = self.adapters["PhasedCheckpointEngineManager"](
            CheckpointEngineConfig(backend="naive"), Actor(), [Replica(), Replica()])
        try:
            await manager.sleep_replicas()
            await manager.sleep_replicas()
            self.assertEqual(events, ["sleep", "sleep"])
            self.assertTrue(manager.student_asleep)
            await manager.update_weights(global_steps=1)
            self.assertEqual(events[-1], "native_sync")
            self.assertFalse(manager.student_asleep)
            with self.assertRaises(phase.PhaseContractError):
                await manager.update_weights(global_steps=1)
        finally:
            manager.close()

    async def test_native_fqn_create_dispatches_custom_worker_class(self):
        from verl.utils.import_utils import load_class_from_fqn
        import ray
        worker_classes, constructed = [], []

        class RemoteClass:
            def options(self, **kwargs):
                return self

            def remote(self, *args):
                constructed.append(args)
                return SimpleNamespace()

        def remote(cls):
            worker_classes.append(cls)
            return RemoteClass()

        rollout = SimpleNamespace(n=1, max_model_len=65536, agent=SimpleNamespace(num_workers=8))
        config = SimpleNamespace(mopd=SimpleNamespace(execution_profile="single_node"),
                                 data=SimpleNamespace(train_batch_size=2048),
                                 actor_rollout_ref=SimpleNamespace(rollout=rollout, model=SimpleNamespace()))
        loaded = load_class_from_fqn("d4_mopd.phase_runtime.PhasedAgentLoopManager", "AgentLoopManager")
        with mock.patch.object(ray, "remote", side_effect=remote), \
             mock.patch.object(ray, "nodes", return_value=[{"NodeID": "ab" * 28, "Alive": True, "Resources": {"CPU": 32}}]):
            manager = await loaded.create(config, llm_client=object(), teacher_client={})
        self.assertIs(loaded, self.adapters["PhasedAgentLoopManager"])
        self.assertEqual(worker_classes, [self.adapters["PhasedAgentLoopWorker"]])
        self.assertEqual(len(constructed), 8)
        self.assertEqual(len(manager.agent_loop_workers), 8)

    async def test_native_manager_dataproto_chunk_concat_preserves_all_2048_rows(self):
        from verl.experimental.agent_loop.agent_loop import AgentLoopManager
        worker, calls = self.make_worker()
        batch = self.make_batch(2048)
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch[phase.FIELDS[0]].copy()
        batch.non_tensor_batch["data_source"] = batch.non_tensor_batch[phase.FIELDS[1]].copy()
        manager = self.adapters["PhasedAgentLoopManager"].__new__(self.adapters["PhasedAgentLoopManager"])
        events = []
        manager.phase_runtime = phase.PhaseRuntime(Checkpoint(events), Teachers(events))
        manager.agent_loop_workers = [SimpleNamespace(score_teacher_batch=SimpleNamespace(
            remote=lambda chunk: worker.score_teacher_batch(chunk))) for _ in range(8)]

        async def native_generation(self, prompts):
            return batch

        with mock.patch.object(AgentLoopManager, "generate_sequences", native_generation):
            output = await manager.generate_sequences(batch)
        self.assertIs(output, batch)
        self.assertEqual(len(calls), 2048)
        self.assertEqual(tuple(output.batch["teacher_ids"].shape), (2048, 7, 1))
        phase.require_same_order(output.non_tensor_batch["uid"], output.non_tensor_batch[phase.FIELDS[0]])
        manager.phase_runtime.assert_training_ready()

    async def test_native_manager_rejects_changed_chunk_lineage(self):
        from verl.experimental.agent_loop.agent_loop import AgentLoopManager
        worker, _ = self.make_worker()
        batch = self.make_batch(2048)
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch[phase.FIELDS[0]].copy()
        batch.non_tensor_batch["data_source"] = batch.non_tensor_batch[phase.FIELDS[1]].copy()
        manager = self.adapters["PhasedAgentLoopManager"].__new__(self.adapters["PhasedAgentLoopManager"])
        events = []
        manager.phase_runtime = phase.PhaseRuntime(Checkpoint(events), Teachers(events))

        async def corrupt(chunk):
            result = await worker.score_teacher_batch(chunk)
            result.non_tensor_batch[phase.FIELDS[0]] = result.non_tensor_batch[phase.FIELDS[0]][::-1]
            return result

        manager.agent_loop_workers = [SimpleNamespace(score_teacher_batch=SimpleNamespace(remote=corrupt))
                                      for _ in range(8)]

        async def native_generation(self, prompts):
            return batch

        with mock.patch.object(AgentLoopManager, "generate_sequences", native_generation):
            with self.assertRaisesRegex(phase.PhaseContractError, "reordered"):
                await manager.generate_sequences(batch)
        self.assertEqual(manager.phase_runtime.state, "FAILED")
        with self.assertRaises(phase.PhaseContractError):
            manager.phase_runtime.assert_training_ready()

    async def test_native_manager_rejects_different_canonical_domain_before_teacher_wake(self):
        from verl.experimental.agent_loop.agent_loop import AgentLoopManager
        batch = self.make_batch(2048)
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch[phase.FIELDS[0]].copy()
        batch.non_tensor_batch["data_source"] = batch.non_tensor_batch[phase.FIELDS[1]].copy()
        batch.non_tensor_batch[phase.FIELDS[1]][0] = phase.DOMAINS[1]
        manager = self.adapters["PhasedAgentLoopManager"].__new__(self.adapters["PhasedAgentLoopManager"])
        events = []
        manager.phase_runtime = phase.PhaseRuntime(Checkpoint(events), Teachers(events))

        async def native_generation(self, prompts):
            return batch

        with mock.patch.object(AgentLoopManager, "generate_sequences", native_generation):
            with self.assertRaisesRegex(phase.PhaseContractError, "original per-sample domain"):
                await manager.generate_sequences(batch)
        self.assertEqual(manager.phase_runtime.state, "FAILED")
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
