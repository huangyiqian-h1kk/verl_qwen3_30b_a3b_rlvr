"""CPU replay of the pinned verl metadata path, using synthetic model outputs."""
import json
from types import SimpleNamespace

import numpy as np
import torch

from . import DOMAINS
from .dataset import MOPDDataset
from .integration import check_batch
from .spec import domain_order, load_spec
from d4_teachers.common import need
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn


def replay(domains, *, reward_workers=True):
    """Exercise dataset/collate/pop/agent-postprocess/union/reorder without GPUs."""
    dataset = object.__new__(MOPDDataset)
    dataset.qualification = True
    # Repeated stable IDs are valid when the short qualification dataset wraps.
    dataset.rows = [dict(domain=d, sample_id=f"{d}:{i % 2}", source=d,
        messages_json='[{"role":"user","content":"fixture"}]', tools_json="[]", verifier_json="{}",
        prompt_tokens=2, prompt_token_hash="fixture", generation_max_tokens=8, mopd_split="train")
        for i, d in enumerate(domains)]
    batch = DataProto.from_single_dict(collate_fn([dataset[i] for i in range(len(domains))]))
    batch.non_tensor_batch["uid"] = np.array([f"rollout-{i}" for i in range(len(domains))], dtype=object)
    gen = RayPPOTrainer._get_gen_batch(None, batch).repeat(repeat_times=1, interleave=True)
    need("index" in gen.non_tensor_batch and "index" not in batch.non_tensor_batch,
         "Pinned verl generation metadata contract changed")
    outputs = []
    for i in range(len(domains)):
        prompt, response = torch.tensor([[5, 6]]), torch.tensor([[7, 8, 0]])
        mask = torch.tensor([[1, 1, 0]])
        outputs.append(_InternalAgentLoopOutput(
            prompt_ids=prompt, response_ids=response, response_mask=mask,
            input_ids=torch.cat((prompt, response), dim=1),
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]]), position_ids=torch.arange(5).unsqueeze(0),
            teacher_ids=torch.tensor([[[6], [7], [8], [0], [0]]]),
            teacher_logprobs=-torch.ones(1, 5, 1), reward_score=0., num_turns=2,
            multi_modal_inputs={}, metrics=AgentLoopMetrics(),
            extra_fields={"reward_extra_info": {"task_reward_disabled": 1.}}))
    worker = SimpleNamespace(reward_loop_worker_handles=["fixture"] if reward_workers else None)
    output = AgentLoopWorker._postprocess(worker, outputs, input_non_tensor_batch=gen.non_tensor_batch)
    batch = batch.repeat(repeat_times=1, interleave=True).union(output)
    batch.batch["old_log_probs"] = -2 * torch.ones(len(domains), 3)
    batch.batch["token_level_rewards"] = batch.batch["rm_scores"].clone()
    batch.reorder(torch.arange(len(domains) - 1, -1, -1))
    need(("index" not in batch.non_tensor_batch) == reward_workers,
         "Pinned verl agent output metadata contract changed")
    return batch, [row["sample_id"] for row in dataset.rows]


def probe():
    spec = dict(load_spec(), train_batch_size=4)
    cases = []
    for reward_workers in (False, True):
        for arm in ("domain_mix", "domain_only"):
            for step in (1, 5):  # Fresh process and the first update after step-4 resume.
                domains = list(DOMAINS) if arm == "domain_mix" else [domain_order(spec)[step-1]] * 4
                batch, expected = replay(domains, reward_workers=reward_workers)
                result = check_batch(batch, spec, arm, step, expected)
                cases.append(dict(arm=arm, step=step, reward_workers=reward_workers,
                                  sample_ids_sha256=result["sample_ids_sha256"]))
    return dict(status="PASS", scope="cpu-native-batch-metadata; synthetic model outputs; no GPU execution",
                stable_id="extra_info.sample_id", rollout_n=1, cases=cases, gpu_execution_tested=False)


if __name__ == "__main__":
    print(json.dumps(probe()))
