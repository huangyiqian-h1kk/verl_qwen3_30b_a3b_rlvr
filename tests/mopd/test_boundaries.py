"""Integration boundaries most likely to silently change the experiment."""
import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tensordict import TensorDict

from d4_mopd import DOMAINS
from d4_mopd.config import compose
from d4_mopd.integration import check_batch
from d4_mopd.metrics import source_metrics
from d4_teachers.common import write
from verl import DataProto
from verl.trainer.ppo.utils import create_rl_dataset


def test_native_dataset_factory_preserves_prompt_and_domain(manifest, tmp_path):
    rows = [dict(domain=d, sample_id=d, source="frozen_source", messages_json='[{"role":"user","content":"x"}]',
                 tools_json="[]", verifier_json="{}", prompt_tokens=12, prompt_token_hash="hash",
                 generation_max_tokens=65524, mopd_split="train") for d in DOMAINS]
    pq.write_table(pa.Table.from_pylist(rows), manifest["data"]["train"][0])
    config = compose(manifest, "domain_mix", tmp_path, qualification=True)
    dataset = create_rl_dataset(config.data.train_files, config.data, None, None)
    assert [dataset[i]["data_source"] for i in range(4)] == list(DOMAINS)
    assert dataset[0]["raw_prompt"] == [{"role":"user", "content":"x"}]
    assert dataset[0]["extra_info"]["generation_max_tokens"] == 256
    assert dataset[0]["extra_info"]["prompt_token_hash"] == "hash"
    assert dataset[0]["mopd_split"] == "train"


def make_batch():
    # Prompt width 2, response width 3; teacher IDs already left-shifted by verl.
    response = torch.tensor([[7,8,0]] * 4)
    td = TensorDict({"prompts": torch.ones(4,2,dtype=torch.long), "responses": response,
                    "response_mask": torch.tensor([[1,1,0]]*4),
                    "teacher_ids": torch.tensor([[[6],[7],[8],[0],[0]]]*4),
                    "teacher_logprobs": -torch.ones(4,5,1), "old_log_probs": -torch.ones(4,3)*2,
                    "token_level_rewards": torch.zeros(4,3)}, batch_size=[4])
    return DataProto(batch=td, non_tensor_batch={"data_source":np.array(DOMAINS),"index":np.array(DOMAINS)})


def test_actual_padding_alignment_and_batch_balance_reorder(spec):
    batch = make_batch()
    batch.reorder(torch.tensor([3,2,0,1]))
    result = check_batch(batch, dict(spec, train_batch_size=4), "domain_mix", 1, list(DOMAINS))
    assert result["domains"]["math"]["mean_teacher_minus_student"] == 1


@pytest.mark.parametrize("corruption", ["alignment", "nan", "reward", "empty", "ids", "domain"])
def test_refuse_silent_experiment_corruption(spec, corruption):
    batch = make_batch()
    if corruption == "alignment": batch.batch["teacher_ids"][0,1,0] = 42
    if corruption == "nan": batch.batch["teacher_logprobs"][0,1,0] = float("nan")
    if corruption == "reward": batch.batch["token_level_rewards"][0,0] = 1
    if corruption == "empty": batch.batch["response_mask"][0,:] = 0
    if corruption == "ids": batch.non_tensor_batch["index"][0] = "bad"
    if corruption == "domain": batch.non_tensor_batch["data_source"][0] = "if"
    with pytest.raises(RuntimeError):
        check_batch(batch, dict(spec, train_batch_size=4), "domain_mix", 1, list(DOMAINS))


def test_task_scorer_is_not_invoked_by_training(monkeypatch):
    from d4_mopd.agent_loop import MOPDAgentLoop, MultiDomainAgentLoop
    async def generate(*args, **kwargs):
        return SimpleNamespace(response_ids=[1], reward_score=None, extra_fields={})
    monkeypatch.setattr(MultiDomainAgentLoop, "run", generate)
    loop = object.__new__(MOPDAgentLoop)
    assert asyncio.run(loop.run({}, mopd_split="train")).reward_score == 0
    assert asyncio.run(loop.run({}, mopd_split="validation")).reward_score is None
    from d4_mopd.reward import compute_score
    with pytest.raises(RuntimeError, match="bypass"):
        asyncio.run(compute_score("math", "answer", "truth", extra_info={"split":"train"}))


def test_validation_bridge_uses_isolated_subprocess_and_keeps_provenance(tmp_path):
    from d4_mopd.reward import compute_score
    package = tmp_path / "verifier/md_reward_roundoff_0390"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def compute_score(data_source, solution_str, ground_truth, extra_info):\n"
        "    assert extra_info['num_turns'] == 2\n"
        "    assert extra_info['raw_completion'] == '<tool_call>original</tool_call>'\n"
        "    return {'score': 0.75, 'acc': 0.75}\n")
    path = tmp_path / "manifest.json"
    write(path, dict(manifest_path=str(path), verifier_python=sys.executable))
    result = asyncio.run(compute_score("math", "decoded", "truth", manifest_path=str(path),
        extra_info={"split":"validation", "num_turns":np.int64(2), "source":"deepmath_l8",
                    "sample_id":"sample1", "raw_completion":"<tool_call>original</tool_call>"}))
    assert result == dict(score=.75, acc=.75, mopd_source="deepmath_l8", mopd_domain="math", mopd_sample_id="sample1")


def test_math_validation_family_mean_is_not_row_weighted():
    extra = {"reward":[0.,1.,1.,1.], "mopd_source":["nemotron", "deepmath_l8","deepmath_l8","deepmath_l9"],
             "mopd_domain":["math"]*4, "mopd_sample_id":["a","b","c","d"]}
    out = source_metrics(extra)
    assert out["val-source/math/equal_source_mean"] == .5
    assert out["val-source/math/deepmath/n"] == 3
    extra["mopd_sample_id"][-1] = "a"
    with pytest.raises(RuntimeError, match="duplicate"):
        source_metrics(extra)


def test_formal_start_requires_arm_specific_gpu_resume_proof(manifest, tmp_path, monkeypatch):
    from d4_mopd import control
    monkeypatch.setattr(control, "verify", lambda p: manifest)
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"]))
    q = tmp_path / "qualification/domain_mix/receipt.json"
    write(q, dict(status="PASS", identity=manifest["identity"], resumed_from=0))
    with pytest.raises(RuntimeError, match="qualification"):
        control.start(manifest["manifest_path"], "train", ["domain_mix"])
    assert not (tmp_path / "states.json").exists()
