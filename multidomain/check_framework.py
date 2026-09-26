"""CPU check of the real pinned verl data/reward interfaces; generation transport is mocked."""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from multidomain.common import ROOT, read_config


async def check(c, data_dir):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from transformers import AutoTokenizer
    from multidomain.runtime_config import compose
    from multidomain.dataset import MultiDomainDataset
    from multidomain.agent_loop import MultiDomainAgentLoop
    from multidomain.reward import compute_score, local_score
    from multidomain.template import parse_completion
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
    from verl.protocol import DataProto
    cfg = compose(c, data_dir, Path(data_dir).parent / 'framework_check', 'smoke')
    tok = AutoTokenizer.from_pretrained(c['training']['model_path'], local_files_only=True)
    dataset = MultiDomainDataset(str(Path(data_dir) / 'train.parquet'), tok, cfg.data)
    row = next((r for r in dataset if r['data_source'] == 'swe_pivot'), None)
    if row is None:
        row = next(r for r in dataset if r['data_source'] == 'if')
    payload = json.loads(row['reward_model']['ground_truth'])
    action = payload.get('expected_action')
    text = 'invalid answer'
    if action:
        text = action['content'] if action['type'] == 'message' else '<tool_call>' + json.dumps(
            {'name': action['name'], 'arguments': json.loads(action['arguments'])}) + '</tool_call>'
    expected = local_score(payload, parse_completion(text)[0])
    text += '<|im_end|>'
    response = tok.encode(text, add_special_tokens=False)
    class Server:
        async def generate(self, **kwargs):
            assert kwargs['sampling_params']['max_tokens'] == row['extra_info']['generation_max_tokens']
            return SimpleNamespace(token_ids=response, log_probs=None, routed_experts=None, num_preempted=0, extra_fields={})
    loop = MultiDomainAgentLoop(DictConfigWrap(cfg), Server(), tok, None, MultiDomainDataset, DictConfigWrap(cfg.data))
    output = await loop.run({'temperature': 1.0, 'top_p': 1.0}, **row)
    assert output.extra_fields['raw_completion'] == text
    length = len(output.prompt_ids) + len(response)
    data = DataProto(batch=TensorDict({'responses': torch.tensor([response]),
        'attention_mask': torch.ones((1, length), dtype=torch.long)}, batch_size=[1]), non_tensor_batch={
        'data_source': np.array([row['data_source']], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
        'extra_info': np.array([row['extra_info']], dtype=object),
        'tool_extra_fields': np.array([output.extra_fields], dtype=object)})
    scorer = NaiveRewardManager(cfg, tok, compute_score)
    result = await scorer.run_single(data)
    assert result['reward_score'] == expected, result
    row['extra_info']['prompt_token_hash'] = 'changed'
    try:
        await loop.run({'temperature': 1.0}, **row)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed prompt fingerprint was accepted')
    print('PASS: pinned verl AgentLoopOutput -> DataProto -> NaiveRewardManager; generation transport mocked')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(ROOT / 'config/multidomain/initial.yaml'))
    ap.add_argument('--data-dir', required=True)
    a = ap.parse_args()
    asyncio.run(check(read_config(a.config), a.data_dir))

if __name__ == '__main__':
    main()
