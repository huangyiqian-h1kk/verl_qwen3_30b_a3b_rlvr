"""Dataset for the pinned verl AgentLoop architecture, with uniform Arrow columns."""
import json
import torch
from torch.utils.data import Dataset
from multidomain.common import digest, DataError
from multidomain.template import render


class MultiDomainDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        import pyarrow.parquet as pq
        files = [data_files] if isinstance(data_files, str) else list(data_files)
        self.rows = [row for file in files for row in pq.read_table(file).to_pylist()]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.tokenizer = tokenizer
        if processor is not None:
            raise DataError('This experiment supports text-only models')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        messages, tools = json.loads(row['messages_json']), json.loads(row['tools_json'])
        return {'dummy_tensor': torch.tensor([0], dtype=torch.uint8),
                'raw_prompt': messages, 'tools_json': row['tools_json'],
                'data_source': row['domain'], 'agent_name': 'multidomain_single_turn',
                'reward_model': {'style': 'rule', 'ground_truth': row['verifier_json']},
                'extra_info': {'index': row['sample_id'], 'sample_id': row['sample_id'], 'source': row['source'],
                               'prompt_tokens': row['prompt_tokens'], 'prompt_token_hash': row['prompt_token_hash'],
                               'generation_max_tokens': row['generation_max_tokens']},
                'index': row['sample_id'], 'tools_kwargs': {}, 'interaction_kwargs': {}}

    def resume_dataset_state(self):
        pass  # Rows are serialized in the stateful dataloader checkpoint.
