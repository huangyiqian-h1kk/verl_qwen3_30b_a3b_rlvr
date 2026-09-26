#!/usr/bin/env python3
"""Validate the proposal and resolve optional domains. Does not start training."""
import argparse
import copy
import itertools
import json
from pathlib import Path

import yaml

OPTIONAL = ('science', 'conversational_pivot', 'swe_pivot', 'logic_algorithmic')


def response_budget(prompt_tokens, data, request_limit=None):
    """Shared context cap, calculated after the full target chat template."""
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise ValueError('prompt_tokens must be a positive integer')
    if request_limit is not None and (not isinstance(request_limit, int) or request_limit <= 0):
        raise ValueError('Per-record output limit must be a positive integer or null')
    if prompt_tokens > data['max_prompt_tokens']:
        return 0
    remaining = data['max_context_tokens'] - prompt_tokens
    if remaining < data['minimum_generation_headroom']:
        return 0
    return min(data['max_response_tokens'], remaining,
               request_limit if request_limit is not None else data['max_response_tokens'])


def resolve(config, disabled=()):
    c = copy.deepcopy(config)
    for name in disabled:
        if name not in OPTIONAL:
            raise ValueError(f'Only optional domains can be disabled: {OPTIONAL}; got {name!r}')
        c['domains'][name]['enabled'] = False
    if set(c['domains']) != {'math', 'if', *OPTIONAL}:
        raise ValueError('Unexpected domain registry')
    for name in ('math', 'if'):
        if c['domains'][name]['enabled'] is not True:
            raise ValueError(f'{name} is mandatory in this experiment')
    active = [k for k, v in c['domains'].items() if v['enabled']]
    if len(c['domains']['if']['sources']) != 3:
        raise ValueError('IF must contain all three selected sources')
    d, t, j = c['data'], c['training'], c['judge']
    if abs(sum(d['split_ratios']) - 1) > 1e-10 or any(x <= 0 for x in d['split_ratios']):
        raise ValueError('Invalid split ratios')
    if d['length_mode'] != 'shared_context':
        raise ValueError('This revision uses a shared context budget')
    if not 0 < d['max_response_tokens'] <= d['max_context_tokens']:
        raise ValueError('Invalid generation ceiling')
    if not 0 < d['minimum_generation_headroom'] < d['max_context_tokens']:
        raise ValueError('Invalid generation headroom')
    if not 0 < d['max_prompt_tokens'] <= d['max_context_tokens'] - d['minimum_generation_headroom']:
        raise ValueError('Prompt ceiling leaves no generation headroom')
    if t['ppo_max_token_len_per_gpu'] * t['actor_cp'] < d['max_context_tokens']:
        raise ValueError('Dynamic microbatch token budget cannot fit the longest single sequence')
    if t['log_prob_max_token_len_per_gpu'] * t['actor_cp'] < d['max_context_tokens']:
        raise ValueError('Log-probability token budget cannot fit the longest single sequence')
    if t['train_batch_size'] % t['ppo_mini_batch_size']:
        raise ValueError('Train batch is not divisible by PPO minibatch')
    train_gpus = t['train_nodes'] * t['gpus_per_node']
    if train_gpus % (t['actor_tp'] * t['actor_pp'] * t['actor_cp']):
        raise ValueError('Actor model parallelism does not divide training GPUs')
    if train_gpus % t['rollout_tp']:
        raise ValueError('Rollout TP does not divide training GPUs')
    use_judge = c['domains'][j['enabled_when']]['enabled']
    judge_gpus = j['nodes'] * j['gpus_per_node'] if use_judge else 0
    judge_parallel = j['tensor_parallel_size'] * j['data_parallel_size'] * j['pipeline_parallel_size']
    replicas = judge_gpus // judge_parallel if use_judge else 0
    if use_judge and (judge_gpus % judge_parallel or replicas != j['expected_replicas']):
        raise ValueError('Judge GPU allocation produces an unintended number of model replicas')
    if j['max_output_tokens'] >= j['max_model_len']:
        raise ValueError('Judge has no input-token budget')
    if c['acceptance']['optimizer_steps'] != 0 or c['acceptance']['model_test_mode'] != 'val_only':
        raise ValueError('Initial acceptance is a no-update check')
    if t['intermediate_weight_only_snapshots'] or t['max_actor_checkpoints'] != 2:
        raise ValueError('Checkpoint policy differs from the agreed initial plan')
    c['resolved'] = {
        'enabled_domains': active,
        'domain_count': len(active),
        'domain_target_fraction': {name: 1 / len(active) for name in active},
        'judge_enabled': use_judge,
        'pbs_nodes_for_training_job': t['train_nodes'] + (j['nodes'] if use_judge else 0),
        'train_gpus': train_gpus,
        'judge_gpus': judge_gpus,
        'judge_replicas': replicas,
        'training_rollout_replicas': train_gpus // t['rollout_tp'],
        'rollouts_per_training_batch': t['train_batch_size'] * t['rollout_n'],
        'balanced_sample_count': 'computed from eligible train pools, never from published raw counts',
    }
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'config/multidomain/initial.yaml')
    ap.add_argument('--disable', choices=OPTIONAL, action='append', default=[])
    ap.add_argument('--output', type=Path)
    ap.add_argument('--all-combinations', action='store_true')
    a = ap.parse_args()
    c = yaml.safe_load(a.config.read_text())
    if a.all_combinations:
        output = []
        for switches in itertools.product((False, True), repeat=len(OPTIONAL)):
            disabled = [d for d, on in zip(OPTIONAL, switches) if not on]
            output.append(resolve(c, disabled)['resolved'])
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    c = resolve(c, a.disable)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(c, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(c['resolved'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
