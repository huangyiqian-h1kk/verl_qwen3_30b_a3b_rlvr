#!/usr/bin/env python3
"""Exercise the configured verl reward loader and real spawn-based math scoring."""
import argparse
import asyncio
import json
from pathlib import Path
import sys


def check_entry(project_root):
    root = Path(project_root)
    sys.path.insert(0, str(root))
    from multidomain.common import read_config
    from multidomain.runtime_config import overrides
    from omegaconf import OmegaConf
    from verl.trainer.ppo.reward import get_custom_reward_fn
    c = read_config(root / 'config/multidomain/initial.yaml')
    # Read the actual application override, not a separately hard-coded loader.
    arguments = overrides(c, root / 'data', root / 'outputs', 'smoke')
    key = '++reward.custom_reward_function.path='
    paths = [json.loads(item[len(key):]) for item in arguments if item.startswith(key)]
    if paths != ['pkg://multidomain.reward']:
        raise RuntimeError(f'Unexpected reward entrypoint: {paths}')
    config = OmegaConf.create({'reward': {'custom_reward_function': {
        'path': paths[0], 'name': 'compute_score'}}})
    scorer = get_custom_reward_fn(config)
    raw = scorer.args[0]
    if raw.__module__ != 'multidomain.reward':
        raise RuntimeError(f'Reward was loaded under a temporary module: {raw.__module__}')
    from multiprocessing.reduction import ForkingPickler
    child = raw.__globals__['_math_child']
    ForkingPickler.dumps(child)
    cases = [('integer_correct', '2', r'\boxed{2}', 1.0),
             ('integer_incorrect', '2', r'\boxed{3}', 0.0),
             ('empty_answer', '2', '', 0.0),
             ('fraction_equivalent', '1/2', r'\boxed{\frac{1}{2}}', 1.0)]

    async def evaluate():
        results = []
        for name, expected, generated, target in cases:
            value = await scorer(data_source='math', solution_str=generated,
                ground_truth=json.dumps({'source': 'math', 'expected_answer': expected}),
                extra_info={'raw_completion': generated})
            if value['score'] != target:
                raise RuntimeError(f'Math entry regression {name}: {value} != {target}')
            results.append({'case': name, 'actual': value['score'], 'expected': target})
        return results

    return {'status': 'PASS', 'entrypoint': paths[0], 'reward_module': raw.__module__,
            'spawn_target_module': child.__module__, 'cases': asyncio.run(evaluate()),
            'scope': 'pinned verl get_custom_reward_fn -> async compute_score -> thread -> real spawn child'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    report = check_entry(a.project_root.resolve())
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
