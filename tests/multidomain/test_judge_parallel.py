"""Regression checks for the Qwen3-235B FP8 TP8 block-alignment failure."""
import copy
import json
import unittest
from pathlib import Path

import yaml

from multidomain.common import DataError
from multidomain.preflight import check_judge_fp8_parallelism
from multidomain.preparation.check_config import resolve
from multidomain.runtime_config import overrides


class JudgeParallelTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.c = yaml.safe_load((root / 'config/multidomain/initial.yaml').read_text())
        self.model = {'moe_intermediate_size': 1536, 'num_experts': 128,
                      'quantization_config': {'weight_block_size': [128, 128]}}

    def test_previous_tp8_recipe_fails_before_model_loading(self):
        old = copy.deepcopy(self.c['judge'])
        old['expert_parallel_size'] = 1
        with self.assertRaisesRegex(DataError, '192'):
            check_judge_fp8_parallelism(old, self.model)

    def test_ep8_reaches_runtime_and_preserves_one_replica(self):
        c = resolve(self.c)
        arguments = dict(item[2:].split('=', 1) for item in overrides(c, '/data', '/run', 'smoke') if item.startswith('++'))
        self.assertEqual(json.loads(arguments['reward.reward_model.rollout.expert_parallel_size']), 8)
        self.assertEqual(json.loads(arguments['reward.reward_model.rollout.tensor_model_parallel_size']), 8)
        self.assertEqual(c['resolved']['judge_replicas'], 1)
        report = check_judge_fp8_parallelism(c['judge'], self.model)
        self.assertEqual(report['expert_width_per_rank'], 1536)
        self.assertEqual(report['moe_tp'], 1)

    def test_ep_is_not_an_independent_extra_gpu_dimension(self):
        self.c['judge']['expert_parallel_size'] = 4
        with self.assertRaisesRegex(ValueError, 'EP must equal'):
            resolve(self.c)

    def test_unsupported_pipeline_parallelism_rejected(self):
        self.c['judge']['pipeline_parallel_size'] = 2
        with self.assertRaisesRegex(ValueError, 'pipeline parallelism'):
            resolve(self.c)


if __name__ == '__main__':
    unittest.main()
