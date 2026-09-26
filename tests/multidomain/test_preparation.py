"""CPU checks for config switches, sample balance, leakage and reproducibility."""
import copy
import itertools
import sys
import unittest
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from multidomain.preparation.balance_index import make_selection, select
from multidomain.preparation.check_config import OPTIONAL, resolve, response_budget


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((ROOT / 'config/multidomain/initial.yaml').read_text())
        cls.rows = []
        for domain, cfg in cls.config['domains'].items():
            for source in cfg['sources']:
                size = 450 if domain == 'swe_pivot' else 700
                for i in range(size):
                    cls.rows.append({'sample_id': f'{source}/{i}', 'domain': domain, 'source': source,
                                     'split_group_id': f'{source}/group/{i // 3}', 'prompt_tokens': 100,
                                     'eligible': True})

    def test_all_sixteen_domain_combinations(self):
        for flags in itertools.product((False, True), repeat=len(OPTIONAL)):
            disabled = [d for d, enabled in zip(OPTIONAL, flags) if not enabled]
            c = resolve(self.config, disabled)
            self.assertEqual(c['resolved']['domain_count'], 2 + sum(flags))
            self.assertEqual(c['resolved']['pbs_nodes_for_training_job'], 1 + int(flags[0]))
            chosen, _, m = make_selection(self.rows, c)
            counts = Counter(r['domain'] for r in chosen['train'])
            self.assertEqual(len(set(counts.values())), 1)
            self.assertEqual(next(iter(counts.values())), min(m['eligible_pool_counts']['train'].values()))
            ids = [r['sample_id'] for r in chosen['train']]
            self.assertEqual(len(ids), len(set(ids)))
            by_if = Counter(r['source'] for r in chosen['train'] if r['domain'] == 'if')
            self.assertLessEqual(max(by_if.values()) - min(by_if.values()), 1)

    def test_no_group_leakage_and_input_order_invariance(self):
        chosen, assignments, m = make_selection(self.rows, self.config)
        reverse, _, _ = make_selection(reversed(self.rows), self.config)
        for split in ('train', 'validation', 'test'):
            self.assertEqual([r['sample_id'] for r in chosen[split]], [r['sample_id'] for r in reverse[split]])
        for a, b in itertools.combinations(('train', 'validation', 'test'), 2):
            self.assertFalse({r['split_group_id'] for r in chosen[a]} & {r['split_group_id'] for r in chosen[b]})
        self.assertEqual(m['group_intersections'], 0)
        assignment_map = {r['sample_id']: r['split'] for r in assignments}
        _, other_assignments, _ = make_selection(self.rows, resolve(self.config, ['swe_pivot']))
        for r in other_assignments:
            self.assertEqual(r['split'], assignment_map[r['sample_id']])

    def test_global_groups_cross_domains(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['split_group_id'] = rows[-1]['split_group_id'] = 'shared/problem'
        _, assignments, _ = make_selection(rows, self.config)
        values = {r['split'] for r in assignments if r['split_group_id'] == 'shared/problem'}
        self.assertEqual(len(values), 1)

    def test_length_filter_before_balance(self):
        rows = copy.deepcopy(self.rows)
        for r in rows:
            if r['domain'] == 'swe_pivot' and int(r['sample_id'].split('/')[-1]) % 2 == 0:
                r['prompt_tokens'] = self.config['data']['max_prompt_tokens'] + 1
        chosen, _, m = make_selection(rows, self.config)
        self.assertGreater(m['excluded_index_rows']['overlength'], 0)
        self.assertTrue(all(r['prompt_tokens'] <= self.config['data']['max_prompt_tokens'] for r in chosen['train']))
        self.assertEqual(set(m['bottleneck_domains']), {'swe_pivot'})

    def test_duplicate_ids_and_missing_provenance_fail(self):
        with self.assertRaises(ValueError):
            make_selection(self.rows + [self.rows[0]], self.config)
        bad = copy.deepcopy(self.rows)
        del bad[0]['split_group_id']
        with self.assertRaises(ValueError):
            make_selection(bad, self.config)

    def test_required_domains_cannot_be_disabled(self):
        with self.assertRaises(ValueError):
            resolve(self.config, ['math'])
        bad = copy.deepcopy(self.config)
        bad['domains']['if']['enabled'] = False
        with self.assertRaises(ValueError):
            resolve(bad)

    def test_unintended_judge_replication_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad['judge']['tensor_parallel_size'] = 4
        with self.assertRaises(ValueError):
            resolve(bad)

    def test_inner_source_balance_respects_capacity(self):
        rows = [{'sample_id': f'{s}/{i}', 'source': s} for s, n in [('a', 2), ('b', 10), ('c', 10)] for i in range(n)]
        selected = select(rows, 12, 42, 'test', balance_sources=True)
        self.assertEqual(Counter(r['source'] for r in selected), {'a': 2, 'b': 5, 'c': 5})

    def test_shared_length_budget_and_native_request_limits(self):
        data = self.config['data']
        self.assertEqual(response_budget(2048, data), 63488)
        self.assertEqual(response_budget(49152, data), 16384)
        self.assertEqual(response_budget(61440, data), 4096)
        self.assertEqual(response_budget(65535, data), 1)
        self.assertEqual(response_budget(65536, data), 0)
        self.assertEqual(response_budget(2048, data, 8192), 8192)
        self.assertEqual(response_budget(61440, data, 8192), 4096)
        with self.assertRaises(ValueError):
            response_budget(2048, data, 0)


if __name__ == '__main__':
    unittest.main()
