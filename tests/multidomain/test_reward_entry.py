"""Regression for verl's temporary file-module name versus multiprocessing spawn."""
import importlib.util
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import pickle
import unittest


ROOT = Path(__file__).resolve().parents[2]


class RewardEntryTests(unittest.TestCase):
    def test_old_file_loader_reproduces_unpicklable_child(self):
        from verl.utils.import_utils import load_extern_object
        old = load_extern_object(str(ROOT / 'multidomain/reward.py'), 'compute_score')
        self.assertTrue(old.__module__.startswith('custom_module_'))
        with self.assertRaises(pickle.PicklingError):
            ForkingPickler.dumps(old.__globals__['_math_child'])

    def test_configured_loader_scores_in_real_spawn_children(self):
        path = ROOT / 'scripts/multidomain/check_reward_entry.py'
        spec = importlib.util.spec_from_file_location('md_check_reward_entry', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.check_entry(ROOT)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['spawn_target_module'], 'multidomain.reward')
        self.assertEqual(len(report['cases']), 4)


if __name__ == '__main__':
    unittest.main()
