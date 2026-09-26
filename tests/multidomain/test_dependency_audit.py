import copy
import unittest
from multidomain.dependencies import analyze_snapshots, EXPECTED, KNOWN_BASELINE


def fixture(issues=()):
    versions = dict(EXPECTED, **{
        'math-verify': '0.9.0', 'latex2sympy2-extended': '1.11.0',
        'outlines': '0.1.11', 'outlines-core': '0.2.11', 'vllm': '0.11.0',
        'numpy': '2.2.6', 'megatron-core': '0.13.1', 'decord': '0.6.0',
        'torch': '2.8.0',
    })
    base = {'packages': {n: {'version': v, 'location': '/base/site'} for n, v in versions.items()},
            'pip_check': {'returncode': 1 if issues else 0, 'issues': list(issues)}}
    project = copy.deepcopy(base)
    for name, version in EXPECTED.items():
        project['packages'][name] = {'version': version, 'location': '/project/site'}
    return base, copy.deepcopy(base), project


class DependencyAuditTests(unittest.TestCase):
    def test_clean_base_and_valid_scoring_overlay(self):
        result = analyze_snapshots(*fixture())
        self.assertEqual(result['status'], 'CPU_DEPENDENCIES_READY')

    def test_observed_conflicts_remain_explicitly_unresolved(self):
        result = analyze_snapshots(*fixture(list(KNOWN_BASELINE)[:2]))
        self.assertEqual(result['status'], 'CPU_DEPENDENCIES_READY_WITH_INHERITED_CONFLICTS')
        self.assertEqual(len(result['inherited_conflicts']), 2)

    def test_platform_diagnostic_does_not_disappear_from_report(self):
        args = fixture(KNOWN_BASELINE)
        args[2]['pip_check']['issues'].remove('decord 0.6.0 is not supported on this platform')
        result = analyze_snapshots(*args)
        self.assertEqual(result['base_only_diagnostics'], ['decord 0.6.0 is not supported on this platform'])
        self.assertEqual(len(result['inherited_conflicts']), 3)
        self.assertEqual(result['errors'], [])

    def test_new_conflict_rejected_even_if_it_is_a_known_message(self):
        args = fixture()
        args[2]['pip_check'] = {'returncode': 1, 'issues': [next(iter(KNOWN_BASELINE))]}
        self.assertEqual(analyze_snapshots(*args)['status'], 'FAIL')

    def test_unrecognized_baseline_conflict_is_not_auto_accepted(self):
        self.assertEqual(analyze_snapshots(*fixture(['other package mismatch']))['status'], 'FAIL')

    def test_core_shadowing_rejected_even_at_same_version(self):
        args = fixture()
        args[2]['packages']['torch']['location'] = '/project/site'
        self.assertEqual(analyze_snapshots(*args)['status'], 'FAIL')

    def test_base_modification_rejected(self):
        args = fixture()
        args[1]['packages']['numpy']['version'] = '1.26.4'
        self.assertEqual(analyze_snapshots(*args)['status'], 'FAIL')

    def test_wrong_scoring_version_rejected(self):
        args = fixture()
        args[2]['packages']['xmltodict']['version'] = '0.15.1'
        self.assertEqual(analyze_snapshots(*args)['status'], 'FAIL')

    def test_pip_process_error_is_not_treated_as_a_dependency_warning(self):
        args = fixture()
        args[2]['pip_check'] = {'returncode': 2, 'issues': []}
        self.assertEqual(analyze_snapshots(*args)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
