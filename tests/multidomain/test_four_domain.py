import contextlib
import copy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from multidomain.common import ROOT, DataError, digest, file_sha, read_config, write_json
from multidomain.build import choose, export_parquet
from multidomain.reselect import reselect
from multidomain.runtime_config import overrides


class FourDomainTests(unittest.TestCase):
    def configs(self):
        return (read_config(ROOT / 'config/multidomain/initial.yaml'),
                read_config(ROOT / 'config/multidomain/four_domain.yaml'))

    def test_four_domain_resource_plan(self):
        six, four = self.configs()
        self.assertEqual(six['resolved']['pbs_nodes_for_training_job'], 2)
        self.assertEqual(four['resolved']['pbs_nodes_for_training_job'], 1)
        self.assertEqual(four['training'], six['training'])
        self.assertEqual(four['resolved']['train_gpus'], 8)
        self.assertEqual(four['resolved']['judge_gpus'], 0)
        values = dict(item[2:].split('=', 1) for item in overrides(four, '/data', '/run', 'train')[1:])
        self.assertEqual(values['trainer.nnodes'], '1')
        self.assertEqual(values['reward.reward_model.enable'], 'false')
        self.assertEqual(values['reward.reward_model.enable_resource_pool'], 'false')
        self.assertEqual(values['reward.reward_model.nnodes'], '0')

    def parent(self, folder, config):
        folder.mkdir()
        index = []
        with sqlite3.connect(folder / 'canonical.sqlite') as db:
            db.execute('create table samples (id text primary key, dedup text unique, record text)')
            for domain, info in config['domains'].items():
                for source in info['sources']:
                    for i in range(120 if domain != 'swe_pivot' else 60):
                        sid = f'{source}/{i}'
                        idx = dict(sample_id=sid, domain=domain, source=source, category=str(i % 3),
                                   task='test', eligible=True, split_group_id=sid, prompt_tokens=i + 1,
                                   prompt_token_hash=digest(sid), generation_max_tokens=100)
                        row = dict(sample_id=sid, domain=domain, source=source, verifier_json='{}',
                                   messages_json='[]', tools_json='[]')
                        db.execute('insert into samples values (?, ?, ?)', (sid, sid, json.dumps(row)))
                        index.append(idx)
            selected, balance = choose(index, config)
            for split in ('train', 'validation', 'test'):
                export_parquet(db, selected[split], folder / (split + '.parquet'))
        (folder / 'index.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in index))
        write_json(folder / 'manifest.json', dict(config=config, config_sha256=digest(config),
            tokenizer_fingerprint='fixture', judge_tokenizer_fingerprint='fixture-judge', input_files=[],
            files={p.name: file_sha(p) for p in folder.glob('*.parquet')}, **balance,
            eval_policy='fixed', group_policy='global'))
        return balance

    def test_reselect_full_pool_preserves_heldout_and_parent(self):
        import pyarrow.parquet as pq
        six, four = self.configs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, output = root / 'six', root / 'four'
            old = self.parent(parent, six)
            before = {p.name: file_sha(p) for p in parent.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                reselect(four, parent, output, before['manifest.json'])
            result = json.loads((output / 'manifest.json').read_text())
            self.assertGreater(result['train_rows_per_domain'], old['train_rows_per_domain'])
            self.assertEqual(set(result['selected_counts']['train'].values()), {result['train_rows_per_domain']})
            self.assertEqual(set(result['selected_counts']['train']), set(four['resolved']['enabled_domains']))
            self.assertIsNone(result['judge_tokenizer_fingerprint'])
            for split in ('validation', 'test'):
                expected = [r['sample_id'] for r in pq.read_table(parent / (split + '.parquet')).to_pylist()
                            if r['domain'] in four['resolved']['enabled_domains']]
                self.assertEqual(expected, json.loads((output / (split + '_ids.json')).read_text()))
            self.assertEqual(before, {p.name: file_sha(p) for p in parent.iterdir()})
            for name in ('smoke', 'baseline_train'):
                self.assertEqual(set(pq.read_table(output / (name + '.parquet'))['domain'].to_pylist()),
                                 set(four['resolved']['enabled_domains']))
            with self.assertRaisesRegex(DataError, 'not empty'):
                reselect(four, parent, output, before['manifest.json'])

    def test_changed_seed_or_parent_is_rejected_before_output(self):
        six, four = self.configs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.parent(root / 'six', six)
            expected = file_sha(root / 'six/manifest.json')
            altered = copy.deepcopy(four)
            altered['seed'] += 1
            with self.assertRaisesRegex(DataError, 'Reuse permits'):
                reselect(altered, root / 'six', root / 'four', expected)
            with self.assertRaisesRegex(DataError, 'Parent manifest'):
                reselect(four, root / 'six', root / 'four', 'bad-checksum')
            self.assertFalse((root / 'four').exists())

    def test_pbs_single_node_and_scheduler_environment(self):
        from multidomain import submit
        _, four = self.configs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            four['project_root'] = str(root)
            for stage in ('check-infra', 'accept-four', 'smoke', 'train'):
                argv = ['submit', '--stage', stage, '--source-data-dir', '/parent',
                        '--source-manifest-sha256', 'a' * 64]
                with patch.object(submit, 'ROOT', root), patch.object(submit, 'read_config', return_value=four), \
                        patch('sys.argv', argv), patch.dict('os.environ', PYTHONPATH='bad', PYTHONHOME='bad'), \
                        patch.object(submit.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                    run.return_value.stdout = '123.pbs1\n'
                    submit.main()
                    self.assertNotIn('PYTHONPATH', run.call_args.kwargs['env'])
                    self.assertNotIn('PYTHONHOME', run.call_args.kwargs['env'])
            jobs = list((root / 'outputs/multidomain').glob('*/jobs/*'))
            self.assertEqual(len(jobs), 4)
            for job in jobs:
                self.assertEqual(json.loads((job / 'request.json').read_text())['expected_nodes'], 1)
                pbs = (job / 'stage.pbs').read_text()
                for directive in ('#PBS -P gcg51557', '#PBS -q R9920261000', '#PBS -v RTYPE=rt_HF',
                                  '#PBS -l select=1', '#PBS -N 0390_d4_', '#PBS -j oe', '#PBS -k oe'):
                    self.assertIn(directive, pbs)
                self.assertNotIn('host=hnode', pbs)

    def test_infra_receives_allocated_node_count(self):
        from multidomain import stage
        _, config = self.configs()
        with tempfile.TemporaryDirectory() as temporary:
            req = dict(stage='check-infra', config=config, data_dir=temporary + '/data',
                       run_dir=temporary + '/run', expected_nodes=1)
            with patch.object(stage.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                stage.execute(req)
            argv = run.call_args.args[0]
            self.assertEqual(argv[argv.index('--expected-nodes') + 1], '1')

    def test_acceptance_stops_on_failure_and_resumes_matching_preparation(self):
        from multidomain import stage
        _, config = self.configs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req = dict(stage='accept-four', config=config, data_dir=str(root / 'data'),
                       run_dir=str(root / 'run'), expected_nodes=1)
            write_json(root / 'request.json', req)
            fingerprint = digest(dict(config=digest(config), code=stage.code_fingerprint(), data_id='data'))
            def first_attempt(item):
                if item['stage'] == 'prepare-data':
                    write_json(root / 'data/manifest.json', {'fixture': True})
                    write_json(root / 'run/reports/prepare-data.json', dict(status='PASS', fingerprint=fingerprint))
                else:
                    raise RuntimeError('verifier fixture failure')
            with patch('sys.argv', ['stage', '--request', str(root / 'request.json')]), \
                    patch.object(stage, 'execute', side_effect=first_attempt) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'verifier fixture failure'):
                    stage.main()
                self.assertEqual([call.args[0]['stage'] for call in run.call_args_list], ['prepare-data', 'check-verifiers'])
            with patch('sys.argv', ['stage', '--request', str(root / 'request.json')]), \
                    patch('multidomain.preflight.check'), patch.object(stage, 'execute') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                stage.main()
                self.assertEqual([call.args[0]['stage'] for call in run.call_args_list],
                                 ['check-verifiers', 'check-infra', 'smoke'])


if __name__ == '__main__':
    unittest.main()
