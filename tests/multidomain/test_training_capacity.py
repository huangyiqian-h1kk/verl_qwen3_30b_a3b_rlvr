"""CPU tests: no real scheduler, controller daemon, or GPU submissions."""
import copy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import io
import subprocess
from contextlib import redirect_stdout
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/multidomain/manage_training_capacity.py'
spec = importlib.util.spec_from_file_location('capacity', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def states():
    return {k: dict(phase='WAIT_READY', single={'job_id': k + '_s'}, dual={'job_id': k + '_d'})
            for k in m.ROLES}


def capacity(free, own=0):
    return dict(free_slots=free, available_to_experiment=free + own)


def one_host(name='hnode001', busy=False):
    nodes = {name: dict(state='free', resources_available=dict(host=name, ncpus=0, ngpus=0, node_group='nii'))}
    for i in (0, 1):
        nodes[name + f'[{i}]'] = dict(state='free', resv=m.RESERVATION,
            resources_available=dict(host='@' + name, ncpus=96, ngpus=4, node_group='@' + name),
            resources_assigned=dict(ncpus=96 if busy else 0, ngpus=4 if busy else 0),
            jobs='123.pbs1/0' if busy else '')
    return nodes


# Queue-profile shape; resource values are synthetic. Only this repository's
# existing queue/project/node-group identifiers are retained.
QUEUE_PROFILE = '''Queue: R9920261000
    queue_type = Execution
    acl_group_enable = True
    acl_groups = gcg51557
    default_chunk.node_group = nii
    resources_available.ncpus = 1536
    resources_assigned.ncpus = 0
    enabled = True
    started = True
'''


def pool_host(name='hnode001', busy=False):
    nodes = one_host(name, busy)
    for row in nodes.values():
        row.pop('resv', None)
    return nodes


class QueuePoolTests(unittest.TestCase):
    def test_queue_profile_without_qrstat_counts_only_eligible_nii_hosts(self):
        nodes = {**pool_host(), **pool_host('hnode002', busy=True), **pool_host('hnode003'),
                 **pool_host('hnode004'), **pool_host('hnode005')}
        nodes['hnode003']['resources_available']['node_group'] = 'frai01'
        nodes['hnode004[0]']['resv'] = 'R777.pbs1'
        nodes['hnode005']['state'] = 'offline'
        result = m.free_hosts({'nodes': nodes}, None, m.attributes(QUEUE_PROFILE))
        self.assertEqual(result['capacity_source'], 'queue_node_group')
        self.assertEqual(result['free_hosts'], ['hnode001'])
        self.assertNotIn('hnode003', result['pool_hosts'])
        self.assertIn('other_reservation', result['hosts']['hnode004'])

    def test_pool_still_requires_whole_host_and_queue_cpu_headroom(self):
        nodes = {**pool_host(), **pool_host('hnode002')}
        nodes['hnode002[1]']['resources_available']['node_group'] = 'general'
        queue = m.attributes(QUEUE_PROFILE)
        result = m.free_hosts({'nodes': nodes}, None, queue)
        self.assertEqual(result['free_slots'], 1)
        self.assertIn('wrong_node_group', result['hosts']['hnode002'])
        queue['resources_assigned.ncpus'] = '1400'
        self.assertEqual(m.free_hosts({'nodes': nodes}, None, queue)['free_slots'], 0)

    def test_unrecognized_queue_profile_is_not_a_global_pool_fallback(self):
        for key, value in [('queue_type', 'Route'), ('acl_group_enable', 'False'),
                           ('acl_groups', 'other'), ('default_chunk.node_group', 'general'),
                           ('resources_available.ncpus', '0')]:
            with self.subTest(key=key):
                queue = dict(m.attributes(QUEUE_PROFILE), **{key: value})
                with self.assertRaises(RuntimeError):
                    m.free_hosts({'nodes': pool_host()}, None, queue)

    def test_live_capacity_uses_only_queue_and_nodes_queries(self):
        nodes = {**pool_host(), **pool_host('hnode002', busy=True)}
        replies = {('qstat', '-Qf', m.QUEUE): QUEUE_PROFILE,
                   ('pbsnodes', '-av', '-F', 'json'): json.dumps({'nodes': nodes})}
        rows = {('domain', 'single'): dict(job_state='R', Resource_List={'nodect': 1},
                exec_vnode='(hnode002[0]:ncpus=96+hnode002[1]:ncpus=96)')}
        with tempfile.TemporaryDirectory() as temp, patch.object(m, 'command', side_effect=lambda args: replies[tuple(args)]), \
                patch.object(m, 'reservation_for_queue', side_effect=AssertionError('qrstat must not gate this queue')):
            result = m.Policy(temp).capacity(rows)
            self.assertEqual(result['available_to_experiment'], 2)
            self.assertEqual(result['owned_hosts'], ['hnode002'])


class TargetedQueryTests(unittest.TestCase):
    def test_only_registered_ids_are_queried_including_finished_jobs(self):
        s = {'domain': {'single': {'job_id': '12.pbs1'}, 'dual': {'job_id': '13.pbs1'}},
             'random': {'single': {'job_id': '14.pbs1'}, 'dual': None}}
        jobs = {'12.pbs1': {'job_state': 'F'}, '13.pbs1': {'job_state': 'H'}, '14.pbs1': {'job_state': 'R'}}
        with tempfile.TemporaryDirectory() as temp, patch.object(m, 'command', return_value=json.dumps({'Jobs': jobs})) as cmd:
            self.assertEqual(m.Policy(temp).jobs(s), jobs)
        cmd.assert_called_once_with(['qstat', '-xf', '-F', 'json', '12.pbs1', '13.pbs1', '14.pbs1'])

    def test_omitted_or_invalid_job_is_unknown_not_finished(self):
        for payload in ({'Jobs': {}}, {'Jobs': {'12.pbs1': None}}, {'Jobs': []}):
            with self.assertRaises(m.SchedulerQueryError):
                m.job_rows(json.dumps(payload), ['12.pbs1'])

    def test_timeout_keeps_owners_reportable_without_releasing_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind, path in m.controls(root).items():
                s = states()[kind]
                s['single']['job_id'] = ('12' if kind == 'domain' else '14') + '.pbs1'
                s['dual']['job_id'] = ('13' if kind == 'domain' else '15') + '.pbs1'
                m.write(path / 'state.json', s)
            policy = m.Policy(root)
            with patch.object(m, 'command', side_effect=subprocess.TimeoutExpired('qstat', 20)), \
                    patch.object(policy, 'capacity') as cap, patch.object(policy, 'set_hold') as hold:
                report = policy.reconcile()
            cap.assert_not_called()
            hold.assert_not_called()
            self.assertFalse(report['jobs_known'])
            self.assertIn('timed out', report['query_error'])
            self.assertTrue(all(not v['allow'] for v in report['choices'].values()))

    def test_diagnose_does_not_install_or_mutate_and_does_not_dump_environment(self):
        calls = []
        def run(args, **kwargs):
            calls.append((args, kwargs['env']))
            return subprocess.CompletedProcess(args, 1, stdout='', stderr='')
        with patch.dict(m.os.environ, {'PRIVATE_TOKEN': 'do-not-print', 'PYTHONPATH': '/site/wrapper'}), \
                patch.object(m, 'status', return_value={'monitoring_active': True}), \
                patch.object(m.subprocess, 'run', side_effect=run), \
                patch.object(m, 'write') as write, patch.object(m, 'stop_controls') as stop, \
                redirect_stdout(io.StringIO()) as out:
            report = m.diagnose(Path('/experiment'))
        self.assertTrue(report['read_only'])
        self.assertEqual([c[0][0] for c in calls], ['qstat', 'qrstat', 'qrstat'])
        self.assertIn('PYTHONPATH', calls[1][1])
        self.assertNotIn('PYTHONPATH', calls[2][1])
        self.assertNotIn('do-not-print', out.getvalue())
        write.assert_not_called()
        stop.assert_not_called()


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.q = dict(enabled='True', started='True', **{'default_chunk.node_group': 'nii'})
        self.r = dict(queue=m.QUEUE, reserve_state='RESV_RUNNING',
                      resv_nodes='(hnode001[0]:ncpus=96+hnode001[1]:ncpus=96)')

    def test_reserved_numa_vnodes_count_one_and_own_reservation_allowed(self):
        result = m.free_hosts({'nodes': one_host()}, self.r, self.q)
        self.assertEqual(result['free_slots'], 1)
        self.assertEqual(result['free_hosts'], ['hnode001'])

    def test_free_state_with_assigned_resources_is_not_free(self):
        result = m.free_hosts({'nodes': one_host(busy=True)}, self.r, self.q)
        self.assertEqual(result['free_slots'], 0)

    def test_unreserved_hosts_and_partial_hosts_are_not_counted(self):
        nodes = {**one_host(), **one_host('hnode002')}
        self.r['resv_nodes'] = '(hnode001[0]:ncpus=96)'
        result = m.free_hosts({'nodes': nodes}, self.r, self.q)
        self.assertEqual(result['free_slots'], 0)
        self.assertNotIn('hnode002', result['hosts'])

    def test_no_static_host_exclusion(self):
        self.r['resv_nodes'] = self.r['resv_nodes'].replace('hnode001', 'hnode552')
        self.assertEqual(m.free_hosts({'nodes': one_host('hnode552')}, self.r, self.q)['free_hosts'], ['hnode552'])

    def test_offline_natural_vnode_prevents_whole_host_use(self):
        nodes = one_host()
        nodes['hnode001']['state'] = 'offline'
        self.assertEqual(m.free_hosts({'nodes': nodes}, self.r, self.q)['free_slots'], 0)

    def test_queue_headroom_and_wrong_group(self):
        self.q['resources_available.ncpus'] = '400'
        self.q['resources_assigned.ncpus'] = '300'
        self.assertEqual(m.free_hosts({'nodes': one_host()}, self.r, self.q)['free_slots'], 0)
        self.q.pop('resources_assigned.ncpus')
        self.q['default_chunk.node_group'] = 'other'
        self.assertEqual(m.free_hosts({'nodes': one_host()}, self.r, self.q)['free_slots'], 0)

    def test_multiline_reservation_and_indirect_cycle(self):
        parsed = m.attributes('Resv ID: R1\n    queue = ' + m.QUEUE + '\n    resv_nodes = (hnode001[0]:ncpus=96+\n\thnode001[1]:ncpus=96)\n    reserve_state = RESV_RUNNING\n')
        self.assertEqual(parsed['resv_nodes'], self.r['resv_nodes'])
        nodes = one_host()
        nodes['hnode001']['resources_available']['host'] = '@hnode001'
        with self.assertRaises(RuntimeError):
            m.free_hosts({'nodes': nodes}, self.r, self.q)


class ReservationLookupTests(unittest.TestCase):
    def detailed(self, rid='R321.pbs2', queue=m.QUEUE):
        return ('Resv ID: ' + rid + '\n    queue = ' + queue + '\n'
                '    reserve_state = RESV_RUNNING\n'
                '    resv_nodes = (hnode001[0]:ncpus=96+hnode001[1]:ncpus=96)\n')

    def test_real_queue_mapping_and_server_are_used(self):
        seen = []
        def command(args, **kwargs):
            seen.append(args)
            if args == ['qrstat']:
                return 'Resv ID  Queue  User State\nR321.pbs2 ' + m.QUEUE + ' owner RN\n'
            if args == ['qrstat', '-f', 'R321.pbs2']:
                return self.detailed()
            self.fail('Unexpected assumed reservation: ' + str(args))
        with patch.object(m, 'command', side_effect=command):
            result = m.reservation_for_queue()
        self.assertEqual(result['reservation_id'], 'R321.pbs2')
        nodes = one_host()
        for row in nodes.values():
            row['resv'] = 'R321.pbs2'
        q = dict(enabled='True', started='True', **{'default_chunk.node_group': 'nii'})
        self.assertEqual(m.free_hosts({'nodes': nodes}, result, q)['free_slots'], 1)

    def test_full_listing_records_are_not_accidentally_merged(self):
        combined = self.detailed('R1.pbs1', 'other') + self.detailed() + self.detailed('R9.pbs1', 'last')
        def command(args, **kwargs):
            return combined if args == ['qrstat', '-f'] else ''
        with patch.object(m, 'command', side_effect=command):
            self.assertEqual(m.reservation_for_queue()['reservation_id'], 'R321.pbs2')

    def test_short_id_wrapper_fallback_requires_explicit_queue_binding(self):
        def command(args, **kwargs):
            if args == ['qrstat', '-f', m.QUEUE]:
                return self.detailed(m.QUEUE)
            raise RuntimeError('site wrapper returned exit status 1')
        with patch.object(m, 'command', side_effect=command):
            self.assertEqual(m.reservation_for_queue()['reservation_id'], m.QUEUE)

    def test_cached_binding_is_revalidated_and_failed_query_never_means_zero_free_nodes(self):
        with patch.object(m, 'command', return_value=self.detailed()) as cmd:
            self.assertEqual(m.reservation_for_queue('R321.pbs2')['queue'], m.QUEUE)
            self.assertEqual(cmd.call_args.args[0], ['qrstat', '-f', 'R321.pbs2'])
            self.assertEqual(cmd.call_count, 1)
        with patch.object(m, 'command', side_effect=RuntimeError('exit status 1; stdout=empty; stderr=empty')):
            with self.assertRaisesRegex(RuntimeError, 'Reservation lookup unavailable.*exit status 1'):
                m.reservation_for_queue()

    def test_conflicting_queue_records_rejected(self):
        text = self.detailed('R1.pbs1') + self.detailed('R2.pbs2')
        with patch.object(m, 'command', return_value=text):
            with self.assertRaisesRegex(RuntimeError, 'Multiple reservation records'):
                m.reservation_for_queue()

    def test_empty_command_failure_retains_return_code(self):
        result = subprocess.CompletedProcess(['qrstat'], 1, stdout='', stderr='')
        with patch.object(m.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, "exit status 1; stdout=''; stderr=''"):
                m.command(['qrstat'])


class DecisionTests(unittest.TestCase):
    def test_zero_two_three_nodes_keep_both_singles(self):
        for count in (0, 1, 2, 3):
            for kind in m.ROLES:
                self.assertFalse(m.decision(kind, states(), {}, capacity(count))[0])

    def test_four_nodes_two_queued_singles_domain_first(self):
        self.assertTrue(m.decision('domain', states(), {}, capacity(4))[0])
        self.assertFalse(m.decision('random', states(), {}, capacity(4))[0])

    def test_two_running_singles_plus_two_free_is_four(self):
        rows = {(k, 'single'): {'job_state': 'R'} for k in m.ROLES}
        self.assertTrue(m.decision('domain', states(), rows, capacity(2, 2))[0])

    def test_second_handoff_needs_transient_fifth_node(self):
        s = states()
        s['domain'].update(phase='DUAL_ACTIVE', handoff_completed_utc='done')
        rows = {('domain', 'dual'): {'job_state': 'R'}, ('random', 'single'): {'job_state': 'R'}}
        self.assertFalse(m.decision('random', s, rows, capacity(1, 3))[0])
        self.assertTrue(m.decision('random', s, rows, capacity(2, 3))[0])

    def test_primary_queue_demand_reserved_before_candidate(self):
        s = states()
        s['domain'].update(phase='DUAL_ACTIVE', handoff_completed_utc='done')
        rows = {('domain', 'dual'): {'job_state': 'R'}}
        self.assertFalse(m.decision('random', s, rows, capacity(2, 2))[0])
        self.assertTrue(m.decision('random', s, rows, capacity(3, 2))[0])

    def test_candidate_running_or_pending_blocks_parallel_candidate(self):
        self.assertFalse(m.decision('domain', states(), {('random', 'dual'): {'job_state': 'R'}}, capacity(8))[0])
        s = states()
        s['random']['pending_job'] = 'unresolved'
        self.assertFalse(m.decision('domain', s, {}, capacity(8))[0])

    def test_r_is_not_domain_qualification_pass(self):
        rows = {('domain', 'dual'): {'job_state': 'R'}}
        self.assertFalse(m.decision('random', states(), rows, capacity(8))[0])
        s = states()
        s['domain']['phase'] = 'CATCH_UP'
        self.assertFalse(m.decision('random', s, rows, capacity(8))[0])


class HoldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy = m.Policy(self.root)
        self.folder = self.root / 'candidate'
        self.folder.mkdir()
        self.job = dict(job_id='123.pbs1', folder=str(self.folder))

    def test_running_candidate_is_never_held(self):
        with patch.object(self.policy, 'observe', return_value={'job_state': 'R'}), patch.object(m, 'command') as cmd:
            self.policy.set_hold('domain', self.job, False, {})
        cmd.assert_not_called()

    def test_preexisting_user_or_system_hold_is_never_released(self):
        for held in ('u', 's', 'us'):
            with patch.object(self.policy, 'observe', return_value={'job_state': 'H', 'Hold_Types': held}), patch.object(m, 'command') as cmd:
                self.policy.set_hold('domain', self.job, True, {})
            cmd.assert_not_called()

    def test_only_journaled_user_hold_is_released(self):
        journal = {'123.pbs1': dict(status='HELD')}
        with patch.object(self.policy, 'observe', return_value={'job_state': 'H', 'Hold_Types': 'u'}), patch.object(m, 'command') as cmd:
            self.policy.set_hold('domain', self.job, True, journal)
        cmd.assert_called_once_with(['qrls', '-h', 'u', '123.pbs1'])
        self.assertEqual(journal['123.pbs1']['status'], 'RELEASED')

    def test_hold_intent_survives_response_loss(self):
        with patch.object(self.policy, 'observe', return_value={'job_state': 'Q', 'Hold_Types': 'n'}), patch.object(m, 'command', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                self.policy.set_hold('domain', self.job, False, {})
        journal = m.read(self.policy.area / 'holds.json')
        self.assertEqual(journal['123.pbs1']['status'], 'INTENT')
        with patch.object(self.policy, 'observe', return_value={'job_state': 'H', 'Hold_Types': 'u'}), patch.object(m, 'command') as cmd:
            self.policy.set_hold('domain', self.job, True, journal)
        cmd.assert_called_once_with(['qrls', '-h', 'u', '123.pbs1'])

    def test_initial_hold_adopted_after_crash_and_rename_hash_matches(self):
        text = '#!/bin/bash\n#PBS -q ' + m.QUEUE + '\n#PBS -N 0390_d4n2_train\n'
        rendered = m.hold_script(text, self.folder, 'domain')
        # Actual domain create_job does this replacement after rendering.
        rendered = rendered.replace('#PBS -N 0390_d4n2_train\n', '#PBS -N 0390_d4dbn2_train\n')
        (self.folder / 'stage.pbs').write_text(rendered)
        journal = {}
        with patch.object(self.policy, 'observe', return_value={'job_state': 'H', 'Hold_Types': 'u'}), patch.object(m, 'command') as cmd:
            self.policy.set_hold('domain', self.job, True, journal)
        cmd.assert_called_once_with(['qrls', '-h', 'u', '123.pbs1'])


class IntegrationTests(unittest.TestCase):
    def test_hook_keeps_single_retry_when_resources_unknown_and_only_holds_initial_dual(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            effects = []
            pbs = types.SimpleNamespace(render_pbs=lambda *args: '#!/bin/bash\n#PBS -q ' + m.QUEUE + '\n#PBS -N 0390_d4n2_train\n')
            class Controller:
                state = {}
                def save(self, **fields):
                    effects.append(('save', fields))
                def tick(self):
                    effects.append('single_retry_and_original_handoff')
                def submit(self, first):
                    effects.append(('submit', first))
                    folder = m.controls(root)['domain'].parent / 'jobs' / ('initial' if first else 'continue')
                    folder.mkdir(parents=True)
                    rendered = pbs.render_pbs(root, folder, 'train', '24:00:00', [])
                    (folder / 'stage.pbs').write_text(rendered)
                    # This is the original create_job hash boundary.
                    return rendered, m.sha(folder / 'stage.pbs')
            module = types.SimpleNamespace(Controller=Controller)
            def imports(name):
                return module if name.endswith('.controller') else pbs
            with patch.object(m.importlib, 'import_module', side_effect=imports), patch.object(m.Policy, 'reconcile', side_effect=RuntimeError('unknown capacity')):
                original = Controller.tick
                m.install_hooks(root, 'domain')
                control = Controller()
                control.tick()
                self.assertIn('single_retry_and_original_handoff', effects)
                initial, _ = control.submit(True)
                subsequent, _ = control.submit(False)
                self.assertIn('#PBS -h\n', initial)
                self.assertNotIn('#PBS -h\n', subsequent)
                self.assertIn('#PBS -c n\n', initial)
                self.assertIsNot(Controller.tick, original)

    def test_extension_adds_to_existing_deadline_or_now_if_expired(self):
        at = dt.datetime(2026, 10, 20, 14, tzinfo=dt.timezone.utc)
        self.assertEqual(m.target_deadline('2026-10-21T00:27:53+00:00', 24, at), '2026-10-22T00:27:53+00:00')
        self.assertEqual(m.target_deadline('2026-10-19T00:00:00+00:00', 24, at), '2026-10-21T14:00:00+00:00')

    def test_repeat_operation_reuses_target_and_attempts_both_owners(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = m.area(root) / 'operations/once.json'
            targets = {k: '2026-10-22T00:27:53+00:00' for k in m.ROLES}
            m.write(record, dict(version=m.VERSION, extra_hours=24, phase='INSTALLED', targets=targets))
            with redirect_stdout(io.StringIO()), patch.object(m.Policy, 'reconcile', side_effect=RuntimeError('query unavailable')), patch.object(m, 'start_owner') as start:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, 'Incomplete refresh'):
                        m.apply(root, 24, 'once', Path('/python'))
                self.assertEqual(start.call_count, 4)
            self.assertEqual(m.read(record)['targets'], targets)

    def test_wrong_owner_queue_or_script_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = m.controls(root)['domain'].parent / 'jobs/example'
            job = dict(folder=str(folder))
            row = dict(Job_Owner=m.getpass.getuser() + '@login1', queue=m.QUEUE,
                       Job_Name='0390_d4dbn2_train', Submit_arguments=str(folder / 'stage.pbs'),
                       Resource_List={'nodect': 2})
            m.validate_job(job, row, root, 'domain', 'dual')
            for key, value in [('queue', 'rt_HF'), ('Job_Owner', 'someone_else@login1'), ('Submit_arguments', '/wrong/stage.pbs')]:
                changed = dict(row, **{key: value})
                with self.assertRaises(RuntimeError):
                    m.validate_job(job, changed, root, 'domain', 'dual')

    def test_install_freezes_operational_adapter_without_rewriting_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            originals = {}
            for kind, path in m.controls(root).items():
                contract = {'type': kind}
                identity = m.digest(contract)
                m.write(path / 'plan.json', {'contract': contract, 'identity': identity})
                m.write(path / 'state.json', {'identity': identity, 'phase': 'WAIT_READY',
                    'deadline_utc': '2026-10-21T00:27:53+00:00', 'single': None, 'dual': None})
                original_policy = path / 'runtime_limits_30h_24h/limits_policy.py'
                m.atomic_text(original_policy, '# existing limits policy\n')
                cfg = dict(version=m.LIMITS_VERSION, kind=kind, root=str(root), contract_identity=identity,
                           policy_sha256=m.sha(original_policy), guarded_packages={})
                m.write(original_policy.parent / 'config.json', cfg)
                m.atomic_text(path / 'entry.py', m.limits_entry(cfg))
                originals[kind] = (path / 'plan.json').read_bytes()
            with redirect_stdout(io.StringIO()), patch.object(m, 'start_owner') as start, patch.object(m.Policy, 'reconcile', return_value={}):
                m.apply(root, 24, 'install', Path('/python'))
                targets = m.read(m.area(root) / 'operations/install.json')['targets']
                m.apply(root, 24, 'install', Path('/python'))
                self.assertEqual(start.call_count, 4)
            self.assertEqual(m.read(m.area(root) / 'operations/install.json')['targets'], targets)
            for kind, path in m.controls(root).items():
                self.assertEqual((path / 'plan.json').read_bytes(), originals[kind])
                m.validate_control(root, kind, path)
                state = m.read(path / 'state.json')
                state['pending_job'] = 'recover-me'
                m.write(path / 'state.json', state)
                # Restart must still enter the original pending-qsub recovery path.
                m.validate_control(root, kind, path, allow_pending=True)
                with self.assertRaisesRegex(RuntimeError, 'Submission pending'):
                    m.validate_control(root, kind, path)

    def test_reconcile_holds_random_before_releasing_domain_and_fails_closed_on_unknown_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            s = states()
            for kind, path in m.controls(root).items():
                m.write(path / 'state.json', s[kind])
            policy = m.Policy(root)
            with patch.object(policy, 'jobs', return_value={}), patch.object(policy, 'capacity', return_value=capacity(4)), patch.object(policy, 'set_hold') as hold:
                report = policy.reconcile()
                self.assertEqual([(c.args[0], c.args[2]) for c in hold.call_args_list], [('random', False), ('domain', True)])
                self.assertTrue(report['choices']['domain']['allow'])
            with patch.object(policy, 'jobs', return_value={}), patch.object(policy, 'capacity', side_effect=RuntimeError('no reservation response')), patch.object(policy, 'set_hold') as hold:
                report = policy.reconcile()
                self.assertTrue(all(not c.args[2] for c in hold.call_args_list))
                self.assertEqual(report['query_error'], 'no reservation response')

    def test_read_only_status_does_not_compete_for_held_policy_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind, path in m.controls(root).items():
                m.write(path / 'state.json', dict(states()[kind], controller_status='RUNNING'))
            policy = m.Policy(root)
            with m.locked(m.area(root) / 'policy.lock'):
                with patch.object(policy, 'jobs', return_value={}), patch.object(policy, 'capacity', return_value=capacity(4)), patch.object(policy, 'set_hold') as hold:
                    report = policy.reconcile(mutate=False)
                self.assertTrue(report['read_only'])
                hold.assert_not_called()
                with self.assertRaises(BlockingIOError):
                    policy.reconcile(mutate=True)

    def test_status_still_reports_owners_when_scheduler_query_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind, path in m.controls(root).items():
                m.write(path / 'state.json', dict(states()[kind], controller_status='RUNNING'))
            with redirect_stdout(io.StringIO()), patch.object(m, 'active', return_value=True), patch.object(m.Policy, 'reconcile', side_effect=RuntimeError('PBS unavailable')):
                report = m.status(root)
            self.assertTrue(report['monitoring_active'])
            self.assertFalse(report['capacity_ready'])
            self.assertEqual(report['capacity_now']['query_error'], 'PBS unavailable')

    def test_upgrade_preserves_extended_deadline_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            targets = {k: '2026-10-22T00:27:53+00:00' for k in m.ROLES}
            old_hash = None
            for kind, path in m.controls(root).items():
                contract = {'type': kind}
                identity = m.digest(contract)
                m.write(path / 'plan.json', {'contract': contract, 'identity': identity})
                m.write(path / 'state.json', {'identity': identity, 'phase': 'WAIT_READY',
                    'deadline_utc': targets[kind], 'single': None, 'dual': None})
                limits = path / 'runtime_limits_30h_24h/limits_policy.py'
                m.atomic_text(limits, '# limits\n')
                m.write(limits.parent / 'config.json', dict(version=m.LIMITS_VERSION, kind=kind, root=str(root),
                    contract_identity=identity, policy_sha256=m.sha(limits), guarded_packages={}))
                old = path / 'runtime_capacity/policy.py'
                m.atomic_text(old, '# previously approved policy fixture\n')
                old_hash = m.sha(old)
                m.atomic_text(path / 'entry.py', m.entry_text(old, old_hash))
            record = m.area(root) / 'operations/existing.json'
            m.write(record, dict(version=m.PREVIOUS_VERSION, phase='RUNNING', extra_hours=24, targets=targets))
            with redirect_stdout(io.StringIO()), patch.object(m, 'PREVIOUS_SHA256', old_hash), patch.object(m, 'start_owner') as start, patch.object(m, 'stop_controls', wraps=m.stop_controls) as stop, patch.object(m.Policy, 'reconcile', return_value={'capacity': None, 'query_error': 'lookup failed'}):
                m.apply(root, 24, 'existing', Path('/python'), upgrade=True)
                m.apply(root, 24, 'existing', Path('/python'), upgrade=True)
                self.assertEqual(stop.call_count, 1)
                self.assertEqual(start.call_count, 4)
            op = m.read(record)
            self.assertEqual(op['targets'], targets)
            self.assertEqual(op['deadline_extension_in_this_upgrade'], 0)
            self.assertTrue(op['monitors_verified'])
            self.assertFalse(op['capacity_ready'])
            self.assertEqual(op['phase'], 'RUNNING_CAPACITY_WAIT')
            self.assertIn('lookup failed', op['warnings']['capacity'])
            self.assertFalse(op['errors'])
            for path in m.controls(root).values():
                self.assertEqual(m.sha(path / 'runtime_capacity/policy.py'), m.sha(m.__file__))

    def test_upgrade_never_creates_another_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'No saved extension operation'):
                m.apply(Path(temp), 24, 'missing', Path('/python'), upgrade=True)


if __name__ == '__main__':
    unittest.main()
