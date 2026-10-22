"""Regression and failure tests for effective save policy and validation replay."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import types
from unittest.mock import patch

from test_training_continuation import Harness, checkpoint, FUTURE, PAST
import test_training_continuation as fixtures
from d4_training_continuation import checkpoints as ck, evaluation as ev, checkpoint_restart as restart
from d4_training_continuation import cli, runtime, source, save_policy_upgrade as upgrade
from d4_training_continuation.common import digest, files, paths, read, sha, write
from d4_training_continuation.save_policy import POLICY, training_config
from d4_training_continuation.scheduler import Unknown
from d4_training_continuation.trainer import trainer_class


def metric_values():
    return {key: 0.5 for key in ev.CORE}


def validation_rows(path, step, count=ev.ROWS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as out:
        for _ in range(count):
            out.write(json.dumps(dict(input='prompt', output='answer', score=0.5, step=step)) + '\n')


class SavePolicyTests(Harness):
    def test_composed_runtime_replays_only_due_source_and_rejects_wrong_effective_frequency(self):
        for step, save, expected in [(125,5,True),(120,5,False),(125,25,None)]:
            with self.subTest(step=step,save=save):
                folder=self.root/f'phase_{step}_{save}';folder.mkdir()
                write(folder/'allocation.json',{'execution_id':'allocation'})
                r=dict(job_dir=str(folder),identity='frozen-identity',role='formal',expected_nodes=2,training_policy=POLICY,
                       config={'training':{'save_freq':5,'test_freq':25}},data_dir='/data',run_dir='/run',
                       kind='random',root=str(self.root),reward={'import_path':'pkg://same'})
                config=types.SimpleNamespace(trainer=types.SimpleNamespace(save_freq=save,test_freq=25),
                    reward=types.SimpleNamespace(custom_reward_function=types.SimpleNamespace()))
                omega=types.SimpleNamespace(OmegaConf=types.SimpleNamespace(update=lambda *a,**k:None,
                    save=lambda *a,**k:None))
                compose=types.SimpleNamespace(compose=lambda *a:config)
                with patch.object(runtime,'validate_request',return_value=r), \
                     patch.object(runtime,'select_source',return_value={'step':step,'world_size':16,'path':'/pin'}), \
                     patch.dict(os.environ), \
                     patch.dict(sys.modules,{'omegaconf':omega,'multidomain.runtime_config':compose}), \
                     patch('d4_training_continuation.optimizer_repair.configure'), \
                     patch.object(runtime,'run_trainer') as run:
                    if expected is None:
                        with self.assertRaisesRegex(RuntimeError,'lost the save5'):runtime.phase(folder/'request.json','train')
                        run.assert_not_called()
                    else:
                        runtime.phase(folder/'request.json','train')
                        self.assertEqual(config.trainer.val_before_train,expected)
                        self.assertEqual(config.trainer.save_freq,5)
                        self.assertEqual(config.trainer.test_freq,25)
                        run.assert_called_once()

    def test_stale_request_25_becomes_effective_5_in_both_topologies_without_other_changes(self):
        plan = fixtures.IntegrationTests.template_plan(self, 'random')
        plan['old_request']['config']['training']['save_freq'] = 25
        before = copy.deepcopy(plan)
        for nodes in (1, 2):
            made = source.make_job(plan, 'formal', nodes, {'lineage': 'same', 'record': {}}, FUTURE)
            req = source.validate_request(Path(made['folder']) / 'request.json')
            expected = source.topology_config(before['old_request']['config'], nodes)
            expected['training']['save_freq'] = 5
            self.assertEqual(req['config'], expected)
            self.assertEqual(req['training_policy'], POLICY)
            self.assertEqual(req['contract']['training_policy'], POLICY)
        self.assertEqual(plan, before)

    def test_unknown_policy_is_not_silently_overwritten(self):
        for save, test in [(10,25),(5,10),(None,25)]:
            with self.assertRaisesRegex(RuntimeError, 'Unexpected checkpoint'):
                training_config({'training':{'save_freq':save,'test_freq':test}})

    def test_removing_bound_policy_does_not_disable_validation_replay(self):
        plan = fixtures.IntegrationTests.template_plan(self, 'random')
        made = source.make_job(plan, 'formal', 2, {'lineage':'same','record':{}}, FUTURE)
        path = Path(made['folder']) / 'request.json'
        request = read(path); request.pop('training_policy'); write(path,request)
        with self.assertRaisesRegex(RuntimeError, 'policy contract changed'):
            source.validate_request(path)


class EvaluationTests(Harness):
    def fixture(self, step=125):
        run = self.root / 'run'; folder = self.root / 'job'; phase = folder / 'train'
        phase.mkdir(parents=True)
        record = checkpoint(run/'checkpoints'/f'global_step_{step}',step,16,'lineage')
        cfg = dict(training=dict(save_freq=5,test_freq=25,val_n=1,val_temperature=0),seed=42)
        req = dict(config=cfg,kind='random',lineage='lineage',run_dir=str(run),job_dir=str(folder),
                   control_dir=str(self.root/'control'),data_dir=str(self.root/'data'),training_policy=POLICY,
                   contract=dict(config=cfg,snapshot_files={'md_reward_x/__init__.py':'same'}))
        req['identity']=digest(req['contract']); write(folder/'request.json',req)
        output=run/'generations/train'/f'{step}.jsonl'
        return req,record,phase,output

    def trainer(self, req, cp, phase, output):
        return types.SimpleNamespace(cont_req=req,global_steps=cp['step'],cont_resume=cp,
            cont_phase=phase,_dump_futures=[],config=types.SimpleNamespace(trainer=types.SimpleNamespace(
            validation_data_dir=str(output.parent))))

    def test_interrupted_evaluation_has_no_receipt_and_retries_same_weights(self):
        r, cp, phase, output = self.fixture(); t=self.trainer(r,cp,phase,output)
        validation_rows(output,125,20)
        def interrupted(**_):
            validation_rows(output,125,50)
            raise RuntimeError('271 during validation')
        with self.assertRaisesRegex(RuntimeError,'271'):
            ev.validate(t,interrupted)
        self.assertFalse(ev.receipt_path(r,cp).exists())
        def complete(**_):
            validation_rows(output,125)
            return metric_values()
        result=ev.validate(t,complete)
        self.assertEqual(result,metric_values())
        self.assertEqual(t.global_steps,125)
        self.assertEqual(read(phase/'validation_125.json')['reused'],False)
        self.assertEqual(len(list((output.parent/'replay_archive').glob('*.jsonl'))),2)

    def test_complete_evaluation_reused_across_hardlink_pins_and_new_allocation(self):
        r,cp,phase,output=self.fixture(); validation_rows(output,125)
        ev.publish(r,cp,metric_values(),output,{'mode':'test'})
        pinned=ck.pin(cp,self.root/'pin')
        new=copy.deepcopy(r);new['config']['training']['train_nodes']=1
        t=self.trainer(new,pinned,phase,output)
        with patch('builtins.print'):
            result=ev.validate(t,lambda **_: self.fail('Repeated expensive evaluation'))
        self.assertEqual(result,metric_values())
        self.assertTrue(read(phase/'validation_125.json')['reused'])

    def test_async_output_failure_cannot_publish_pass(self):
        r,cp,phase,output=self.fixture();t=self.trainer(r,cp,phase,output)
        class Future:
            def result(self): raise OSError('disk full')
        t._dump_futures=[Future()]
        with self.assertRaisesRegex(OSError,'disk full'):
            ev.validate(t,lambda **_: metric_values())
        self.assertFalse(ev.receipt_path(r,cp).exists())

    def test_partial_rows_and_nonfinite_metrics_cannot_publish_pass(self):
        r,cp,phase,output=self.fixture();validation_rows(output,125,20)
        with self.assertRaisesRegex(RuntimeError,'output count'):
            ev.publish(r,cp,metric_values(),output,{})
        validation_rows(output,125)
        bad=metric_values();bad[ev.CORE[0]]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'non-finite'):
            ev.publish(r,cp,bad,output,{})
        self.assertFalse(ev.receipt_path(r,cp).exists())

    def test_receipt_does_not_match_different_checkpoint_or_reward(self):
        r,cp,phase,output=self.fixture();validation_rows(output,125)
        ev.publish(r,cp,metric_values(),output,{})
        foreign=dict(cp,lineage='other')
        self.assertIsNone(ev.completed(r,foreign))
        new=copy.deepcopy(r);new['contract']['snapshot_files']['md_reward_x/__init__.py']='different'
        self.assertIsNone(ev.completed(new,cp))
        output.write_text('changed')
        with self.assertRaises((ValueError,RuntimeError)):
            ev.completed(r,cp)

    def test_legacy_import_requires_final_metrics_and_complete_outputs(self):
        r,cp,phase,output=self.fixture();folder=Path(r['job_dir'])
        job=dict(job_id='123.pbs1',folder=str(folder))
        validation_rows(output,125)
        (folder/'pbs.log').write_text('validation generation end\nstep:125 - actor/loss:0.1\n')
        self.assertFalse(ev.import_legacy(job,cp))
        text='step:125 - '+' - '.join(k+':np.float64(0.5)' for k in ev.CORE)+' - timing_s/testing:17700\n'
        (folder/'pbs.log').write_text(text)
        self.assertTrue(ev.import_legacy(job,cp))
        self.assertEqual(ev.completed(r,cp)['origin']['mode'],'legacy-complete')

    def test_wrapper_uses_validation_guard_and_does_not_advance_sampler(self):
        r,cp,phase,output=self.fixture()
        class Base:
            def _validate(base,merged=False):
                self.assertTrue(base.cont_validating)
                self.assertEqual(base.global_steps,125)
                validation_rows(output,125)
                return metric_values()
        t=trainer_class(Base)()
        t.__dict__.update(self.trainer(r,cp,phase,output).__dict__)
        t.cont_validating=False;t.training_batch_ids=['next-batch-126']
        self.assertEqual(t._validate(),metric_values())
        self.assertFalse(t.cont_validating)
        self.assertEqual(t.training_batch_ids,['next-batch-126'])
        self.assertEqual(t.global_steps,125)


class RestartTests(Harness):
    def fixture(self, step=100):
        c=self.controller('random')
        c.plan['training_policy']=dict(POLICY)
        write(c.control/'operations/synthetic/plan.json',c.plan)
        run=c.p['run']/'lineages/accepted'
        cp=checkpoint(run/'checkpoints'/f'global_step_{step}',step,16,'accepted')
        job=c.state['active'];folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
        cfg={'training':{'save_freq':25,'test_freq':25}}
        req=dict(config=cfg,lineage='accepted',run_dir=str(run),job_dir=str(folder),kind='random',
                 control_dir=str(c.control),data_dir=str(self.root/'data'),contract=dict(config=cfg,snapshot_files={}))
        req['identity']=digest(req['contract']);write(folder/'request.json',req)
        c.save(phase='ACTIVE',promoted=True,migration_complete=True,authority=dict(run=str(run),lineage='accepted'))
        c.save(checkpoint_restart=restart.arm(c.state,125))
        return c,run,cp

    def advance(self,run,step=125):
        return checkpoint(run/'checkpoints'/f'global_step_{step}',step,16,'accepted')

    def test_waits_for_complete_checkpoint_then_pins_cancels_and_submits_once(self):
        c,run,cp=self.fixture();self.pbs.free=2
        c.tick();self.assertFalse(self.pbs.cancellations)
        incomplete=run/'checkpoints/global_step_125';incomplete.mkdir()
        c.tick();self.assertFalse(self.pbs.cancellations)
        incomplete.rmdir();cp=self.advance(run)
        c.tick();self.assertEqual(len(self.pbs.cancellations),1)
        ck.validate(c.state['checkpoint_restart']['protected'])
        c=self.reload(c);c.tick()
        self.assertEqual(c.state['checkpoint_restart']['phase'],'COMPLETE')
        self.assertEqual(len(self.pbs.submissions),1)
        self.assertEqual(self.pbs.submissions[0]['source']['record']['step'],125)
        self.assertEqual(self.pbs.submissions[0]['expected_nodes'],2)
        for _ in range(3):c=self.reload(c);c.tick()
        self.assertEqual(len(self.pbs.submissions),1)

    def test_unknown_scheduler_never_cancels_or_submits(self):
        c,run,_=self.fixture(125);self.pbs.unknown.add(c.state['active']['job_id'])
        with self.assertRaises(Unknown):c.tick()
        self.assertFalse(self.pbs.cancellations or self.pbs.submissions)

    def test_corrupt_complete_checkpoint_never_cancels(self):
        c,run,cp=self.fixture(125);(Path(cp['path'])/'data.pt').unlink()
        with self.assertRaises(RuntimeError):c.tick()
        self.assertFalse(self.pbs.cancellations)

    def test_natural_271_keeps_normal_retry_and_does_not_force_125(self):
        c,run,_=self.fixture();job=c.state['active']
        self.pbs.rows[job['job_id']]={'job_state':'F','Exit_status':271}
        c.tick()
        self.assertEqual(c.state['checkpoint_restart']['phase'],'NATURAL_EXIT')
        self.assertEqual(self.pbs.submissions[0]['source']['record']['step'],100)
        self.assertFalse(self.pbs.cancellations)

    def test_application_failure_before_planned_qdel_is_not_retried(self):
        c,run,_=self.fixture(125)
        self.pbs.rows[c.state['active']['job_id']]={'job_state':'F','Exit_status':1}
        c.tick()
        self.assertFalse(self.pbs.submissions or self.pbs.cancellations)
        self.assertIn('Application exit 1',c.state['training_blocked'])

    def test_deadline_before_cancel_keeps_current_training(self):
        c,run,_=self.fixture(125);c.save(deadline_utc=PAST)
        c.tick();self.assertEqual(c.state['checkpoint_restart']['phase'],'EXPIRED')
        self.assertFalse(self.pbs.cancellations or self.pbs.submissions)

    def test_ambiguous_submission_never_duplicates(self):
        c,run,_=self.fixture(125);c.tick();self.pbs.ambiguous=True
        with self.assertRaises(Unknown):c.tick()
        c=self.reload(c)
        with self.assertRaises(Unknown):c.tick()
        self.assertEqual(len(self.pbs.submissions),1)

    def test_crash_after_active_publication_adopts_replacement(self):
        c,run,_=self.fixture(125);c.tick();original=c.save
        def crash(**fields):
            original(**fields)
            if 'active' in fields:
                raise RuntimeError('crash after active publication')
        with patch.object(c,'save',side_effect=crash):
            with self.assertRaisesRegex(RuntimeError,'publication'):c.tick()
        c=self.reload(c);c.tick()
        self.assertEqual(c.state['checkpoint_restart']['phase'],'COMPLETE')
        c.tick();self.assertFalse(c.state['pending'])
        self.assertEqual(len(self.pbs.submissions),1)

    def test_final_checkpoint_after_stop_can_be_newer_than_initial_pin(self):
        c,run,_=self.fixture(125);c.tick();self.advance(run,130);c.tick()
        self.assertEqual(self.pbs.submissions[0]['source']['record']['step'],130)

    def test_no_free_node_preserves_single_node_fallback_policy(self):
        c,run,_=self.fixture(125);self.pbs.free=0;c.tick();c.tick()
        self.assertEqual(self.pbs.submissions[0]['expected_nodes'],1)

    def test_recovery_pins_retained_until_replacement_full_checkpoint_advances(self):
        c,run,_=self.fixture(125);c.tick();c.tick()
        pin=Path(c.state['checkpoint_restart']['protected']['path'])
        c.tick();self.assertTrue(pin.exists())
        self.advance(run,126);c.tick()
        self.assertFalse(pin.exists())
        self.assertTrue(c.state['checkpoint_restart']['pins_retired'])

    def test_retrying_repair_does_not_arm_a_job_already_using_save5(self):
        c,run,_=self.fixture()
        path=Path(c.state['active']['folder'])/'request.json'
        r=read(path);r['config']['training']['save_freq']=5;write(path,r)
        c.state.pop('checkpoint_restart')
        request=restart.arm(c.state,125)
        self.assertEqual(request['phase'],'SUPERSEDED')
        self.assertEqual(request['reason'],'active_job_already_save5')

    def test_lost_qdel_exit_code_and_expiry_after_intent_still_finish_one_replacement(self):
        c,run,_=self.fixture(125);c.tick()
        self.pbs.rows[c.state['active']['job_id']]={'job_state':'F','Exit_status':None}
        c.save(deadline_utc=PAST);c.tick()
        self.assertEqual(c.state['checkpoint_restart']['phase'],'COMPLETE')
        self.assertEqual(len(self.pbs.submissions),1)


class UpgradeTests(Harness):
    def fixture(self):
        c=self.controller('random');plan=fixtures.IntegrationTests.template_plan(self,'random')
        target=c.control/'operations/synthetic/template';shutil.move(plan['template'],target)
        plan.update(template=str(target),template_files=files(target));plan['old_request']['config']['training']['save_freq']=25
        path=c.control/'operations/synthetic/plan.json';write(path,plan)
        c.save(phase='ACTIVE',promoted=True,migration_complete=True)
        return c,path,plan

    def test_frozen_repair_preserves_old_request_deadlines_and_opportunity_settings(self):
        c,path,plan=self.fixture();before=files(plan['template']);state=copy.deepcopy(c.state)
        _,old,new=upgrade.prepare(self.root)
        self.assertEqual(new['old_request'],old['old_request'])
        self.assertEqual(new['training_policy'],POLICY)
        self.assertEqual(files(plan['template']),before)
        self.assertEqual(read(c.control/'state.json'),state)
        write(path,new)
        self.assertEqual(upgrade.prepare(self.root)[1:],(new,new))

    def test_unsupported_runtime_and_inflight_submission_block_before_mutation(self):
        c,path,plan=self.fixture();c.save(pending={'formal':{'folder':'pending'}})
        with self.assertRaisesRegex(RuntimeError,'in-flight'):upgrade.prepare(self.root)
        c.save(pending={})
        p=Path(plan['template'])/'runtime/d4_training_continuation/source.py';p.write_text('unreviewed')
        plan['template_files']=files(plan['template']);write(path,plan)
        with self.assertRaisesRegex(RuntimeError,'Unsupported frozen'):upgrade.prepare(self.root)
        self.assertEqual(read(path),plan)

    def test_domain_is_not_restarted_or_modified(self):
        d=self.controller('domain');before=read(d.control/'state.json')
        with self.assertRaisesRegex(RuntimeError,'scoped to random'):
            upgrade.prepare(self.root,'domain')
        self.assertEqual(read(d.control/'state.json'),before)


class UpgradeProcessTests(Harness):
    def test_actual_frozen_owner_repair_arms_wait_without_touching_jobs_or_domain(self):
        import subprocess
        import test_training_continuation_launch as launch
        from d4_training_continuation import cancellation_repair, opportunity_upgrade
        self.children=[]
        self.addCleanup(lambda:launch.LaunchTests.stop_children(self))
        c,planfile,original,entry=launch.LaunchTests.fixture(self)
        run=c.p['run']/'lineages/accepted'
        checkpoint(run/'checkpoints/global_step_100',100,16,'accepted')
        c.save(phase='ACTIVE',promoted=True,old_candidate_retired=True,
               authority=dict(run=str(run),lineage='accepted'))
        write(Path(c.state['active']['folder'])/'request.json',
              dict(lineage='accepted',config={'training':{'save_freq':25,'test_freq':25}}))
        d=self.controller('domain');domain_before=(d.control/'state.json').read_bytes()
        before=copy.deepcopy(c.state);original_files=files(original)
        with (c.control/'controller.log').open('a') as out:
            child=subprocess.Popen([sys.executable,'-I',str(entry),'_run','--root',str(self.root),
                                    '--kind','random'],stdout=out,stderr=subprocess.STDOUT)
        self.children.append(child)
        launch.LaunchTests.wait_for(self,lambda:(c.control/'legacy_identity.json').exists(),c.control)
        identity=read(c.control/'legacy_identity.json');real_spawn=cli.spawn
        def spawn(root,kind):
            process=real_spawn(root,kind)
            if process is not None:self.children.append(process)
            return process
        legacy=copy.deepcopy(opportunity_upgrade.LEGACY)
        legacy['cli.py'].add(sha(original/'runtime/d4_training_continuation/cli.py'))
        with patch.object(opportunity_upgrade,'LEGACY',legacy), \
             patch.object(cancellation_repair,'_process_identity',return_value=(identity['uid'],identity['argv'])), \
             patch.object(cli,'spawn',side_effect=spawn):
            result=upgrade.repair(self.root,'random',125)
            self.assertEqual(result['status'],'SAVE_POLICY_MONITOR_RUNNING',
                             repr(result)+'\n'+(c.control/'controller.log').read_text())
            self.assertEqual(child.wait(timeout=3),0)
            state=read(c.control/'state.json')
            self.assertEqual(state['checkpoint_restart']['phase'],'WAIT_CHECKPOINT')
            self.assertEqual(state['checkpoint_restart']['observed_checkpoint_step'],100)
            self.assertEqual(state['active'],before['active'])
            self.assertEqual(state['deadline_utc'],before['deadline_utc'])
            modules=state['runtime_binding']['modules']
            for name in ['controller','source','save_policy','checkpoint_restart']:
                self.assertIn('d4_training_continuation.'+name,modules)
            again=upgrade.repair(self.root,'random',125)
            self.assertFalse(again['controller_restart_requested'])
        self.assertEqual(files(original),original_files)
        self.assertEqual((d.control/'state.json').read_bytes(),domain_before)
        calls=[json.loads(s) for s in (self.root/'pbs_calls.jsonl').read_text().splitlines()]
        self.assertFalse(any(x[0] in ('qsub','qdel') for x in calls))
