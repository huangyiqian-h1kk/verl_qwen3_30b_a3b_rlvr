"""Validation interruption/replay, zero-update backfill and deployment faults."""
from collections import defaultdict
import copy
import importlib.util
import json
import getpass
import os
from pathlib import Path
import shutil
import sys
import types
import uuid
from unittest.mock import patch
import numpy as np

from test_training_continuation import Harness, checkpoint, FUTURE, PAST
import test_training_continuation as fixtures
import test_training_continuation_save_policy as save_tests
from test_training_continuation_save_policy import validation_rows, metric_values
from d4_training_continuation import backfill, cli, evaluation as ev, runtime, source, validation_journal as journal
from d4_training_continuation import validation_upgrade as upgrade, checkpoints as ck
from d4_training_continuation.common import digest, files, paths, read, write
from d4_training_continuation.scheduler import Unknown
from d4_training_continuation.trainer import trainer_class

FIXTURE = Path(__file__).parent / "fixtures/validation/pinned_validate.py"


class Attr(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Tensor:
    def __init__(self, values): self.values = np.array(values)
    def sum(self, axis): return Tensor(self.values.sum(axis))
    def cpu(self): return self
    def tolist(self): return self.values.tolist()
    @property
    def shape(self): return self.values.shape


class Batch(dict):
    def __len__(self): return len(self['prompts'])


class Proto:
    def __init__(self, data):
        self.batch = Batch(prompts=data['prompts'])
        self.non_tensor_batch = {k:v for k,v in data.items() if k != 'prompts'}
        self.meta_info = {}
    @classmethod
    def from_single_dict(cls, data): return cls(copy.deepcopy(data))
    def repeat(self, repeat_times, interleave):
        assert repeat_times == 1
        return self
    def __iter__(self):
        for value in self.non_tensor_batch['reward_model']:
            yield types.SimpleNamespace(non_tensor_batch={'reward_model':value})
    def union(self, other):
        self.batch.update(other.batch); self.non_tensor_batch.update(other.non_tensor_batch)
        return self


class JournalTests(Harness):
    def validator(self, fail=None):
        spec = importlib.util.spec_from_file_location('pinned_validation_test', FIXTURE)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.__dict__.update(np=np, uuid=uuid, defaultdict=defaultdict, DataProto=Proto,
            pad_dataproto_to_divisor=lambda b,n:(b,0), unpad_dataproto=lambda b,pad_size:b,
            extract_reward=lambda b:(Tensor([[1.] for _ in range(len(b.batch))]),
                                     {'acc':[1.]*len(b.batch)}))
        obj = module.PinnedValidator()
        obj.config = Attr(actor_rollout_ref=Attr(rollout=Attr(val_kwargs=Attr(n=1,do_sample=False),agent=Attr(num_workers=2))),
                          trainer=Attr(validation_data_dir=None))
        obj.global_steps=125; obj.use_rm=False
        obj.tokenizer=types.SimpleNamespace(eos_token_id=1,pad_token_id=0,decode=lambda ids,**_:str(ids[0]))
        obj.val_dataloader=[]
        for start in range(0,1006,8):
            ids=list(range(start,min(start+8,1006)))
            obj.val_dataloader.append(dict(prompts=np.array([[x] for x in ids]),
                reward_model=np.array([{'ground_truth':str(x)} for x in ids],dtype=object),
                data_source=np.array([ev.DOMAINS[x%4] for x in ids],dtype=object)))
        obj._get_gen_batch=lambda batch:batch
        obj._maybe_log_val_generations=lambda **kw:setattr(obj,'logged',kw)
        def metrics(domains,uids,extra,turns):
            obj.aggregation=(list(domains),uids,dict(extra),turns)
            return {f'val-core/{d}/acc/mean@1':np.mean([v for s,v in zip(domains,extra['acc']) if s==d]) for d in ev.DOMAINS}
        obj._val_metrics_update=metrics
        obj.generated=[]
        def generate(batch):
            first=int(batch.batch['prompts'][0][0]);obj.generated.append(first)
            if first == fail: raise RuntimeError('simulated walltime')
            batch.batch['responses']=batch.batch['prompts']
            return batch
        obj.async_rollout_manager=types.SimpleNamespace(generate_sequences=generate)
        return obj

    def attach(self,obj, directory=None, protocol='same'):
        obj.cont_journal=journal.Journal(directory or self.root/'journal',dict(checkpoint='cp125',protocol=protocol),
                                        dict(batches=126))
        return journal.instrument(obj._validate)

    def test_pinned_validator_interrupt_reuses_completed_batch_and_aggregates_all_1006_rows(self):
        first=self.validator(fail=8);call=self.attach(first)
        with patch('builtins.print'), self.assertRaisesRegex(RuntimeError,'walltime'):call()
        self.assertEqual(read(self.root/'journal/progress.json')['completed_rows'],8)
        self.assertFalse((self.root/'journal/batch_000002.json').exists())
        resumed=self.validator();call=self.attach(resumed)
        with patch('builtins.print'):metrics=call()
        proof=resumed.cont_journal.finish()
        self.assertEqual(proof['reused_batches'],1)
        self.assertEqual(metrics,{k:1. for k in ev.CORE})
        self.assertEqual(resumed.generated[0],8)
        self.assertEqual(len(resumed.logged['inputs']),1006)
        self.assertEqual(resumed.logged['inputs'],[str(x) for x in range(1006)])
        self.assertEqual(len(set(resumed.aggregation[1])),1006)
        again=self.validator();call=self.attach(again)
        with patch('builtins.print'):self.assertEqual(call(),metrics)
        self.assertEqual(again.cont_journal.finish()['reused_batches'],126)
        self.assertFalse(again.generated)

    def test_changed_inputs_or_batch_checksum_cannot_be_reused(self):
        obj=self.validator(fail=8)
        with patch('builtins.print'),self.assertRaises(RuntimeError):self.attach(obj)()
        changed=self.validator();changed.val_dataloader[0]['prompts'][0][0]=10000
        with self.assertRaisesRegex(RuntimeError,'inputs reordered'):self.attach(changed)()
        file=self.root/'journal/batch_000001.json';data=read(file)
        data['payload']['lists']['sample_scores'][0]=0.;write(file,data)
        with self.assertRaisesRegex(RuntimeError,'batch changed'):self.attach(self.validator())()

    def test_changed_checkpoint_protocol_or_runtime_rejects_existing_journal(self):
        self.attach(self.validator())
        with self.assertRaisesRegex(RuntimeError,'binding changed'):
            self.attach(self.validator(),protocol='other-scorer')
        with self.assertRaisesRegex(RuntimeError,'binding changed'):
            journal.Journal(self.root/'journal',dict(checkpoint='cp150',protocol='same'),dict(batches=126))

    def test_changed_upstream_validator_is_rejected_before_generation(self):
        journal.check_source(FIXTURE)
        modified=self.root/'changed.py';modified.write_text(FIXTURE.read_text().replace('sample_scores = []','sample_scores = [0]'))
        with self.assertRaisesRegex(RuntimeError,'differs from pinned'):journal.check_source(modified)

    def test_crash_after_batch_commit_before_progress_write_keeps_completed_batch(self):
        obj=self.validator();call=self.attach(obj);original=journal.write
        def interrupted(path,value):
            if Path(path).name=='progress.json':raise OSError('crash after batch fsync')
            return original(path,value)
        with patch.object(journal,'write',side_effect=interrupted),patch('builtins.print'), \
             self.assertRaisesRegex(OSError,'fsync'):call()
        self.assertTrue((self.root/'journal/batch_000001.json').exists())
        resumed=self.validator()
        with patch('builtins.print'):self.attach(resumed)()
        self.assertEqual(resumed.cont_journal.finish()['reused_batches'],1)
        self.assertEqual(resumed.generated[0],8)

    def test_receipt_is_published_only_after_all_batches_and_complete_output(self):
        cp=checkpoint(self.root/'checkpoint/global_step_125',125,16,'random')
        phase=self.root/'job/train';phase.mkdir(parents=True)
        output=self.root/'generations/125.jsonl'
        req=dict(kind='random',job_dir=str(phase.parent),run_dir=str(self.root/'run'),control_dir=str(self.root/'control'),
                 lineage='random',validation_policy=journal.POLICY,validation_semantics={'kind':'random'})
        def trainer(fail=None):
            obj=self.validator(fail)
            obj.cont_req=req;obj.cont_resume=cp;obj.cont_phase=phase;obj._dump_futures=[]
            obj.config.trainer.validation_data_dir=str(output.parent)
            def dump(inputs,outputs,gts,scores,reward_extra_infos_dict,dump_path):
                output.parent.mkdir(exist_ok=True)
                output.write_text(''.join(json.dumps(dict(input=i,output=o,gts=g,score=s,step=125))+'\n'
                                         for i,o,g,s in zip(inputs,outputs,gts,scores)))
            obj._dump_generations=dump
            return obj
        first=trainer(8)
        with patch.object(journal,'runtime_contract',return_value={'batches':126}),patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError,'walltime'):ev.validate(first,first._validate)
            self.assertIsNone(ev.completed(req,cp))
            resumed=trainer();ev.validate(resumed,resumed._validate)
        proof=ev.completed(req,cp)
        self.assertEqual(proof['outputs']['rows'],1006)
        self.assertEqual(proof['origin']['journal']['reused_batches'],1)
        self.assertEqual(resumed.generated[0],8)
        again=trainer()
        with patch.object(journal,'runtime_contract',return_value={'batches':126}):
            ev.validate(again,lambda **_:self.fail('Completed validation was repeated'))
        self.assertTrue(read(phase/'validation_125.json')['reused'])
        with self.assertRaisesRegex(RuntimeError,'runtime/dataset changed'):
            ev.completed(req,cp,runtime={'batches':127})


class BackfillHarness(Harness):
    def plan(self,kind):
        plan=fixtures.IntegrationTests.template_plan(self,kind)
        old=Path(plan['old_request']['reward']['runtime_path'])
        folder=Path(plan['template']);(folder/'synthetic_reward').rename(folder/'md_reward_test')
        plan['old_request']['reward']['runtime_path']=str(old.parent.parent/'md_reward_test/__init__.py')
        plan['validation_policy']=dict(journal.POLICY)
        plan['old_request']['config']['verl_src']=str(self.root/'verl')
        data=self.root/'data';data.mkdir(exist_ok=True)
        (data/'validation.parquet').write_bytes(b'synthetic bound validation data')
        dest=self.root/'verl/verl/trainer/ppo/ray_trainer.py';dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(FIXTURE,dest)
        target=paths(self.root,kind)['control']/'operations/synthetic/template'
        target.parent.mkdir(parents=True,exist_ok=True);shutil.move(folder,target)
        plan.update(template=str(target),template_files=files(target))
        return plan

    def historical(self):
        sources={}
        for kind,t in backfill.TARGETS.items():
            run=paths(self.root,kind)['run'];folder=run/'jobs'/t['job'];folder.mkdir(parents=True)
            (folder/'job_id.txt').write_text(t['job_id'])
            cp=checkpoint(run/'original'/f'global_step_{t["step"]}',t['step'],t['world'],kind,kind)
            pin=ck.pin(cp,run/'control/resume_sources/historical')
            ck.pin(cp,folder/'train/source')
            contract=dict(source={'record':pin},kind=kind)
            write(folder/'request.json',dict(contract=contract,identity=digest(contract),source={'record':pin}))
            sources[kind]=pin
        return sources

    def campaign(self):
        self.historical();protected=backfill.protect(self.root)
        plans={k:self.plan(k) for k in ('random','domain')}
        area=backfill.prepare(self.root,'repair-test',plans,protected,FUTURE)
        return area,plans,protected


class BackfillTests(BackfillHarness):
    def test_protection_is_independent_of_both_temporary_source_cleanups(self):
        original=self.historical();protected=backfill.protect(self.root)
        for kind,t in backfill.TARGETS.items():
            run=paths(self.root,kind)['run']
            ck.remove_pin(Path(original[kind]['path']).parent)
            ck.remove_pin(run/'jobs'/t['job']/'train/source')
            shutil.rmtree(run/'original')
            ck.validate(protected[kind])
        self.assertEqual(backfill.protect(self.root),protected)

    def test_one_direct_queue_job_fresh_children_no_training_authority_changes(self):
        area,plans,protected=self.campaign()
        first=backfill.tick(area,self.pbs);self.assertEqual(len(self.pbs.submissions),1)
        job=first['job'];folder=Path(job['folder'])
        r=backfill.validate_request(folder/'request.json')
        self.assertIn('#PBS -q R9920261000\n',(folder/'stage.pbs').read_text())
        self.assertIn('#PBS -l select=1\n',(folder/'stage.pbs').read_text())
        self.assertEqual(self.pbs.capacity_calls,0)
        self.assertFalse(self.pbs.cancellations)
        for kind,child in r['children'].items():
            req=source.validate_request(Path(child['folder'])/'request.json')
            self.assertEqual(req['role'],'evaluation')
            self.assertEqual(req['run_dir'],str(Path(child['folder'])/'evaluation'))
            self.assertEqual(req['source']['record'],protected[kind])
            self.assertEqual(req['config']['training']['save_freq'],5)
        self.assertEqual(backfill.tick(area,self.pbs)['job'],job)
        self.assertEqual(len(self.pbs.submissions),1)

    def test_both_children_use_fresh_processes_and_completed_setup_is_skipped_on_retry(self):
        area,_,protected=self.campaign();state=backfill.tick(area,self.pbs)
        folder=Path(state['job']['folder']);r=read(folder/'request.json')
        calls=[]
        def run(args,**kw):
            child=Path(args[2]).parent;req=read(child/'request.json');cp=req['source']['record']
            calls.append(req['kind'])
            output=Path(req['run_dir'])/'generations/train'/f'{cp["step"]}.jsonl'
            validation_rows(output,cp['step']);ev.publish(req,cp,metric_values(),output,{'mode':'test'})
            write(child/'train/result.json',dict(status='PASS',evaluated_step=cp['step'],optimizer_updates=0))
        with patch.object(runtime,'wait_gpus'),patch.object(backfill.subprocess,'run',side_effect=run), \
             patch.dict(backfill.os.environ,{'PBS_JOBID':state['job']['job_id']}):
            backfill.execute(folder/'request.json')
        self.assertEqual(calls,['random','domain'])
        self.assertEqual(read(folder/'result.json')['status'],'PASS')
        self.pbs.rows[state['job']['job_id']].update(job_state='F',Exit_status=0)
        self.assertEqual(backfill.tick(area,self.pbs)['phase'],'PASS')
        with patch.object(runtime,'wait_gpus'),patch.object(backfill.subprocess,'run') as run_again, \
             patch.dict(backfill.os.environ,{'PBS_JOBID':state['job']['job_id']}):
            backfill.execute(folder/'request.json')
            run_again.assert_not_called()

    def test_retry_keeps_protocol_across_diagnostic_path_changes_and_rejects_app_failure(self):
        area,_,_=self.campaign();first=backfill.tick(area,self.pbs)
        a=read(Path(first['job']['folder'])/'request.json')
        self.pbs.rows[first['job']['job_id']].update(job_state='F',Exit_status=271)
        second=backfill.tick(area,self.pbs);b=read(Path(second['job']['folder'])/'request.json')
        self.assertEqual(len(self.pbs.submissions),2)
        for kind in ('random','domain'):
            x=read(Path(a['children'][kind]['folder'])/'request.json')
            y=read(Path(b['children'][kind]['folder'])/'request.json')
            self.assertNotEqual(x['job_dir'],y['job_dir'])
            self.assertEqual(ev.protocol(x),ev.protocol(y))
        self.pbs.rows[second['job']['job_id']].update(job_state='F',Exit_status=1)
        self.assertEqual(backfill.tick(area,self.pbs)['phase'],'BLOCKED')
        backfill.tick(area,self.pbs);self.assertEqual(len(self.pbs.submissions),2)

    def test_unknown_qsub_or_exit_never_creates_another_job(self):
        area,_,_=self.campaign();self.pbs.ambiguous=True
        for _ in range(2):
            with self.assertRaises(Unknown):backfill.tick(area,self.pbs)
        self.assertEqual(len(self.pbs.submissions),1)

    def test_unknown_exit_waits_and_only_explicit_walltime_minus29_retries(self):
        area,_,_=self.campaign();first=backfill.tick(area,self.pbs)
        row=self.pbs.rows[first['job']['job_id']];row.update(job_state='F',Exit_status=None)
        with self.assertRaises(Unknown):backfill.tick(area,self.pbs)
        self.assertEqual(len(self.pbs.submissions),1)
        row.update(Exit_status=-29,comment='Job exceeded resource walltime')
        second=backfill.tick(area,self.pbs);self.assertEqual(len(self.pbs.submissions),2)
        self.pbs.rows[second['job']['job_id']].update(job_state='F',Exit_status=-29,comment='unknown')
        self.assertEqual(backfill.tick(area,self.pbs)['phase'],'BLOCKED')

    def test_eval_role_cannot_update_actor_or_save_checkpoint(self):
        obj=trainer_class(object)();obj.cont_req={'role':'evaluation'}
        with self.assertRaisesRegex(RuntimeError,'updates are forbidden'):obj._update_actor(None)
        with self.assertRaisesRegex(RuntimeError,'writes are forbidden'):obj._save_checkpoint()

    def test_eval_only_and_final_468_resume_stop_after_validation(self):
        for role,step,val_only in [('formal',150,False),('evaluation',125,True),('formal',468,True)]:
            folder=self.root/f'{role}_{step}';folder.mkdir()
            write(folder/'allocation.json',{'execution_id':'bound'})
            req=dict(job_dir=str(folder),role=role,identity='bound',expected_nodes=1,kind='random',
                     root=str(self.root),data_dir='/data',run_dir='/run',reward={'import_path':'pkg://same'},
                     config={'training':{'train_nodes':1}},training_policy={'save_freq':5})
            cfg=types.SimpleNamespace(trainer=types.SimpleNamespace(save_freq=5,test_freq=25),
                                      reward=types.SimpleNamespace(custom_reward_function=types.SimpleNamespace()))
            omega=types.SimpleNamespace(OmegaConf=types.SimpleNamespace(update=lambda *a,**k:None,save=lambda *a,**k:None))
            with patch.object(runtime,'validate_request',return_value=req), \
                 patch.object(runtime,'select_source',return_value=dict(step=step,world_size=8,path='/cp')), \
                 patch.dict(os.environ),patch.dict(sys.modules,{'omegaconf':omega,
                 'multidomain.runtime_config':types.SimpleNamespace(compose=lambda *a:cfg)}), \
                 patch('d4_training_continuation.optimizer_repair.configure'),patch.object(runtime,'run_trainer'):
                runtime.phase(folder/'request.json','train')
            self.assertEqual(cfg.trainer.val_only,val_only)
            self.assertTrue(cfg.trainer.val_before_train)


class DeploymentTests(BackfillHarness):
    def fixture(self):
        self.historical();controllers={}
        for kind in ('random','domain'):
            c=self.controller(kind);plan=self.plan(kind)
            plan.pop('validation_policy')
            plan['old_request']['config']['training']['save_freq']=25 if kind=='random' else 5
            write(c.control/'operations/synthetic/plan.json',plan)
            run=c.p['run']/'lineages/current';cp=checkpoint(run/'checkpoints/global_step_150',150,8,'current',kind)
            active=c.state['active'];folder=Path(active['folder'])
            config=copy.deepcopy(plan['old_request']['config'])
            req=dict(lineage='current',config=config,job_dir=str(folder),run_dir=str(run),kind=kind,
                     control_dir=str(c.control),data_dir=str(self.root/'data'),
                     contract=dict(config=config,snapshot_files={},root_files={}))
            req['identity']=digest(req['contract']);write(folder/'request.json',req)
            c.save(phase='EXPIRED',controller_status='STOPPED',host='some-other-login',migration_complete=True,
                deadline_utc=PAST,promoted=kind=='random',handoff_failed=kind=='domain',monitoring_finished=True,
                authority=dict(run=str(run),lineage='current'),old_candidate_retired=True)
            controllers[kind]=c
        return controllers

    def test_expired_owners_any_login_extended_once_and_actual_plans_upgraded(self):
        controllers=self.fixture();spawned=[]
        def spawn(root,kind):
            spawned.append(kind)
            # Protected sources must precede any restart/cleanup.
            for k in ('random','domain'):
                ck.validate(read(self.root/'outputs/multidomain/validation_backfill_sources'/f'{k}_step_{backfill.TARGETS[k]["step"]}'/'source.json'))
        original_tick=backfill.tick
        def tick(area):return original_tick(area,self.pbs)
        bound={'verified':True,'first_check_completed':True}
        with patch.object(cli,'spawn',side_effect=spawn),patch.object(cli,'runtime_status',return_value=bound), \
             patch.object(backfill,'tick',side_effect=tick):
            a=upgrade.deploy(self.root,'same-op',24)
            self.assertEqual(a['status'],'RUNNING')
            b=upgrade.deploy(self.root,'same-op',24)
        self.assertEqual(a['targets'],b['targets'])
        self.assertEqual(spawned,['domain','random'])
        self.assertEqual(len(self.pbs.submissions),1)
        for k,c in controllers.items():
            state=read(c.control/'state.json');plan=read(c.control/'operations/synthetic/plan.json')
            self.assertEqual(plan['validation_policy'],journal.POLICY)
            self.assertEqual(plan['training_policy']['save_freq'],5)
            self.assertEqual(plan['opportunistic']['min_free_slots'],2)
            self.assertEqual(state['deadline_utc'],a['targets'][k])
            self.assertFalse(state['monitoring_finished'])
        self.assertEqual(read(controllers['random'].control/'state.json')['checkpoint_restart']['minimum_step'],150)

    def test_restart_target_advances_past_already_completed_training_updates(self):
        c=self.fixture()['random'];state=read(c.control/'state.json')
        write(Path(state['active']['folder'])/'train/step_151.json',dict(status='PASS',actor_update_completed=True,step=151))
        self.assertEqual(upgrade.restart_target(state,'random'),175)

    def test_partial_plan_publish_retries_same_deadlines_and_backfill_intent(self):
        controllers=self.fixture();original_write=upgrade.write;failed=[False]
        def interrupted(path,value):
            if Path(path)==controllers['domain'].control/'state.json' and not failed[0]:
                failed[0]=True;raise OSError('publish interrupted')
            original_write(path,value)
        original_tick=backfill.tick
        with patch.object(cli,'spawn'),patch.object(cli,'runtime_status',return_value=dict(verified=True,first_check_completed=True)), \
             patch.object(backfill,'tick',side_effect=lambda area:original_tick(area,self.pbs)):
            with patch.object(upgrade,'write',side_effect=interrupted),self.assertRaisesRegex(OSError,'interrupted'):
                upgrade.deploy(self.root,'interrupted-op',24)
            ledger=self.root/'outputs/multidomain/continuation_operations/interrupted-op.validation_repair.json'
            targets=read(ledger)['targets']
            result=upgrade.deploy(self.root,'interrupted-op',24)
        self.assertEqual(result['targets'],targets)
        self.assertEqual(result['status'],'RUNNING')
        self.assertEqual(len(self.pbs.submissions),1)

    def test_random_controlled_restart_waits_for_current_validation_not_old_125(self):
        c,run,_=save_tests.RestartTests.fixture(self,150)
        c.plan['validation_policy']=dict(journal.POLICY)
        c.state['checkpoint_restart']['minimum_step']=150
        c.tick()
        self.assertFalse(self.pbs.cancellations)
        self.assertEqual(c.state['checkpoint_restart']['wait_reason'],'scheduled_validation_in_progress')
        with patch.object(ev,'import_legacy',return_value=True):c.tick()
        self.assertEqual(len(self.pbs.cancellations),1)

    def test_real_isolated_owners_load_new_modules_and_submit_one_backfill(self):
        import test_training_continuation_launch as launch
        controllers=self.fixture();self.children=[]
        self.addCleanup(lambda:launch.LaunchTests.stop_children(self))
        rows={}
        for c in controllers.values():
            job=c.state['active'];c.p['old_control'].mkdir(parents=True,exist_ok=True)
            write(c.p['monitor']/'state.json',dict(status='STOPPED'))
            rows[job['job_id']]=dict(Job_Owner=getpass.getuser()+'@login',queue='R9920261000',
                Job_Name=job['job_name'],Submit_arguments=str(Path(job['folder'])/'stage.pbs'),
                job_state='R',Resource_List={'nodect':1})
        write(self.root/'pbs_rows.json',rows)
        binary=self.root/'bin';binary.mkdir()
        queue=('Queue: R9920261000\n    queue_type = Execution\n    enabled = True\n    started = True\n'
               '    default_chunk.node_group = nii\n    acl_group_enable = True\n    acl_groups = gcg51557\n'
               '    resources_available.ncpus = 1920\n    resources_assigned.ncpus = 1920\n')
        body=('import json,os,re,sys\nfrom pathlib import Path\n'
              "root=Path(os.environ['D4_TEST_PBS']);name=Path(sys.argv[0]).name\n"
              "with (root/'calls.jsonl').open('a') as f:f.write(json.dumps([name]+sys.argv[1:])+'\\n')\n"
              "rows=json.loads((root/'pbs_rows.json').read_text())\n"
              "if name=='qstat':\n"
              f"    print({queue!r} if '-Qf' in sys.argv else json.dumps({{'Jobs':rows}}))\n"
              "elif name=='pbsnodes':print(json.dumps({'nodes':{}}))\n"
              "elif name=='qsub':\n"
              "    path=Path(sys.argv[1]);jobname=re.search(r'^#PBS -N (.+)$',path.read_text(),re.M)[1]\n"
              "    jid='900001.pbs1'\n"
              f"    rows[jid]=dict(Job_Owner={getpass.getuser()+'@login'!r},queue='R9920261000',Job_Name=jobname,\n"
              "                   Submit_arguments=str(path),job_state='Q',Resource_List={'nodect':1})\n"
              "    (root/'pbs_rows.json').write_text(json.dumps(rows));print(jid)\n"
              "else:raise RuntimeError('Unexpected mutation: '+name)\n")
        for name in ('qstat','qsub','qdel','pbsnodes'):
            p=binary/name;p.write_text('#!'+sys.executable+'\n'+body);p.chmod(0o700)
        real_spawn=cli.spawn
        def spawn(root,kind):
            child=real_spawn(root,kind)
            if child:self.children.append(child)
            return child
        with patch.dict(os.environ,{'PATH':str(binary)+os.pathsep+os.environ['PATH'],'D4_TEST_PBS':str(self.root)}), \
             patch.object(cli,'spawn',side_effect=spawn):
            result=upgrade.deploy(self.root,'real-processes',24)
            self.assertEqual(result['status'],'RUNNING',repr(result))
            for k,c in controllers.items():
                s=read(c.control/'state.json')
                self.assertTrue(cli.runtime_status(self.root,k)['verified'])
                self.assertIsNone(s.get('last_error'),repr(s))
                for mod in ('validation_journal','evaluation','backfill'):
                    self.assertIn('d4_training_continuation.'+mod,s['runtime_binding']['modules'])
            again=upgrade.deploy(self.root,'real-processes',24)
            self.assertEqual(again['targets'],result['targets'])
        calls=[json.loads(x) for x in (self.root/'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len([x for x in calls if x[0]=='qsub']),1)
        self.assertFalse([x for x in calls if x[0]=='qdel'])

