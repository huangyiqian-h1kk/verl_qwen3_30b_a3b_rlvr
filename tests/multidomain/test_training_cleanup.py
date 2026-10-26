import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/multidomain/cleanup_finished_training.py'
spec = importlib.util.spec_from_file_location('cleanup_finished_training', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'root'
        self.base = self.root / 'outputs/multidomain'
        self.base.mkdir(parents=True)
        self.review = self.root / 'outputs/analysis/review'
        self.review.mkdir(parents=True)
        self.keep = {k: {} for k in m.f.KINDS}
        self.retire = {k: {} for k in m.f.KINDS}
        self.archives = {}
        for kind in m.f.KINDS:
            self.owner(kind)
            for step, mode in ((100, 'weights'), (150, 'full'), (195, 'retire'), (200, 'full')):
                self.archive(kind, step, mode)
            self.archive(kind, 50, 'unverified_weights')
            self.index(kind)
        # An unretained historical recovery pin shares the retained final inode.
        final = self.archives['random', 200]['payload']
        self.pin = m.f.paths(self.root, 'random')['run'] / 'jobs/old/train/source/global_step_200'
        for file in Path(final).rglob('*'):
            if file.is_file():
                dest = self.pin / file.relative_to(final)
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(file, dest)
        self.old = self.file(m.f.paths(self.root, 'domain')['single'] / 'checkpoints/global_step_5/actor/model/old.distcp', 'z'*16384)
        clone = self.old.parents[2] / 'actor/optimizer/old.distcp'
        clone.parent.mkdir(parents=True, exist_ok=True)
        os.link(self.old, clone)
        self.file(self.old.parents[2] / 'COMPLETE_0390.json', '{}')
        self.log = self.file(m.f.paths(self.root, 'random')['run'] / 'jobs/old/pbs.log', 'valuable results')
        self.outside = self.file(self.root / 'models/base.safetensors', 'do not delete')
        self.addCleanup(patch.stopall)
        patch.object(m, 'KEEP', self.keep).start()
        patch.object(m, 'RETIRE', self.retire).start()
        patch.object(m.f, 'pbs_jobs', return_value={}).start()
        with contextlib.redirect_stdout(io.StringIO()):
            m.f.inventory(self.root, self.review)
        self.json(self.review / 'stop.json', dict(status='STOPPED', root=str(self.root), owners_fenced=True, remaining_jobs={}))
        patch.object(m, 'REPORT_HASHES', {name:m.sha(self.review/name) for name in m.REPORT_HASHES}).start()

    def file(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def json(self, path, obj):
        return self.file(path, json.dumps(obj))

    def owner(self, kind):
        p = m.f.paths(self.root, kind)
        control = p['control']
        self.json(control/'retired.json', dict(version=m.f.VERSION))
        self.json(control/'state.json', dict(operation='owner', phase='STOPPED_BY_USER', controller_status='STOPPED',
                  authority=dict(lineage=kind, run=str(p['run']/'lineages'/kind))))
        for d in (p['handoff'],p['monitor'],self.base/'checkpoint_archive_service'):
            self.file(d/'stop.request','')
        template=control/'operations/owner/template'
        text='from pathlib import Path\n'
        for function in ('spawn','daemon'):
            text += (f'def {function}(root, kind):\n'
                     '    # Persistent user-requested retirement; no automatic restart.\n'
                     f'    if Path({str(control / "retired.json")!r}).exists():\n        return\n')
        path=self.file(template/'runtime/d4_training_continuation/cli.py',text)
        self.json(control/'operations/owner/plan.json',dict(template=str(template), retirement_guard=m.f.VERSION,
                  template_files={'runtime/d4_training_continuation/cli.py':m.sha(path)}))

    def archive(self,kind,step,final_mode):
        mode = 'unverified_weights' if final_mode == 'unverified_weights' else 'full'
        cp = m.f.paths(self.root,kind)['run'] / 'fixture' / ('global_step_'+str(step))
        contents = {k:dict(path=p,format='megatron_dist_checkpoint') for k,p in
                    [('model','model'),('optimizer','optimizer/dist_ckpt'),('lr_scheduler','extra/lr_scheduler'),('rng_state','extra/rng_state')]}
        self.json(cp/'actor/ckpt_contents.json', dict(schema_version=2,global_step=step,world_size=8,contents=contents))
        self.file(cp/'actor/transformer_config.json','{}')
        for name in ('actor/model/a.distcp','actor/model/.metadata'):
            self.file(cp/name,kind*2048)
        if mode=='full':
            for name in ('actor/optimizer/dist_ckpt/o.distcp','actor/extra/lr_scheduler/l.distcp','actor/extra/rng_state/r.pt','data.pt'):
                self.file(cp/name,'a'*8192)
        record=dict(path=str(cp),step=step,world_size=8,lineage=kind if mode=='full' else None,files=m.ck.inventory(cp))
        key=m.digest({k:record[k] for k in ('lineage','step','world_size','files')})
        directory=m.f.paths(self.root,kind)['run']/'checkpoint_archive/objects'/key
        payload=directory/'payload'/cp.name
        payload.parent.mkdir(parents=True)
        cp.rename(payload)
        record['path']=str(payload)
        inventory=dict(record['files'])
        if mode=='full':
            self.json(payload/m.ck.MARKER,record)
            s=(payload/m.ck.MARKER).stat()
            inventory[m.ck.MARKER]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=m.sha(payload/m.ck.MARKER))
        model=m.model_files(record)
        obj=dict(kind=kind,key=key,step=step,mode=mode,lineage=record['lineage'],payload=str(payload),original=record,
                 inventory=inventory,model_inventory=model,trusted_inventory=mode=='full',
                 holds=['historical-rescue-step-'+str(step)] if final_mode in ('weights','unverified_weights') else [])
        self.json(directory/'manifest.json',obj)
        self.archives[kind,step]=obj
        if final_mode=='retire':self.retire[kind][key]=step
        else:self.keep[kind][key]=(step,final_mode)

    def index(self,kind):
        cohort={}
        for metric,step in (('overall',150),('overall_full',150),('math',100),('if',200)):
            o=self.archives[kind,step]
            cohort[metric]=dict(key=o['key'],step=step,path=o['payload'],mode=o['mode'],score=.5)
        self.json(m.f.paths(self.root,kind)['run']/'checkpoint_archive/index.json',dict(cohorts={'p':cohort},full_keep=[],weights_keep=[]))

    def plan(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return m.make_plan(self.root,self.review,'test')

    def apply(self,plan):
        with contextlib.redirect_stdout(io.StringIO()):
            return m.apply(self.root,'test',plan['sha256'])

    def test_plan_only_and_exact_full_weight_retention_with_idempotent_apply(self):
        plan=self.plan()
        self.assertTrue(self.old.exists())
        self.assertTrue((Path(self.archives['random',100]['payload'])/'data.pt').exists())
        before=m.fingerprint(Path(self.archives['random',200]['payload'])/'actor/model/a.distcp')
        result=self.apply(plan)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertFalse(self.old.exists())
        self.assertEqual(self.log.read_text(),'valuable results')
        self.assertEqual(self.outside.read_text(),'do not delete')
        self.assertFalse((self.pin/'actor/model/a.distcp').exists())
        self.assertTrue((self.pin/(m.ck.MARKER+'.before_cleanup')).exists())
        for (kind,step),obj in self.archives.items():
            cp=Path(obj['payload']); directory=cp.parent.parent
            if step in (150,200):
                m.ck.validate(dict(obj['original'],path=str(cp)))
            elif step==195:
                self.assertFalse((directory/'manifest.json').exists())
                self.assertTrue((directory/'retired.json').exists())
            elif step==100:
                self.assertFalse((cp/'data.pt').exists())
                self.assertFalse((cp/m.ck.MARKER).exists())
                self.assertEqual(m.read(directory/'manifest.json')['mode'],'weights')
                self.assertEqual(m.read(directory/'manifest.json')['holds'],[])
            self.assertTrue((cp/'actor/ckpt_contents.json').exists())
        after=m.fingerprint(Path(self.archives['random',200]['payload'])/'actor/model/a.distcp')
        self.assertEqual(before['inode'],after['inode'])
        self.assertEqual(after['nlink'],before['nlink']-1)
        self.assertEqual(self.apply(plan)['status'],'COMPLETE')

    def test_interrupt_after_partial_unlinks_resumes_same_plan(self):
        plan=self.plan(); actual=m.unlink_one; calls=[]
        def interrupted(*args):
            if len(calls)==4:raise RuntimeError('simulated interruption')
            calls.append(args[1]);return actual(*args)
        with patch.object(m,'unlink_one',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):self.apply(plan)
        self.assertEqual(m.read(m.operation_dir(self.root,'test')/'state.json')['status'],'INCOMPLETE')
        self.assertEqual(self.apply(plan)['status'],'COMPLETE')

    def test_interrupt_during_metadata_never_deletes_payload(self):
        plan=self.plan(); actual=m.apply_metadata
        def partial(root,edits):
            actual(root,edits[:3]);raise RuntimeError('metadata interrupted')
        with patch.object(m,'apply_metadata',side_effect=partial):
            with self.assertRaisesRegex(RuntimeError,'metadata interrupted'):self.apply(plan)
        self.assertTrue(all((self.root/p).exists() for p in plan['files']))
        self.assertEqual(self.apply(plan)['status'],'COMPLETE')

    def test_changed_protected_archive_blocks_all_unlinks(self):
        plan=self.plan()
        cp=Path(self.archives['random',150]['payload'])
        (cp/'data.pt').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'inventory changed'):self.apply(plan)
        self.assertTrue(self.old.exists())

    def test_replaced_target_is_never_deleted(self):
        plan=self.plan()
        relative=next(iter(plan['files']));p=self.root/relative
        p.unlink();p.write_text('replacement')
        with self.assertRaisesRegex(RuntimeError,'changed since plan'):self.apply(plan)
        self.assertEqual(p.read_text(),'replacement')

    def test_hardlink_added_after_preview_blocks_deletion(self):
        plan=self.plan();p=self.root/next(iter(plan['files']))
        os.link(p,self.root/'unexpected-link')
        with self.assertRaisesRegex(RuntimeError,'Hardlink count changed'):self.apply(plan)
        self.assertTrue(self.old.exists())

    def test_symlink_replacement_does_not_touch_external_target(self):
        plan=self.plan();p=self.root/next(iter(plan['files']))
        p.unlink();p.symlink_to(self.outside)
        with self.assertRaisesRegex(RuntimeError,'Symlink'):self.apply(plan)
        self.assertEqual(self.outside.read_text(),'do not delete')

    def test_wrong_confirmation_and_modified_plan_are_rejected(self):
        plan=self.plan()
        with self.assertRaisesRegex(RuntimeError,'confirm-plan'):
            m.apply(self.root,'test','wrong')
        p=m.operation_dir(self.root,'test')/'plan.json'
        value=m.read(p);value['files']={};m.write(p,value)
        with self.assertRaisesRegex(RuntimeError,'SHA256 changed'):self.apply(plan)
        self.assertTrue(self.old.exists())

    def test_unknown_pbs_and_active_job_refuse_cleanup(self):
        plan=self.plan()
        with patch.object(m.f,'pbs_jobs',side_effect=RuntimeError('PBS unavailable')):
            with self.assertRaisesRegex(RuntimeError,'PBS unavailable'):self.apply(plan)
        with patch.object(m.f,'owned_jobs',return_value={'123.pbs1':{}}):
            with self.assertRaisesRegex(RuntimeError,'PBS jobs still exist'):self.apply(plan)
        self.assertTrue(self.old.exists())

    def test_busy_stage_lease_refuses_before_mutation(self):
        plan=self.plan()
        lock=m.f.paths(self.root,'random')['run']/'lineages/random/stage.lock'
        with m.f.lease(lock):
            with self.assertRaises(BlockingIOError):self.apply(plan)
        self.assertTrue(self.old.exists())

    def test_removed_retirement_or_changed_metadata_refuses_cleanup(self):
        plan=self.plan()
        p=m.f.paths(self.root,'random')['control']/'state.json'
        s=m.read(p);s['phase']='ACTIVE';m.write(p,s)
        with self.assertRaisesRegex(RuntimeError,'not persistently retired'):self.apply(plan)
        s['phase']='STOPPED_BY_USER';m.write(p,s)
        p=m.f.paths(self.root,'random')['run']/'checkpoint_archive/index.json'
        s=m.read(p);s['changed']=True;m.write(p,s)
        with self.assertRaisesRegex(RuntimeError,'Metadata changed'):self.apply(plan)
        self.assertTrue(self.old.exists())

    def test_new_checkpoint_after_report_blocks_planning(self):
        self.file(m.f.paths(self.root,'random')['run']/'lineages/random/checkpoints/global_step_205/data.pt','new')
        with self.assertRaisesRegex(RuntimeError,'paths changed'):self.plan()
        self.assertTrue(self.old.exists())

    def test_new_checkpoint_after_preview_blocks_apply(self):
        plan=self.plan()
        self.file(m.f.paths(self.root,'random')['run']/'lineages/random/checkpoints/global_step_205/data.pt','new')
        with self.assertRaisesRegex(RuntimeError,'paths changed since preview'):self.apply(plan)
        self.assertTrue(self.old.exists())

    def test_changed_review_report_is_rejected(self):
        p=self.review/'storage.json';v=m.read(p);v['changed']=True;m.write(p,v)
        with self.assertRaisesRegex(RuntimeError,'reviewed Oct 27 report'):self.plan()

    def test_external_links_before_plan_are_rejected(self):
        os.link(self.old,self.root/'external-link')
        with self.assertRaisesRegex(RuntimeError,'external hardlinks'):self.plan()


if __name__=='__main__':unittest.main()
