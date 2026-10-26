"""Real filesystem + failure injection; never submit jobs or load model tensors."""
import copy
import errno
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

from d4_training_continuation import checkpoints as ck, evaluation as ev
from d4_training_continuation import best_checkpoints as best, best_archiver as bridge
from d4_training_continuation import best_upgrade as upgrade
from d4_training_continuation.common import digest, files, now, paths, read, sha, write


class BestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kind, self.lineage = "random", "formal-lineage"
        self.archive = best.Archive(self.root, self.kind)
        self.p = paths(self.root, self.kind)
        write(self.p["control"] / "state.json", dict(authority={"lineage": self.lineage}, operation="test"))

    def cp(self, step, lineage=None, kind=None):
        kind, lineage = kind or self.kind, lineage or self.lineage
        root = paths(self.root, kind)["run"] / "lineages" / lineage / "checkpoints" / f"global_step_{step}"
        spec = dict(schema_version=2, global_step=step, world_size=8, contents={})
        for section in ("model", "optimizer", "lr_scheduler", "rng_state"):
            folder = root / "actor" / section / "dist_ckpt"
            folder.mkdir(parents=True)
            (folder / ".metadata").write_bytes(b"not unpickled by archive")
            (folder / "__0_0.distcp").write_bytes((section + str(step)).encode())
            (folder / "__1_0.distcp").write_bytes((section + str(step) + "-rank1").encode())
            spec["contents"][section] = dict(path=section + "/dist_ckpt", format="megatron_dist_checkpoint")
        write(root / "actor/ckpt_contents.json", spec)
        write(root / "actor/transformer_config.json", {"architecture": "synthetic"})
        (root / "data.pt").write_bytes(b"loader")
        if kind == "domain":
            write(root / "sampling.json", {"plan": "immutable"})
        record = dict(path=str(root), step=step, world_size=8, lineage=lineage, files=ck.inventory(root))
        write(root / ck.MARKER, {k: v for k, v in record.items() if k != "path"})
        return record

    def request(self, record, protocol="stable", role="formal", kind=None):
        kind = kind or self.kind
        p = paths(self.root, kind)
        folder = p["run"] / "jobs" / (role + "_" + str(record["step"]) + "_" + protocol)
        contract = dict(kind=kind, lineage=record["lineage"], role=role, source={"record": record},
                        validation_semantics={"scoring": protocol})
        request = dict(contract, contract=contract, identity=digest(contract), config={"training": {}},
                       root=str(self.root), job_dir=str(folder), control_dir=str(p["control"]))
        write(folder / "request.json", request)
        return request

    def proof(self, record, scores, protocol="stable", role="formal"):
        request = self.request(record, protocol, role)
        output = Path(request["job_dir"]) / "val.jsonl"
        output.write_text("".join('{"step":%d,"input":"x","output":"y","score":1}\n' % record["step"] for _ in range(ev.ROWS)))
        metrics = dict(zip(ev.CORE, scores))
        ev.publish(request, record, metrics, output, {"job_dir": request["job_dir"]})
        return request

    def test_hardlinks_survive_rolling_deletion_and_metadata_references_archive(self):
        record = self.cp(125)
        obj = self.archive.put(record)
        cp = Path(obj["payload"])
        member = "actor/model/dist_ckpt/__0_0.distcp"
        self.assertEqual((cp/member).stat().st_ino, (Path(record["path"])/member).stat().st_ino)
        shutil.rmtree(record["path"])
        self.archive.verify(obj)
        self.assertEqual(self.archive.put(record)["key"], obj["key"])
        ck.validate(dict(record, path=str(cp)))

    def test_missing_optimizer_or_loader_rescues_all_model_shards_only(self):
        record = self.cp(125)
        shutil.rmtree(Path(record["path"]) / "actor/optimizer")
        (Path(record["path"]) / "data.pt").unlink()
        obj = self.archive.put(record)
        self.assertEqual(obj["mode"], "weights")
        self.assertTrue((Path(obj["payload"]) / "actor/model/dist_ckpt/__1_0.distcp").exists())
        self.assertFalse((Path(obj["payload"]) / "data.pt").exists())

    def test_one_missing_model_shard_is_not_declared_recovered(self):
        record = self.cp(125)
        (Path(record["path"]) / "actor/model/dist_ckpt/__1_0.distcp").unlink()
        with self.assertRaises(RuntimeError): self.archive.put(record)
        self.assertEqual(self.archive.objects(), [])

    def test_model_metadata_and_no_follow_symlinks(self):
        record = self.cp(125)
        path = Path(record["path"]) / "actor/model/dist_ckpt/__0_0.distcp"
        saved = path.with_name("elsewhere")
        path.rename(saved); path.symlink_to(saved)
        with self.assertRaises(RuntimeError): self.archive.put(record)

    def test_disk_or_link_failure_does_not_publish_success_or_delete_source(self):
        record = self.cp(125)
        with patch.object(best.os, "link", side_effect=OSError(errno.ENOSPC, "quota")):
            with self.assertRaises(OSError): self.archive.put(record)
        self.assertEqual(self.archive.objects(), [])
        ck.validate(record)

    def test_quarantine_does_not_claim_full_restore(self):
        record = self.cp(125)
        obj = self.archive.put(record, hold="legacy", trusted_inventory=False)
        self.assertEqual(obj["mode"], "unverified_weights")
        self.assertFalse((Path(obj["payload"]) / "data.pt").exists())

    def test_macro_best_full_and_domain_best_weights_survive_worse_later_scores(self):
        a, b = self.cp(125), self.cp(150)
        self.archive.accept(self.proof(a, [.9,.9,.9,.6]), a)
        self.archive.accept(self.proof(b, [.4,.4,.4,1.]), b)
        self.archive.put(self.cp(155)); self.archive.put(self.cp(160))
        index = self.archive.reconcile({self.lineage}, {self.lineage})
        cohort = next(iter(index["cohorts"].values()))
        self.assertEqual(cohort["overall"]["step"], 125)
        self.assertEqual(cohort["overall"]["mode"], "full")
        self.assertEqual(cohort["logic_algorithmic"]["step"], 150)
        self.assertEqual(cohort["logic_algorithmic"]["mode"], "weights")
        obj = read(self.archive.object_dir(ev.checkpoint_key(b)) / "manifest.json")
        self.assertEqual(obj["mode"], "weights")
        self.archive.verify(obj)
        ck.validate(a); ck.validate(b)  # Only archive-owned links were removed.

    def test_incomplete_validation_cannot_replace_best_or_be_unpinned(self):
        old, new = self.cp(125), self.cp(150)
        self.archive.accept(self.proof(old, [.5]*4), old)
        request = self.proof(new, [.9]*4)
        output = Path(request["job_dir"]) / "val.jsonl"
        output.write_text(output.read_text().splitlines()[0] + "\n")
        self.archive.put(new)
        with self.assertRaises(RuntimeError): self.archive.accept(request, new)
        self.archive.put(self.cp(155)); self.archive.put(self.cp(160))
        index = self.archive.reconcile({self.lineage}, {self.lineage})
        self.assertEqual(next(iter(index["cohorts"].values()))["overall"]["step"],125)
        self.assertIn(ev.checkpoint_key(new), index["full_keep"])

    def test_different_protocols_are_ranked_separately(self):
        a, b = self.cp(125), self.cp(150)
        self.archive.accept(self.proof(a,[.9]*4,"verifier-v1"),a)
        self.archive.accept(self.proof(b,[.5]*4,"verifier-v2"),b)
        index = self.archive.reconcile({self.lineage},{self.lineage})
        self.assertEqual(len(index["cohorts"]),2)
        self.assertEqual({x['overall']['step'] for x in index['cohorts'].values()},{125,150})

    def test_equal_scores_keep_earlier_step(self):
        a,b = self.cp(125),self.cp(150)
        for r in (b,a): self.archive.accept(self.proof(r,[.5]*4),r)
        index=self.archive.reconcile({self.lineage},{self.lineage})
        self.assertEqual(next(iter(index['cohorts'].values()))['overall']['step'],125)

    def test_uncommitted_candidate_cannot_win_and_is_admitted_after_commit(self):
        a,b=self.cp(125),self.cp(150,lineage='candidate')
        self.archive.accept(self.proof(a,[.5]*4),a)
        self.archive.accept(self.proof(b,[1.]*4,role='candidate'),b)
        first=self.archive.reconcile({self.lineage},{self.lineage,'candidate'})
        self.assertEqual(next(iter(first['cohorts'].values()))['overall']['step'],125)
        second=self.archive.reconcile({self.lineage,'candidate'},{'candidate'})
        self.assertEqual(next(iter(second['cohorts'].values()))['overall']['step'],150)

    def test_foreign_setup_or_checkpoint_binding_rejected(self):
        record=self.cp(125); request=self.proof(record,[.5]*4)
        wrong=copy.deepcopy(request);wrong['kind']='domain'
        with self.assertRaises(RuntimeError):self.archive.accept(wrong,record)
        other=self.cp(150)
        with self.assertRaises(RuntimeError):self.archive.accept(request,other)

    def test_source_race_never_publishes_partial_archive(self):
        record=self.cp(125); original=best.os.link; calls=[]
        def race(source,destination,**kwargs):
            calls.append(source)
            if len(calls)==3: raise FileNotFoundError('writer pruned checkpoint')
            return original(source,destination,**kwargs)
        with patch.object(best.os,'link',side_effect=race):
            with self.assertRaises(FileNotFoundError):self.archive.put(record)
        self.assertEqual(self.archive.objects(),[])
        self.archive.put(record)

    def test_crash_after_link_publish_recovers_even_after_origin_is_pruned(self):
        record=self.cp(125)
        original=best.write
        def crash(path,value):
            if Path(path).name=='manifest.json':raise OSError('crash before manifest')
            return original(path,value)
        with patch.object(best,'write',side_effect=crash):
            with self.assertRaises(OSError):self.archive.put(record)
        shutil.rmtree(record['path'])
        result=bridge.tick(self.root,self.kind)
        self.assertTrue(result['captured'])
        recovered=self.archive.objects()[0]
        self.assertEqual(recovered['mode'],'full')
        self.archive.verify(recovered)

    def test_crash_during_archive_gc_does_not_leave_a_broken_live_object(self):
        for step in (151,155,160):self.archive.put(self.cp(step))
        with patch.object(best.shutil,'rmtree',side_effect=OSError('interrupted unlink')):
            with self.assertRaises(OSError):self.archive.reconcile({self.lineage},{self.lineage})
        self.assertEqual({x['step'] for x in self.archive.objects()},{155,160})
        for obj in self.archive.objects():self.archive.verify(obj)
        self.archive.reconcile({self.lineage},{self.lineage})

    def test_bad_new_archive_does_not_remove_previous_best(self):
        a,b=self.cp(125),self.cp(150)
        self.archive.accept(self.proof(a,[.5]*4),a)
        self.archive.reconcile({self.lineage},{self.lineage})
        before=read(self.archive.path/'index.json')
        with patch.object(best.os,'link',side_effect=OSError('quota')):
            with self.assertRaises(OSError):self.archive.accept(self.proof(b,[.9]*4),b)
        self.assertEqual(read(self.archive.path/'index.json'),before)
        self.archive.verify(self.archive.objects()[0])

    def test_index_write_failure_precedes_any_archive_deletion(self):
        a,b,c=self.cp(151),self.cp(155),self.cp(160)
        for record in (a,b,c):self.archive.put(record)
        original=best.write
        def fail(path,value):
            if Path(path).name=='index.json':raise OSError('disk full')
            return original(path,value)
        with patch.object(best,'write',side_effect=fail):
            with self.assertRaises(OSError):self.archive.reconcile({self.lineage},{self.lineage})
        self.assertEqual(len(self.archive.objects()),3)
        for obj in self.archive.objects():self.archive.verify(obj)

    def test_rescue_hold_survives_worse_score_and_newer_checkpoints(self):
        old=self.cp(100);self.archive.put(old,hold='historical-rescue-step-100')
        self.archive.accept(self.proof(old,[.1]*4),old)
        new=self.cp(150);self.archive.accept(self.proof(new,[.9]*4),new)
        self.archive.put(self.cp(155));self.archive.put(self.cp(160))
        self.archive.reconcile({self.lineage},{self.lineage})
        self.archive.verify(read(self.archive.object_dir(ev.checkpoint_key(old))/'manifest.json'))

    def test_bridge_scans_existing_job_source_and_verified_receipt(self):
        record=self.cp(150);self.proof(record,[.7]*4)
        result=bridge.tick(self.root,self.kind,historical=True)
        self.assertEqual(next(iter(result['cohorts'].values()))['overall']['step'],150)
        self.assertEqual(result['rescue_targets']['150'][0]['mode'],'full')
        shutil.rmtree(record['path'])
        again=bridge.tick(self.root,self.kind)
        self.assertEqual(next(iter(again['cohorts'].values()))['overall']['step'],150)

    def test_changed_verified_receipt_is_reported(self):
        record=self.cp(150);request=self.proof(record,[.7]*4)
        bridge.tick(self.root,self.kind)
        path=ev.receipt_path(request,record);receipt=read(path)
        receipt['metrics'][ev.CORE[0]]=1.
        write(path,receipt)
        self.assertTrue(bridge.tick(self.root,self.kind)['receipt_errors'])

    def test_native_hook_stops_before_rolling_delete_if_archive_fails(self):
        from d4_training_continuation.trainer import trainer_class
        record=self.cp(125)
        class Base:
            def _save_checkpoint(trainer):
                (Path(trainer.config.trainer.default_local_dir)/'global_step_125').mkdir()
        trainer=trainer_class(Base)()
        trainer.global_steps=125;trainer.cont_saved=None;trainer.cont_plan=None
        trainer.cont_req=dict(role='formal',run_dir=str(self.p['run']),lineage=self.lineage,kind=self.kind)
        trainer.cont_resume=record;trainer.cont_phase=self.root/'phase'
        write(trainer.cont_phase/'phase.json',dict(target_world=8))
        trainer.config=types.SimpleNamespace(trainer=types.SimpleNamespace(
            default_local_dir=str(Path(record['path']).parent),max_actor_ckpt_to_keep=None))
        with patch.object(ck,'commit',return_value=record),patch.object(ck,'retain') as retain, \
             patch.object(best,'capture_saved',side_effect=OSError('quota')):
            with self.assertRaises(OSError):trainer._save_checkpoint()
            retain.assert_not_called()
        ck.validate(record)

    def test_rescue_missing_domain125_is_explicit_not_success(self):
        result=bridge.tick(self.root,self.kind,historical=True)
        self.assertEqual(result['rescue_targets']['125'],[])

    def test_wrong_setup_cannot_capture_foreign_payload(self):
        record=self.cp(125,kind='domain')
        with self.assertRaisesRegex(RuntimeError,'outside this setup'):self.archive.put(record)

    def test_best_weights_without_optimizer_keep_separate_best_full(self):
        a,b=self.cp(100),self.cp(125)
        ra,rb=self.proof(a,[.5]*4),self.proof(b,[.9]*4)
        shutil.rmtree(Path(b['path'])/'actor/optimizer')
        self.archive.accept(ra,a);self.archive.accept(rb,b)
        self.archive.put(self.cp(155));self.archive.put(self.cp(160))
        index=self.archive.reconcile({self.lineage},{self.lineage})
        cohort=next(iter(index['cohorts'].values()))
        self.assertEqual((cohort['overall']['step'],cohort['overall']['mode']),(125,'weights'))
        self.assertEqual((cohort['overall_full']['step'],cohort['overall_full']['mode']),(100,'full'))


class DeploymentTests(unittest.TestCase):
    setUp = BestTests.setUp
    cp = BestTests.cp
    request = BestTests.request
    proof = BestTests.proof
    def plan(self,kind='random',host=None,deadline='2099-01-01T00:00:00+00:00'):
        from d4_training_continuation import VERSION
        p=paths(self.root,kind);operation=p['control']/'operations/test'
        template=operation/'template'
        package=Path(best.__file__).parent
        shutil.copytree(package,template/'runtime/d4_training_continuation',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (template/'frozen_reward.py').write_text('# fixed scoring\n')
        plan=dict(root=str(self.root),kind=kind,operation='test',template=str(template),template_files=files(template),
                  training_policy={'save_freq':5},validation_policy={'complete':True},opportunistic={'unchanged':True})
        write(operation/'plan.json',plan)
        state=dict(version=VERSION,operation='test',migration_complete=True,phase='ACTIVE',
                   authority={'lineage':self.lineage},host=host or socket.gethostname(),deadline_utc=deadline,
                   promoted=True,pending={},opportunity={'phase':'QUALIFYING','job':None},pid=os.getpid())
        write(p['control']/'state.json',state)
        return operation/'plan.json',state

    def test_remote_owner_is_never_signalled_and_native_remains_explicitly_pending(self):
        from d4_training_continuation import cli,cancellation_repair
        planfile,state=self.plan(host='different-login')
        before=planfile.read_bytes()
        with patch.object(upgrade,'active',return_value=True),patch.object(cli,'runtime_status',return_value={'verified':False}), \
             patch.object(cancellation_repair,'stop_and_acquire') as stop,patch.object(cli,'spawn') as spawn:
            result=upgrade.deploy_kind(self.root,'random',signal_local=True)
        stop.assert_not_called();spawn.assert_not_called()
        self.assertEqual(result['status'],'BRIDGE_ACTIVE_NATIVE_PENDING_OWNER')
        self.assertEqual(planfile.read_bytes(),before)
        self.assertEqual(read(self.p['control']/'state.json'),state)

    def test_idle_expired_owner_template_installs_once_without_extending_or_starting_training(self):
        from d4_training_continuation import cli
        planfile,state=self.plan(deadline='2000-01-01T00:00:00+00:00')
        original=read(planfile);old_bytes=files(original['template'])
        with patch.object(cli,'runtime_status',return_value={'verified':False}),patch.object(cli,'spawn') as spawn:
            first=upgrade.deploy_kind(self.root,'random')
            installed=planfile.read_bytes()
            second=upgrade.deploy_kind(self.root,'random')
        spawn.assert_not_called()
        self.assertEqual(first['status'],'NATIVE_PLAN_INSTALLED')
        self.assertEqual(second['status'],'NATIVE_PLAN_INSTALLED')
        self.assertEqual(planfile.read_bytes(),installed)
        self.assertEqual(read(self.p['control']/'state.json'),state)
        self.assertEqual(files(original['template']),old_bytes)
        self.assertEqual(read(planfile)['best_checkpoint_policy'],best.POLICY)

    def test_unknown_runtime_is_rejected_without_plan_publish(self):
        planfile,_=self.plan()
        plan=read(planfile);target=Path(plan['template'])/'runtime/d4_training_continuation/trainer.py'
        target.write_text(target.read_text()+'\n# unexpected modification\n')
        plan['template_files']=files(plan['template']);write(planfile,plan)
        before=planfile.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'Unsupported continuation source'):upgrade.prepare(self.root,'random')
        self.assertEqual(planfile.read_bytes(),before)

    def test_new_job_binds_policy_without_changing_sampling_or_update_configuration(self):
        from test_training_continuation import IntegrationTests
        from d4_training_continuation import source,validation_journal
        plan=IntegrationTests.template_plan(self,'domain')
        plan.update(best_checkpoint_policy=best.POLICY,validation_policy=validation_journal.POLICY)
        original=copy.deepcopy(plan['old_request'])
        with patch.object(ev,'semantics',return_value={'fixture':'stable-validation'}):
            made=source.make_job(plan,'formal',2,{'lineage':'domainlineage','record':{}},'2099-01-01T00:00:00+00:00')
            path=Path(made['folder'])/'request.json';request=source.validate_request(path)
            self.assertEqual(request['best_checkpoint_policy'],best.POLICY)
            self.assertEqual(request['contract']['best_checkpoint_policy'],best.POLICY)
            self.assertEqual(request['sampling'],original['sampling'])
            self.assertEqual(request['config']['training'],original['config']['training'])
            request.pop('best_checkpoint_policy');write(path,request)
            with self.assertRaisesRegex(RuntimeError,'best_checkpoint_policy'):source.validate_request(path)

    def test_status_reports_one_broken_setup_without_hiding_other(self):
        self.plan('random');self.plan('domain')
        (paths(self.root,'domain')['control']/'state.json').write_text('{bad')
        result=bridge.status(self.root)
        self.assertIsNotNone(result['setups']['domain']['policy_read_error'])
        self.assertIsNone(result['setups']['random']['policy_read_error'])

    def test_real_frozen_bridge_runs_from_any_host_and_survives_independent_setup_failure(self):
        from d4_training_continuation.common import acquire,active
        # Keep both controller leases held, proving the bridge needs no process
        # signalling or scheduler invocation even from another login host.
        handles=[]
        for kind in ('random','domain'):
            self.plan(kind,host='remote-login')
            handle=acquire(paths(self.root,kind)['control']/'controller.lock');handles.append(handle)
            self.addCleanup(handle.close)
        record=self.cp(150);self.proof(record,[.7]*4)
        # Break only domain metadata. Random rescue must still advance.
        (paths(self.root,'domain')['control']/'state.json').write_text('{broken')
        real_popen=subprocess.Popen;children=[]
        def spawn(*args,**kwargs):
            child=real_popen(*args,**kwargs);children.append(child);return child
        def cleanup():
            (bridge.service(self.root)/'stop.request').touch()
            for child in children:
                try:child.wait(timeout=4)
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3)
        self.addCleanup(cleanup)
        with patch.object(bridge.subprocess,'Popen',side_effect=spawn):bridge.start(self.root)
        directory=bridge.service(self.root);until=time.monotonic()+8
        while time.monotonic()<until:
            statefile=directory/'state.json'
            if statefile.exists() and self.archive.path.joinpath('index.json').exists():break
            time.sleep(.05)
        self.assertTrue(active(directory/'watch.lock'))
        index=read(self.archive.path/'index.json')
        self.assertEqual(next(iter(index['cohorts'].values()))['overall']['step'],150)
        self.assertIn('domain',read(directory/'state.json')['errors'])
        self.assertTrue((directory/'runtimes').exists())




if __name__=='__main__':unittest.main()
