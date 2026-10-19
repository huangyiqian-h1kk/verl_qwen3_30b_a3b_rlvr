"""CPU regression of pinned verl dispatch and guarded candidate migration."""
import ast
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures/switch_worker"
spec = importlib.util.spec_from_file_location("switch_repair", REPO / "scripts/multidomain/repair_switch_worker.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def factory_fixture():
    """Actual pinned factory; stand-ins only for Ray and training side effects."""
    events = []
    magic = "_test_rpc_metadata"
    dispatch = SimpleNamespace(ONE_TO_ALL="all", DIRECT_ROLLOUT_METHOD="direct")

    def register(**metadata):
        def wrap(f):
            setattr(f, magic, metadata)
            return f
        return wrap

    class Worker:
        def __init__(self):
            if os.environ.get("DISABLE_WORKER_INIT") != "1":
                events.append(("base",))

    class ActorRolloutRefWorker(Worker):
        def __init__(self, config, role):
            Worker.__init__(self)
            self.config, self.role = config, role
            events.append(("actor", config, role))

        @register(dispatch_mode=dispatch.ONE_TO_ALL)
        def constructor_probe(self):
            return self.config, self.role

    class MultiDomainWorker(ActorRolloutRefWorker):
        pass

    class TwoNodeWorker(MultiDomainWorker):
        def load_checkpoint(self, *a):
            return "base load"

    class Remote:
        def __init__(self, cls):
            # Ray adds its own wrapper subclass; its extra MRO depth is harmless
            # when application class names do not collide with the reserved name.
            self.__ray_actor_class__ = type(cls.__name__, (cls,), {})

    class Args:
        def __init__(self, cls, *args, **kwargs):
            self.cls, self.args, self.kwargs = cls, args, kwargs

    @contextmanager
    def temp_env_var(key, value):
        with patch.dict(os.environ, {key: value}):
            yield

    ns = dict(Worker=Worker, RayClassWithInitArgs=Args, ray=SimpleNamespace(remote=Remote),
              temp_env_var=temp_env_var, inspect=inspect, MAGIC_ATTR=magic, Dispatch=dispatch)
    exec(compile((FIXTURES / "verl_colocated_pinned.py").read_text(), "pinned_verl_factory", "exec"), ns)
    return ns, TwoNodeWorker, events, register


def candidate_class(source, two_node, register, dispatch):
    tree = ast.parse(source)
    # Avoid imports of torch/Megatron. Compile the unmodified class body plus
    # its new alias; Python must create real zero-argument super() closures.
    body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.Assign))]
    module = ast.Module(body=body, type_ignores=[])
    ns = dict(TwoNodeWorker=two_node, register=register, Dispatch=dispatch, __name__="test_candidate")
    exec(compile(module, "actual_candidate_worker", "exec"), ns)
    return ns["Worker"]


class ConstructionTests(unittest.TestCase):
    def test_actual_pinned_factory_and_selector_are_fingerprinted(self):
        repair.check_verl(FIXTURES / "verl_colocated_pinned.py")

    def test_common_check_constructs_but_both_old_candidates_fail(self):
        ns, two_node, events, register = factory_fixture()
        args = ns["RayClassWithInitArgs"]
        remote = ns["ray"].remote
        create = ns["create_colocated_worker_cls"]
        with redirect_stdout(io.StringIO()):
            common = create({"actor": args(remote(two_node), config={"train": 1}, role="actor_rollout")})
        obj = common.cls.__ray_actor_class__()
        self.assertEqual(obj.actor_constructor_probe(), ({"train": 1}, "actor_rollout"))
        self.assertEqual(events, [("base",), ("actor", {"train": 1}, "actor_rollout")])
        for kind in repair.ROLES:
            events.clear()
            source = (FIXTURES / (kind + "_worker_before.py")).read_text()
            cls = candidate_class(source, two_node, register, ns["Dispatch"])
            with redirect_stdout(io.StringIO()):
                old = create({"actor": args(remote(cls), config={}, role="actor_rollout")})
            with self.assertRaisesRegex(TypeError, "missing 2 required positional arguments: 'config' and 'role'"):
                old.cls.__ray_actor_class__()
            self.assertEqual(events, [])  # It fails before any real model constructor.

    def test_fixed_candidates_forward_arguments_once_and_keep_rpc_and_super(self):
        for kind in repair.ROLES:
            ns, two_node, events, register = factory_fixture()
            source = (FIXTURES / (kind + "_worker_before.py")).read_text()
            fixed = repair.patched_worker(kind, source)
            cls = candidate_class(fixed, two_node, register, ns["Dispatch"])
            self.assertEqual(cls.__name__, repair.ROLES[kind][3])
            self.assertEqual(cls.__bases__, (two_node,))
            # Both load and save use the renamed class in their super() closure.
            for method in (cls.load_checkpoint, cls.save_checkpoint):
                cells = dict(zip(method.__code__.co_freevars, method.__closure__))
                self.assertIs(cells["__class__"].cell_contents, cls)
            args, remote = ns["RayClassWithInitArgs"], ns["ray"].remote
            config = {"optimizer": "preserve", "sampler": kind}
            with redirect_stdout(io.StringIO()):
                result = ns["create_colocated_worker_cls"]({
                    "actor": args(remote(cls), config=config, role="actor_rollout"),
                    "ref": args(remote(two_node), config=config, role="ref"),
                })
            obj = result.cls.__ray_actor_class__()
            self.assertIs(obj.worker_dict["actor"].config, config)
            self.assertEqual(obj.actor_constructor_probe(), (config, "actor_rollout"))
            self.assertEqual(obj.ref_constructor_probe(), (config, "ref"))
            self.assertEqual(events, [("base",), ("actor", config, "actor_rollout"), ("actor", config, "ref")])
            self.assertEqual(getattr(type(obj).actor_load_checkpoint, ns["MAGIC_ATTR"]),
                             getattr(cls.load_checkpoint, ns["MAGIC_ATTR"]))

    def test_changed_worker_source_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
            repair.patched_worker("random", "class Worker: pass\n")


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kind = "random"
        self.control = self.root / "outputs/multidomain" / repair.ROLES["random"][0] / "handoff"
        self.control.mkdir(parents=True)
        self.worker = self.control / "runtime/d4_managed_switch/worker.py"
        self.worker.parent.mkdir(parents=True)
        self.worker.write_bytes((FIXTURES / "random_worker_before.py").read_bytes())
        self.context = self.make_context("random", self.control)
        for name, value in (("plan.json", self.context["plan"]), ("state.json", self.context["state"]),
                            ("runtime_limits_30h_24h/config.json", self.context["limits"])):
            repair.write(self.control / name, value)

    def make_context(self, kind, control):
        package = repair.ROLES[kind][1]
        hashes = {package: {"worker.py": repair.OLD_WORKERS[kind]}, "d4_two_node_resumefix": {"worker.py": "check-sha"}}
        contract = dict(check_identity="accepted-common", run_dir=str(control.parent), sampling={"plan": [0, 1, 2, 3]})
        if kind == "random":
            contract["package"] = hashes[package]
        else:
            contract["packages"] = hashes
        plan = dict(identity=repair.digest(contract), contract=contract, hours=30,
                    info={"identity": "accepted-common", "identity_inputs": {"package": hashes["d4_two_node_resumefix"]}},
                    sampling={"offset": 42}, config={"must": "not change"})
        state = dict(identity=plan["identity"], phase="WAIT_READY", controller_status="STOPPED", submissions=1,
                     single={"job_id": "900.pbs1"}, dual={"job_id": "901.pbs1", "folder": str(control.parent / "jobs/old")},
                     handoff_blocked="candidate exit 1", pending_job=None, pending_check=None)
        limits = dict(contract_identity=plan["identity"], guarded_packages=copy.deepcopy(hashes),
                      monitor_default_hours=30, future_random_training_walltime="24:00:00")
        worker = control / "runtime" / package / "worker.py"
        worker.parent.mkdir(parents=True, exist_ok=True)
        worker.write_bytes((FIXTURES / (kind + "_worker_before.py")).read_bytes())
        return dict(kind=kind, control=control, worker=worker, plan=plan, state=state, limits=limits,
                    candidate_observation={"job_state": "F", "Exit_status": 1}, fixed=False)

    def test_only_worker_hash_contract_and_rearm_metadata_change(self):
        for kind in repair.ROLES:
            context = self.make_context(kind, self.control / kind)
            before = copy.deepcopy(context["plan"])
            edited = repair.changes(context)
            plan, state, limits = (json.loads(edited[n]) for n in ("plan.json", "state.json", "runtime_limits_30h_24h/config.json"))
            self.assertNotEqual(plan["identity"], before["identity"])
            self.assertEqual(plan["identity"], repair.digest(plan["contract"]))
            self.assertEqual(state["identity"], limits["contract_identity"])
            self.assertEqual(plan["info"], before["info"])
            self.assertEqual(plan["config"], before["config"])
            self.assertEqual(plan["sampling"], before["sampling"])
            self.assertEqual(state["single"], context["state"]["single"])
            self.assertEqual(state["submissions"], 1)
            self.assertEqual(state["phase"], "WAIT_CHECKPOINT")
            self.assertEqual(state["previous_candidate"], context["state"]["dual"])
            self.assertIsNone(state["dual"])
            self.assertIsNone(state["handoff_blocked"])
            self.assertEqual(limits["future_random_training_walltime"], "24:00:00")
            self.assertEqual(limits["guarded_packages"], repair.package_contract(kind, plan))
            self.assertEqual(context["plan"], before)

    def test_publication_keeps_old_candidate_and_checkpoints_untouched(self):
        old = self.control.parent / "jobs/old"
        old.mkdir(parents=True)
        (old / "request.json").write_text("historical candidate")
        checkpoint = self.root / "single/checkpoints/global_step_65/actor/data.bin"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"real states untouched")
        before = repair.sha(checkpoint)
        repair.publish(self.context)
        self.assertEqual(repair.read(self.control / "worker_name_repair/transaction.json")["status"], "INSTALLED")
        self.assertEqual(repair.sha(checkpoint), before)
        self.assertEqual((old / "request.json").read_text(), "historical candidate")
        self.assertEqual(repair.read(self.control / "state.json")["phase"], "WAIT_CHECKPOINT")

    def test_failed_publication_rolls_back_every_file(self):
        paths = [self.worker, self.control / "plan.json", self.control / "state.json",
                 self.control / "runtime_limits_30h_24h/config.json"]
        before = {p: p.read_bytes() for p in paths}
        original = repair.atomic_text
        failed = [False]

        def fail_once(path, value):
            if path == self.control / "state.json" and not failed[0]:
                failed[0] = True
                raise OSError("simulated failed filesystem write")
            return original(path, value)

        with patch.object(repair, "atomic_text", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "filesystem"):
                repair.publish(self.context)
        self.assertEqual({p: p.read_bytes() for p in paths}, before)
        self.assertFalse((self.control / "worker_name_repair/receipt.json").exists())

    def test_interrupted_publication_recovers_before_retry(self):
        before = self.worker.read_text()
        after = repair.patched_worker("random", before)
        repair.write(self.control / "worker_name_repair/transaction.json", {
            "version": repair.VERSION, "status": "PREPARED", "files": {
                "runtime/d4_managed_switch/worker.py": {"before": before, "after": after},
            }})
        self.worker.write_text(after)
        repair.recover_interrupted(self.control)
        self.assertEqual(self.worker.read_text(), before)
        repair.publish(self.context)
        self.assertEqual(self.worker.read_text(), after)

    def test_recovery_rejects_an_independent_edit(self):
        repair.write(self.control / "worker_name_repair/transaction.json", {
            "version": repair.VERSION, "status": "PREPARED", "files": {
                "plan.json": {"before": "old", "after": "new"}}})
        with self.assertRaisesRegex(RuntimeError, "independently"):
            repair.recover_interrupted(self.control)


class PreflightTests(unittest.TestCase):
    """Exercise real validation against an on-disk miniature deployment."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidates = {}
        self.contexts = {}
        self.rows = {}
        known = {}
        limits_text = "# Test-only operational adapter\n"
        limits_sha = repair.text_sha(limits_text)
        patcher = patch.object(repair, "LIMITS_SHA", limits_sha)
        patcher.start()
        self.addCleanup(patcher.stop)
        verl = self.root / "verl-src/verl/single_controller/ray/base.py"
        verl.parent.mkdir(parents=True)
        verl.write_bytes((FIXTURES / "verl_colocated_pinned.py").read_bytes())
        check = self.root / "outputs/multidomain" / (repair.BASE + "_n2_v1") / "jobs/common_check"
        check.mkdir(parents=True)
        (check / "job_id.txt").write_text("899.pbs1\n")
        (check / "proof.json").write_text("passed common evidence")
        acceptance = dict(status="PASS", identity="common-identity", grpo_steps_completed=2,
                          job_id="899.pbs1", evidence={str(check / "proof.json"): repair.sha(check / "proof.json")})
        repair.write(check / "result.json", acceptance)
        for index, (kind, (run, package, job_name, _)) in enumerate(repair.ROLES.items()):
            control = self.root / "outputs/multidomain" / run / "handoff"
            worker = control / "runtime" / package / "worker.py"
            worker.parent.mkdir(parents=True)
            worker.write_bytes((FIXTURES / (kind + "_worker_before.py")).read_bytes())
            n2 = control / "runtime/d4_two_node_resumefix"
            n2.mkdir()
            (n2 / "__init__.py").write_text("# Common package remains byte-identical\n")
            hashes = {package: repair.packages(worker.parent), "d4_two_node_resumefix": repair.packages(n2)}
            known[kind] = {repair.digest(hashes[package])}
            contract = dict(check_identity="common-identity", run_dir=str(control.parent))
            contract["package" if kind == "random" else "packages"] = hashes[package] if kind == "random" else hashes
            plan = dict(contract=contract, identity=repair.digest(contract), hours=30,
                        info={"identity": "common-identity", "identity_inputs": {"package": hashes["d4_two_node_resumefix"]},
                              "base_config": {"verl_src": str(verl.parents[3])}}, single_monitor_files={})
            folder = control.parent / "jobs/failed_train"
            folder.mkdir(parents=True)
            import shutil
            shutil.copytree(control / "runtime", folder / "runtime")
            jid = f"{901 + index}.pbs1"
            self.candidates[kind] = jid
            (folder / "job_id.txt").write_text(jid + "\n")
            (folder / "stage.pbs").write_text(f"#PBS -q {repair.QUEUE}\n#PBS -N {job_name}\n")
            (folder / "pbs.log").write_text("verl/single_controller/ray/base.py\nsuper().__init__()\nTypeError: " + repair.ERROR)
            request = dict(identity=plan["identity"], contract=contract, info=plan["info"], expected_nodes=2,
                           snapshot_files={"stage.pbs": repair.sha(folder / "stage.pbs")})
            repair.write(folder / "request.json", request)
            repair.write(folder / "result.json", dict(status="FAIL", job_id=jid, identity=plan["identity"]))
            state = dict(identity=plan["identity"], phase="WAIT_READY", controller_status="STOPPED", submissions=1,
                         single={"job_id": "900.pbs1"}, dual={"job_id": jid, "folder": str(folder), "job_name": job_name},
                         handoff_blocked="candidate exit 1", pending_job=None, pending_check=None)
            cfg = dict(root=str(self.root), kind=kind, contract_identity=plan["identity"], guarded_packages=hashes,
                       policy_sha256=limits_sha, monitor_default_hours=30, future_random_training_walltime="24:00:00")
            policy = control / "runtime_limits_30h_24h"
            policy.mkdir()
            (policy / "limits_policy.py").write_text(limits_text)
            repair.write(policy / "config.json", cfg)
            (control / "entry.py").write_text('from pathlib import Path\nimport hashlib, json, sys\n'
                'here=Path(__file__).resolve().parent\npolicy=here/"runtime_limits_30h_24h"\n'
                'if hashlib.sha256((policy/"limits_policy.py").read_bytes()).hexdigest()!=' + repr(limits_sha)
                + ': raise RuntimeError("Limits policy changed")\n'
                'sys.path.insert(0,str(policy))\nfrom limits_policy import boot\nboot(here,sys.argv[1:])\n')
            for name, obj in (("plan.json", plan), ("state.json", state), ("acceptance.json", acceptance)):
                repair.write(control / name, obj)
            (control / "controller.lock").touch()
            self.rows[jid] = dict(job_state="F", Exit_status=1, queue=repair.QUEUE, Job_Name=job_name,
                                  Submit_arguments=str(folder / "stage.pbs"))
            self.contexts[kind] = dict(control=control, folder=folder, worker=worker)
        patcher = patch.object(repair, "KNOWN_PACKAGES", known)
        patcher.start()
        self.addCleanup(patcher.stop)

    def inspect(self, kind):
        return repair.inspect(self.root, kind, self.candidates[kind], observe=lambda jid: self.rows[jid])

    def test_full_preflight_publish_and_idempotent_recheck(self):
        for kind in repair.ROLES:
            context = self.inspect(kind)
            self.assertFalse(context["fixed"])
            repair.publish(context)
            with patch.object(repair, "query", side_effect=AssertionError("Already fixed must not query old PBS history")):
                fixed = self.inspect(kind)
            self.assertTrue(fixed["fixed"])
            self.assertEqual(fixed["state"]["phase"], "WAIT_CHECKPOINT")
            # Rerunning after a different candidate starts must not clear it.
            state_path = context["control"] / "state.json"
            state = repair.read(state_path)
            state.update(phase="WAIT_READY", dual={"job_id": "999.pbs1"})
            repair.write(state_path, state)
            self.assertEqual(self.inspect(kind)["state"]["dual"]["job_id"], "999.pbs1")

    def test_active_unknown_or_nonmatching_pbs_job_blocks_rearm(self):
        jid = self.candidates["random"]
        original = copy.deepcopy(self.rows[jid])
        for update in ({"job_state": "R"}, {"Exit_status": 271}, {"queue": "rt_HF"},
                       {"Submit_arguments": "/another/stage.pbs"}):
            self.rows[jid] = {**original, **update}
            with self.assertRaises(RuntimeError):
                self.inspect("random")

    def test_different_failure_or_progress_evidence_blocks_rearm(self):
        folder = self.contexts["domain"]["folder"]
        log = folder / "pbs.log"
        old = log.read_text()
        log.write_text("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "diagnosed worker"):
            self.inspect("domain")
        log.write_text(old)
        (folder / "restore_rank_0.json").write_text("restored something")
        with self.assertRaisesRegex(RuntimeError, "restoration/update"):
            self.inspect("domain")

    def test_fence_or_pending_submission_blocks_rearm(self):
        control = self.contexts["random"]["control"]
        repair.write(control / "qdel.json", {"intent": "existing"})
        with self.assertRaisesRegex(RuntimeError, "Handoff evidence"):
            self.inspect("random")
        (control / "qdel.json").unlink()
        state = repair.read(control / "state.json")
        state["pending_job"] = "unresolved qsub"
        repair.write(control / "state.json", state)
        with self.assertRaisesRegex(RuntimeError, "already in progress"):
            self.inspect("random")

    def test_corrupt_snapshot_or_accepted_check_is_rejected(self):
        worker = self.contexts["random"]["worker"]
        worker.write_text(worker.read_text() + "\n# unrecorded change\n")
        with self.assertRaisesRegex(RuntimeError, "Frozen package changed"):
            self.inspect("random")
        common = self.root / "outputs/multidomain" / (repair.BASE + "_n2_v1") / "jobs/common_check/proof.json"
        common.write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "Common check evidence changed"):
            self.inspect("domain")


if __name__ == "__main__":
    unittest.main()
