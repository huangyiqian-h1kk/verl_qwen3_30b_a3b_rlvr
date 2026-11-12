"""Native metadata lifecycle, audited frozen adapter, and worker import paths."""
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from d4_mopd import ARMS, DOMAINS, cuda_repair, integration, launcher_repair, metadata_repair, preflight
from d4_mopd.batch_contract import probe, replay
from d4_teachers.common import files, read, write
from d4_training_continuation.scheduler import Unknown
from test_cuda_runtime import TRACE
from test_launcher_repair import frozen

ROOT = Path(__file__).resolve().parents[2]


def test_pinned_verl_reproduces_old_index_failure_and_corrected_check_passes(spec):
    batch, expected = replay(DOMAINS)
    assert "index" not in batch.non_tensor_batch
    source = inspect.getsource(integration.check_batch)
    old_source = source.replace(metadata_repair.NEW_IDS, metadata_repair.OLD_IDS)
    assert old_source != source
    namespace = dict(vars(integration))
    exec(old_source, namespace)
    with pytest.raises(KeyError, match="index"):
        namespace["check_batch"](batch, dict(spec, train_batch_size=4), "domain_mix", 1, expected)
    result = integration.check_batch(batch, dict(spec, train_batch_size=4), "domain_mix", 1, expected)
    assert set(result["domains"]) == set(DOMAINS)


def test_native_lifecycle_in_both_reward_metadata_branches():
    result = probe()
    assert result["status"] == "PASS" and result["gpu_execution_tested"] is False
    assert len(result["cases"]) == 8
    for arm in ARMS:
        for step in (1, 5):
            cases = [r for r in result["cases"] if r["arm"] == arm and r["step"] == step]
            assert len({c["sample_ids_sha256"] for c in cases}) == 1


@pytest.mark.parametrize("corruption", ["missing-extra", "missing-id", "empty-id", "non-string", "wrong-id", "multiplicity"])
def test_uid_or_top_level_index_cannot_hide_wrong_stable_ids(spec, corruption):
    batch, expected = replay(DOMAINS)
    # These volatile/generation-only fields must not act as a fallback.
    batch.non_tensor_batch["index"] = np.array(expected[::-1])
    batch.non_tensor_batch["uid"] = np.array(expected[::-1])
    extra = batch.non_tensor_batch["extra_info"]
    if corruption == "missing-extra": batch.non_tensor_batch.pop("extra_info")
    if corruption == "missing-id": extra[0].pop("sample_id")
    if corruption == "empty-id": extra[0]["sample_id"] = ""
    if corruption == "non-string": extra[0]["sample_id"] = 42
    if corruption == "wrong-id": extra[0]["sample_id"] = "unplanned"
    if corruption == "multiplicity": extra[0]["sample_id"] = extra[1]["sample_id"]
    with pytest.raises(RuntimeError, match="prompt"):
        integration.check_batch(batch, dict(spec, train_batch_size=4), "domain_mix", 1, expected)


@pytest.mark.parametrize("valid", [False, True])
def test_actor_boundary_checks_before_update_and_records_only_success(spec, tmp_path, monkeypatch, valid):
    batch, expected = replay(DOMAINS)
    trainer = object.__new__(integration.MOPDTrainer)
    trainer.config = OmegaConf.create(dict(mopd=dict(
        run_dir=str(tmp_path), manifest_path=str(tmp_path / "manifest.json"), identity="fixture",
        effective_spec=dict(spec, train_batch_size=4), arm="domain_mix")))
    trainer.global_steps = 1
    trainer.mopd_started = time.monotonic()
    trainer.train_dataloader = SimpleNamespace(sampler=SimpleNamespace(indices=list(range(4))))
    trainer.train_dataset = SimpleNamespace(rows=[{"sample_id": v} for v in expected])
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "fixture")
    calls = []
    def update(self, value):
        calls.append(value)
        return SimpleNamespace(meta_info={"metrics": {"actor/grad_norm": 1., "distillation/loss": .5}})
    monkeypatch.setattr(integration.RayPPOTrainer, "_update_actor", update)
    if valid:
        trainer._update_actor(batch)
        assert calls == [batch]
        receipt = read(tmp_path / "attempts/fixture/update_1.json")
        assert receipt["optimizer_updates_per_rollout_batch"] == 1 and receipt["identity"] == "fixture"
    else:
        batch.non_tensor_batch["extra_info"][0]["sample_id"] = "wrong"
        with pytest.raises(RuntimeError, match="prompt position"):
            trainer._update_actor(batch)
        assert not calls and not list(tmp_path.glob("attempts/*/update_*.json"))


@pytest.fixture
def metadata_case(frozen, tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    for name in ("metadata_repair.py", "batch_contract.py"):
        (runtime / "d4_mopd" / name).unlink()
    adapter = runtime / "d4_mopd/integration.py"
    adapter.write_text(adapter.read_text().replace(metadata_repair.NEW_IDS, metadata_repair.OLD_IDS))
    for name in ("multidomain", "d4_teachers", "d4_training_continuation", "config/mopd"):
        shutil.copytree(ROOT / name, runtime / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen["runtime_files"] = files(runtime)
    write(tmp_path / "manifest.json", frozen)
    # Materialize the exact previous CUDA plan; its algorithm/files must remain valid.
    changes, inventory, revision, folder = cuda_repair.revision_plan(frozen)
    shutil.copytree(runtime, folder / "runtime")
    for name, content in changes.items():
        (folder / "runtime" / name).write_text(content)
    write(folder / "receipt.json", dict(status="PASS", schema=cuda_repair.SCHEMA,
        identity=frozen["identity"], revision=revision, runtime_files=inventory))
    write(tmp_path / "launcher_repair.json", dict(schema=cuda_repair.SCHEMA, identity=frozen["identity"],
        revision=revision, folder=str(folder)))
    state = read(tmp_path / "states.json")
    for s in state.values():
        attempt = Path(s["job"]["folder"])
        write(attempt / "request.json", dict(identity=frozen["identity"], phase="qualification", arm=s["arm"],
                                             manifest=frozen["manifest_path"], folder=str(attempt)))
        if s["arm"] == "domain_mix":
            result = dict(status="FAIL", identity=frozen["identity"], error="1", traceback=TRACE)
        else:
            result = dict(status="FAIL", identity=frozen["identity"], error="RayTaskError(KeyError)",
                traceback=f'''Traceback (most recent call last):
  File "{adapter}", line 144, in _update_actor
    record = check_batch(batch, spec, arm, step, expected)
  File "{adapter}", line 53, in check_batch
{metadata_repair.OLD_IDS}KeyError: 'index'
''')
            write(attempt / "resume.json", dict(identity=frozen["identity"], step=0, checkpoint=None,
                                                model_optimizer_scheduler_rng_and_loader=False))
        write(attempt / "result.json", result)
        for prefix in ("cuda_runtime", "gpu_node"):
            for rank in (0, 1):
                write(attempt / f"{prefix}_{rank}.json", dict(status="PASS", rank=rank, identity=frozen["identity"]))
    from d4_mopd import control
    class PBS:
        def observe(self, job):
            if "domain_mix" in job["folder"]:
                return dict(job_state="F", Exit_status=271, comment="terminated by root@fixture")
            return dict(job_state="F", Exit_status=1)
    monkeypatch.setattr(control, "PBS", PBS)
    monkeypatch.setattr(metadata_repair, "verify", lambda _: frozen)
    return frozen


def test_migration_preserves_previous_revision_and_model_recipe(metadata_case, tmp_path):
    m = metadata_case
    original_manifest = (tmp_path / "manifest.json").read_bytes()
    before = read(tmp_path / "states.json")
    previous = launcher_repair.active_runtime(m)
    previous_files = files(previous)
    result = metadata_repair.repair(m["manifest_path"])
    selected = launcher_repair.active_runtime(m)
    changed_from_v2 = {p for p, h in files(selected).items() if previous_files.get(p) != h}
    assert changed_from_v2 == {"d4_mopd/integration.py", "d4_mopd/cli.py", "d4_mopd/job_exit.py", "d4_mopd/batch_contract.py"}
    assert (selected / "d4_mopd/integration.py").read_text() == (ROOT / "d4_mopd/integration.py").read_text()
    assert files(previous) == previous_files and files(tmp_path / "runtime") == m["runtime_files"]
    assert (tmp_path / "manifest.json").read_bytes() == original_manifest
    assert read(tmp_path / "states.json") == before
    assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
    assert result["failure_evidence"]["qualification:domain_mix"] == "pbs_root_sigterm_explicit_retry_only"
    assert metadata_repair.repair(m["manifest_path"])["revision"] == result["revision"]
    # Model/configuration, loss and optimizer code remain byte-identical.
    assert (selected / "d4_mopd/config.py").read_bytes() == (tmp_path / "runtime/d4_mopd/config.py").read_bytes()
    assert (selected / "config/mopd/baseline.json").read_bytes() == (tmp_path / "runtime/config/mopd/baseline.json").read_bytes()


def test_driver_worker_and_cpu_probe_use_the_same_frozen_correction(metadata_case, tmp_path):
    m = metadata_case
    metadata_repair.repair(m["manifest_path"])
    selected = launcher_repair.active_runtime(m)
    for arm in ARMS:
        write(tmp_path / "runs" / arm / "receipt.json", dict(status="PASS", identity=m["identity"], selection={"step": 10}))
    # A clean guardian-side interpreter imports the snapshot, not test-process modules.
    code = ("import sys,json; sys.path[:0]=" + repr([str(selected), m["verl_src"]])
            + "; from d4_mopd.control import new_attempt; m=" + repr(m)
            + "; print(json.dumps([new_attempt(m,phase,arm) for arm in " + repr(ARMS)
            + " for phase in ('qualification','train','test')]))")
    guardian = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True, timeout=30)
    for job in json.loads(guardian.stdout):
        attempt = Path(job["folder"])
        assert str(selected / "d4_mopd/node.sh") in (attempt / "stage.pbs").read_text()
        p = subprocess.run([sys.executable, "-I", str(attempt / "entry.py"), "paths", "--request", str(attempt / "request.json")],
                           capture_output=True, text=True, check=True, timeout=30)
        assert p.stdout.strip() == str(selected) + os.pathsep + m["verl_src"]
    # The same PYTHONPATH is passed to remote Ray actors by node.sh and cli.py.
    worker = subprocess.run([sys.executable, "-c", "import d4_mopd; print(d4_mopd.__file__)"], cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": p.stdout.strip()}, capture_output=True, text=True, check=True, timeout=30)
    assert worker.stdout.strip() == str(selected / "d4_mopd/__init__.py")
    proof = preflight.batch_metadata_probe(m, selected)
    assert proof["status"] == "PASS" and proof["runtime"] == str(selected) and len(proof["cases"]) == 8


@pytest.mark.parametrize("change", ["update", "checkpoint", "resume", "missing-resume", "wrong-identity", "wrong-error",
                                    "generic-exit", "non-root", "missing-proof", "node-fail", "running", "unknown"])
def test_repair_does_not_rearm_unrelated_failures(metadata_case, tmp_path, monkeypatch, change):
    from d4_mopd import control
    m = metadata_case
    state = read(tmp_path / "states.json")
    only = Path(state["qualification:domain_only"]["job"]["folder"])
    mix = Path(state["qualification:domain_mix"]["job"]["folder"])
    if change == "update": write(mix / "update_1.json", {"step": 1})
    if change == "checkpoint": (mix.parent.parent / "checkpoints/global_step_4").mkdir(parents=True)
    if change == "resume": write(only / "resume.json", dict(identity=m["identity"], step=4, checkpoint="saved"))
    if change == "missing-resume": (only / "resume.json").unlink()
    if change in ("wrong-identity", "wrong-error", "generic-exit"):
        path = (mix if change == "generic-exit" else only) / "result.json"
        row = read(path)
        if change == "wrong-identity": row["identity"] = "other"
        else: row["traceback"] = "SystemExit: 1" if change == "generic-exit" else "KeyError: 'other'"
        write(path, row)
    if change == "missing-proof": (mix / "gpu_node_1.json").unlink()
    if change == "node-fail": write(mix / "cuda_runtime_1.json", dict(status="FAIL", error="compile failed"))
    if change in ("running", "unknown", "non-root"):
        class PBS:
            def observe(self, job):
                if change == "unknown": raise Unknown("qstat timeout")
                return dict(job_state="R" if change == "running" else "F", Exit_status=271, comment="terminated by user@fixture")
        monkeypatch.setattr(control, "PBS", PBS)
    pointer = (tmp_path / "launcher_repair.json").read_bytes()
    with pytest.raises((RuntimeError, FileNotFoundError, Unknown)):
        metadata_repair.repair(m["manifest_path"])
    assert (tmp_path / "launcher_repair.json").read_bytes() == pointer
    assert read(tmp_path / "states.json") == state
    assert len(list((tmp_path / "launcher_revisions").iterdir())) == 1
