"""CPU adoption checks with complete native-layout states and real file locks."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from d4_mopd import ARMS, DOMAINS, checkpoints
from d4_mopd.execution import inspect_resume, make_record, qualification_fields
from d4_mopd.metrics import select_best, source_metrics
from d4_mopd.spec import PlannedSampler
from d4_teachers.common import digest, file_sha, read, write
from d4_training_continuation.common import active, locked

MODULE = Path(__file__).resolve().parents[2] / "scripts/mopd_adopt_qualification.py"
spec = importlib.util.spec_from_file_location("mopd_adopt_test", MODULE)
adopt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adopt)


def checkpoint(root, m, arm, step, sampler, segment=None):
    folder = root / "checkpoints" / f"global_step_{step}"
    actor = folder / "actor"
    contents = {}
    for name in ("model", "optimizer", "lr_scheduler", "rng_state"):
        path = actor / name / "shard.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes((arm + name).encode())
        contents[name] = dict(path=name)
    write(actor / "ckpt_contents.json", dict(framework="megatron", global_step=step, world_size=8, contents=contents))
    (folder / "data.pt").write_bytes(b"complete-loader-cursor")
    (folder / "driver.pt").write_bytes(b"complete-driver-rng")
    if segment:
        write(folder / "qualification_segment.json", segment)
    checkpoints.commit(folder, m["identity"], arm, step, sampler, resumed_from=4 if step == 8 else 0)
    return folder


@pytest.fixture
def case(manifest, tmp_path):
    m = deepcopy(manifest)
    training = [dict(sample_id=f"train-{d}-{i}", domain=d, source="math" if d == "math" else d)
                for d in DOMAINS for i in range(3)]
    validation = [dict(sample_id=f"val-{d}-{i}", domain=d,
                       source=("math" if i < 128 else "deepmath_l8") if d == "math" else d)
                  for d in DOMAINS for i in range(256)]
    m["data_summary"] = {}
    for split, rows in (("train", training), ("validation", validation)):
        pq.write_table(pa.Table.from_pylist(rows), m["data"][split][0])
        m["data_summary"][split] = dict(ids_sha256=digest([r["sample_id"] for r in rows]))
    m["execution"] = make_record(m, "single_node", {"code.py": "0" * 64},
                                 qualification_transition=("single_node", "single_node"))
    control = Path(m["execution"]["control_root"])
    control.mkdir(parents=True)
    (control / "STOP").touch()
    (tmp_path / "STOP").touch()
    states, old_states, jobs, originals = {}, {}, {}, {}
    for n, arm in enumerate(ARMS):
        sampler = PlannedSampler(training, m["spec"], arm).identity
        run = control / "qualification" / arm
        segments = [dict(profile="single_node", attempt_id=f"{arm}-{start}-{end}", loaded_step=start,
                         completed_step=end, execution_identity=m["execution"]["execution_identity"],
                         runtime_identity=m["execution"]["runtime_identity"]) for start, end in ((0, 4), (4, 8))]
        source = checkpoint(run, m, arm, 8, sampler, segments[1])
        receipt = dict(status="PASS", identity=m["identity"], arm=arm, scope="qualification", step=8,
                       resumed_from=4, checkpoint=str(source), qualification_segments=segments, **qualification_fields(m))
        write(run / "receipt.json", receipt)
        attempt = run / "attempts" / segments[0]["attempt_id"]
        write(attempt / "request.json", dict(manifest=m["manifest_path"], identity=m["identity"], phase="qualification",
            arm=arm, folder=str(attempt), run_dir=str(run), execution_record=m["execution"]["record_path"],
            execution_identity=m["execution"]["execution_identity"], runtime_identity=m["execution"]["runtime_identity"],
            execution_profile="single_node"))
        extra = dict(reward=[0.25] * len(validation), mopd_sample_id=[r["sample_id"] for r in validation],
                     mopd_domain=[r["domain"] for r in validation], mopd_source=[r["source"] for r in validation])
        row = dict(status="PASS", step=0, identity=m["identity"], arm=arm, split="validation",
                   test_used_for_selection=False, metrics=source_metrics(extra), at_utc="2026-11-18T00:00:00+00:00")
        write(attempt / "validation_0.json", row)
        generations = attempt / "validation_generations/0.jsonl"
        generations.parent.mkdir()
        generations.write_text("".join(json.dumps(dict(step=0, score=0.25, **{k: v[i] for k, v in extra.items()})) + "\n"
                                       for i in range(len(validation))))
        # Qualification step-4/8 metrics are evidence only, never formal candidates.
        write(attempt / "validation_4.json", dict(row, step=4))
        write(attempt / "validation_8.json", dict(row, step=8))
        formal = tmp_path / "runs" / arm
        formal.mkdir(parents=True)
        if arm == "domain_mix":
            checkpoint(formal, m, arm, 5, sampler)
            write(formal / "validation/step_0.json", dict(row, **checkpoints.evaluation_reference(None, m["identity"], arm, 0, None)))
        job = dict(job_id=f"{100+n}.pbs1", folder=str(formal / "attempts" / "current"), job_name="0390_mp_t" + arm)
        write(Path(job["folder"]) / "request.json", dict(identity=m["identity"], arm=arm, phase="train"))
        states["train:" + arm] = dict(phase="train", arm=arm, status="Q", job=job, pending=None)
        states["qualification:" + arm] = dict(phase="qualification", arm=arm, status="DONE",
                                              job=dict(job_id=f"{200+n}.pbs1", folder=str(attempt), job_name="qual"), pending=None)
        old_states["train:" + arm] = dict(phase="train", arm=arm, status="R",
                                         job=dict(job, job_id=f"{90+n}.pbs1"), pending=None)
        jobs[arm] = job["job_id"]
        originals[arm] = {str(p.relative_to(run)): file_sha(p) for p in run.rglob("*") if p.is_file()}
    write(control / "states.json", states)
    write(tmp_path / "states.json", old_states)
    plan = adopt.make_plan(m, "adopt-step8", jobs, dict(revision="qualified-operator-commit", script_sha256="sha"))
    pbs = SimpleNamespace(observe=lambda job: dict(job_state="F", Exit_status=271))
    return SimpleNamespace(m=m, root=tmp_path, control=control, plan=plan, pbs=pbs, originals=originals, jobs=jobs)


def test_prepare_copies_full_state_and_reuses_verified_files(case, monkeypatch):
    calls = []
    copy = adopt.shutil.copy2
    monkeypatch.setattr(adopt.shutil, "copy2", lambda source, dest: (calls.append(str(source)), copy(source, dest))[1])
    before = {arm: adopt.tree(case.root / "runs" / arm) for arm in ARMS}
    assert adopt.prepare(case.m, case.plan)["status"] == "PREPARED"
    copied = len(calls)
    assert copied > 0
    adopt.prepare(case.m, case.plan)
    assert len(calls) == copied
    for arm in ARMS:
        item = case.plan["arms"][arm]
        assert adopt.tree(Path(item["staging"]) / "checkpoints/global_step_8") == item["source_files"]
        assert adopt.tree(case.root / "runs" / arm) == before[arm]
        assert list((Path(item["staging"]) / "validation").iterdir()) == [Path(item["staging"]) / "validation/step_0.json"]


def test_prepare_refuses_corrupt_existing_copy_without_overwriting(case):
    adopt.prepare(case.m, case.plan)
    item = case.plan["arms"]["domain_mix"]
    target = Path(item["staging"]) / "checkpoints/global_step_8/data.pt"
    target.write_bytes(b"corrupt-staged-loader")
    with pytest.raises(RuntimeError, match="Existing staging file changed"):
        adopt.prepare(case.m, case.plan)
    assert target.read_bytes() == b"corrupt-staged-loader"


def test_prepare_rejects_staging_symlink_before_copy(case):
    outside = case.root / "unrelated"
    outside.mkdir()
    stage = Path(case.plan["arms"]["domain_mix"]["staging"])
    stage.parent.mkdir(parents=True)
    stage.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Symlink"):
        adopt.prepare(case.m, case.plan)
    assert not list(outside.iterdir())


def test_prepare_recovers_copy_published_before_ledger_write(case, monkeypatch):
    original = adopt.write
    crashed = False
    def interrupted(path, payload):
        nonlocal crashed
        if Path(path).name == "copied_files.json" and not crashed:
            crashed = True
            raise RuntimeError("disconnect before copy ledger")
        original(path, payload)
    monkeypatch.setattr(adopt, "write", interrupted)
    with pytest.raises(RuntimeError, match="disconnect before copy ledger"):
        adopt.prepare(case.m, case.plan)
    published = [p for p in Path(case.plan["folder"]).glob("staging/**/ckpt_contents.json")]
    assert len(published) == 1
    identity = published[0].stat().st_ino
    monkeypatch.setattr(adopt, "write", original)
    adopt.prepare(case.m, case.plan)
    assert published[0].stat().st_ino == identity


@pytest.mark.parametrize("change", ["identity", "arm", "sampler", "inventory", "missing_optimizer", "baseline_count"])
def test_plan_rejects_incompatible_or_incomplete_source(case, change):
    item = case.plan["arms"]["domain_mix"]
    source = Path(item["source"])
    marker = source / "mopd_complete.json"
    proof = read(marker)
    if change in ("identity", "arm", "sampler"):
        proof[{"identity": "identity", "arm": "arm", "sampler": "sampler_identity"}[change]] = "other"
        write(marker, proof)
    elif change == "inventory":
        (source / "data.pt").write_bytes(b"changed")
    elif change == "missing_optimizer":
        (source / "actor/optimizer/shard.bin").unlink()
    else:
        path = Path(item["baseline"]["generations"])
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        adopt.make_plan(case.m, "new-plan", case.jobs, {})


def test_activate_preserves_history_and_resumes_only_step8_with_base_selection(case):
    adopt.prepare(case.m, case.plan)
    result = adopt.activate(case.m, case.plan["folder"], case.pbs)
    assert result["status"] == "ACTIVATED"
    before = case.plan["controller_states"][str(case.control)]
    states = read(case.control / "states.json")
    for arm in ARMS:
        item = case.plan["arms"][arm]
        run = Path(item["destination"])
        assert adopt.tree(item["archive"]) == item["old_files"]
        assert states["qualification:" + arm] == before["qualification:" + arm]
        assert states["train:" + arm]["job"] is None and states["train:" + arm]["status"] == "READY"
        assert inspect_resume(run, case.m, arm, item["sampler_identity"])["step"] == 8
        assert select_best(run, case.m["identity"], arm)["step"] == 0
        assert checkpoints.prune(run, case.m["identity"], arm, item["sampler_identity"], 2, 200)["retained_steps"] == [8]
        source_run = Path(item["source"]).parents[1]
        assert {str(p.relative_to(source_run)): file_sha(p) for p in source_run.rglob("*")
                if p.is_file() and p.name != "writer.lock"} == case.originals[arm]
    assert (case.control / "STOP").is_file()


@pytest.mark.parametrize("failure", ["running", "pending", "guardian", "writer", "new_writer", "receipt", "stop"])
def test_activate_requires_quiescent_writers_and_terminal_jobs(case, failure):
    adopt.prepare(case.m, case.plan)
    before = {arm: adopt.tree(case.root / "runs" / arm) for arm in ARMS}
    if failure == "pending":
        s = read(case.control / "states.json")
        s["train:domain_mix"]["pending"] = {"ambiguous": True}
        write(case.control / "states.json", s)
    if failure == "receipt":
        write(case.root / "runs/domain_mix/receipt.json", {"status": "PASS"})
    if failure == "stop":
        (case.control / "STOP").unlink()
    if failure == "running":
        case.pbs.observe = lambda job: dict(job_state="R")
    lease = {"guardian": case.control / "guardian.lock", "writer": case.root / "runs/domain_mix/writer.lock",
             "new_writer": Path(case.plan["arms"]["domain_mix"]["staging"]) / "writer.lock"}.get(failure)
    from contextlib import ExitStack
    with ExitStack() as stack:
        if lease:
            stack.enter_context(locked(lease))
        with pytest.raises((RuntimeError, BlockingIOError)):
            adopt.activate(case.m, case.plan["folder"], case.pbs)
    for arm in ARMS:
        assert not Path(case.plan["arms"][arm]["archive"]).exists()
        if failure != "receipt":
            assert adopt.tree(case.root / "runs" / arm) == before[arm]


@pytest.mark.parametrize("crash_after", [1, 2, 3, 4])
def test_activate_recovers_after_each_atomic_directory_move(case, monkeypatch, crash_after):
    adopt.prepare(case.m, case.plan)
    rename = adopt.rename_new
    count = 0
    def interrupted(source, destination):
        nonlocal count
        rename(source, destination)
        count += 1
        assert active(Path(destination) / "writer.lock")
        assert read(case.control / "states.json") == case.plan["controller_states"][str(case.control)]
        if count == crash_after:
            raise RuntimeError("simulated disconnect")
    monkeypatch.setattr(adopt, "rename_new", interrupted)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        adopt.activate(case.m, case.plan["folder"], case.pbs)
    monkeypatch.setattr(adopt, "rename_new", rename)
    result = adopt.activate(case.m, case.plan["folder"], case.pbs)
    assert result["status"] == "ACTIVATED"
    assert adopt.activate(case.m, case.plan["folder"], case.pbs)["status"] == "ACTIVATED"


def test_activate_recovers_states_published_before_final_journal(case, monkeypatch):
    adopt.prepare(case.m, case.plan)
    original = adopt.write
    def interrupted(path, payload):
        original(path, payload)
        if Path(path) == case.control / "states.json":
            for arm in ARMS:
                assert active(case.root / "runs" / arm / "writer.lock")
                assert active(Path(case.plan["arms"][arm]["archive"]) / "writer.lock")
            raise RuntimeError("disconnect after new states")
    monkeypatch.setattr(adopt, "write", interrupted)
    with pytest.raises(RuntimeError, match="disconnect after new states"):
        adopt.activate(case.m, case.plan["folder"], case.pbs)
    assert read(Path(case.plan["folder"]) / "activation.json")["status"] == "ACTIVATING"
    monkeypatch.setattr(adopt, "write", original)
    assert adopt.activate(case.m, case.plan["folder"], case.pbs)["status"] == "ACTIVATED"
