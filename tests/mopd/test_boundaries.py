"""Integration boundaries most likely to silently change the experiment."""
import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tensordict import TensorDict

from d4_mopd import DOMAINS
from d4_mopd.config import compose
from d4_mopd.integration import check_batch
from d4_mopd.metrics import source_metrics, SELECTION_KEY
from d4_teachers.common import write
from verl import DataProto
from verl.trainer.ppo.utils import create_rl_dataset


def test_native_dataset_factory_preserves_prompt_and_domain(manifest, tmp_path):
    rows = [dict(domain=d, sample_id=d, source="frozen_source", messages_json='[{"role":"user","content":"x"}]',
                 tools_json="[]", verifier_json="{}", prompt_tokens=12, prompt_token_hash="hash",
                 generation_max_tokens=65524, mopd_split="train") for d in DOMAINS]
    pq.write_table(pa.Table.from_pylist(rows), manifest["data"]["train"][0])
    config = compose(manifest, "domain_mix", tmp_path, qualification=True)
    dataset = create_rl_dataset(config.data.train_files, config.data, None, None)
    assert [dataset[i]["data_source"] for i in range(4)] == list(DOMAINS)
    assert dataset[0]["raw_prompt"] == [{"role":"user", "content":"x"}]
    assert dataset[0]["extra_info"]["generation_max_tokens"] == 256
    assert dataset[0]["extra_info"]["prompt_token_hash"] == "hash"
    assert dataset[0]["mopd_split"] == "train"


def make_batch():
    # Prompt width 2, response width 3; teacher IDs already left-shifted by verl.
    response = torch.tensor([[7,8,0]] * 4)
    td = TensorDict({"prompts": torch.ones(4,2,dtype=torch.long), "responses": response,
                    "response_mask": torch.tensor([[1,1,0]]*4),
                    "teacher_ids": torch.tensor([[[6],[7],[8],[0],[0]]]*4),
                    "teacher_logprobs": -torch.ones(4,5,1), "old_log_probs": -torch.ones(4,3)*2,
                    "token_level_rewards": torch.zeros(4,3)}, batch_size=[4])
    return DataProto(batch=td, non_tensor_batch={"data_source":np.array(DOMAINS),
        "extra_info":np.array([{"sample_id":d} for d in DOMAINS], dtype=object)})


def test_actual_padding_alignment_and_batch_balance_reorder(spec):
    batch = make_batch()
    batch.reorder(torch.tensor([3,2,0,1]))
    result = check_batch(batch, dict(spec, train_batch_size=4), "domain_mix", 1, list(DOMAINS))
    assert result["domains"]["math"]["mean_teacher_minus_student"] == 1


@pytest.mark.parametrize("corruption", ["alignment", "nan", "reward", "empty", "ids", "domain"])
def test_refuse_silent_experiment_corruption(spec, corruption):
    batch = make_batch()
    if corruption == "alignment": batch.batch["teacher_ids"][0,1,0] = 42
    if corruption == "nan": batch.batch["teacher_logprobs"][0,1,0] = float("nan")
    if corruption == "reward": batch.batch["token_level_rewards"][0,0] = 1
    if corruption == "empty": batch.batch["response_mask"][0,:] = 0
    if corruption == "ids": batch.non_tensor_batch["extra_info"][0]["sample_id"] = "bad"
    if corruption == "domain": batch.non_tensor_batch["data_source"][0] = "if"
    with pytest.raises(RuntimeError):
        check_batch(batch, dict(spec, train_batch_size=4), "domain_mix", 1, list(DOMAINS))


def test_task_scorer_is_not_invoked_by_training(monkeypatch):
    from d4_mopd.agent_loop import MOPDAgentLoop, MultiDomainAgentLoop
    async def generate(*args, **kwargs):
        return SimpleNamespace(response_ids=[1], reward_score=None, extra_fields={})
    monkeypatch.setattr(MultiDomainAgentLoop, "run", generate)
    loop = object.__new__(MOPDAgentLoop)
    assert asyncio.run(loop.run({}, mopd_split="train")).reward_score == 0
    assert asyncio.run(loop.run({}, mopd_split="validation")).reward_score is None
    from d4_mopd.reward import compute_score
    with pytest.raises(RuntimeError, match="bypass"):
        asyncio.run(compute_score("math", "answer", "truth", extra_info={"split":"train"}))


def test_validation_bridge_uses_isolated_subprocess_and_keeps_provenance(tmp_path):
    from d4_mopd.reward import compute_score
    package = tmp_path / "verifier/md_reward_roundoff_0390"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def compute_score(data_source, solution_str, ground_truth, extra_info):\n"
        "    assert extra_info['num_turns'] == 2\n"
        "    assert extra_info['raw_completion'] == '<tool_call>original</tool_call>'\n"
        "    return {'score': 0.75, 'acc': 0.75}\n")
    path = tmp_path / "manifest.json"
    write(path, dict(manifest_path=str(path), verifier_python=sys.executable))
    result = asyncio.run(compute_score("math", "decoded", "truth", manifest_path=str(path),
        extra_info={"split":"validation", "num_turns":np.int64(2), "source":"deepmath_l8",
                    "sample_id":"sample1", "raw_completion":"<tool_call>original</tool_call>"}))
    assert result == dict(score=.75, acc=.75, mopd_source="deepmath_l8", mopd_domain="math", mopd_sample_id="sample1")


def test_math_validation_family_mean_is_not_row_weighted():
    extra = {"reward":[0.,1.,1.,1.], "mopd_source":["nemotron", "deepmath_l8","deepmath_l8","deepmath_l9"],
             "mopd_domain":["math"]*4, "mopd_sample_id":["a","b","c","d"]}
    out = source_metrics(extra)
    assert out["val-source/math/equal_source_mean"] == .5
    assert out["val-source/math/deepmath/n"] == 3
    extra["mopd_sample_id"][-1] = "a"
    with pytest.raises(RuntimeError, match="duplicate"):
        source_metrics(extra)


def validation_candidates():
    groups = [("math", "math", 238), ("math", "deepmath_l8", 213), ("math", "deepmath_l9", 43)]
    groups += [(d, d, 256) for d in DOMAINS if d != "math"]
    return [dict(domain=d, source=s, sample_id=f"{s}:{i}", mopd_split="validation")
            for d, s, count in groups for i in range(count)]


def test_validation_subset_matches_mix_rl_scale_and_preserves_sources(spec, tmp_path):
    from collections import Counter
    from d4_mopd.assets import select_validation
    candidates = validation_candidates()
    path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(candidates), path)
    selected = select_validation(pq.read_table(path).to_pylist(), spec)
    assert len(candidates) == 1262 and len(selected) == 1024
    assert Counter(r["domain"] for r in selected) == {d: 256 for d in DOMAINS}
    math = [r for r in selected if r["domain"] == "math"]
    assert Counter(r["source"] for r in math) == {"math": 128, "deepmath_l8": 107, "deepmath_l9": 21}
    assert {r["sample_id"] for r in selected} <= {r["sample_id"] for r in candidates}
    # Keep every existing IF/CP/Logic item and its order; no new train/test samples.
    assert [r for r in selected if r["domain"] != "math"] == [r for r in candidates if r["domain"] != "math"]
    assert pq.read_table(path).to_pylist() == candidates
    assert selected == select_validation(candidates, spec)
    assert {r["sample_id"] for r in selected} == {r["sample_id"] for r in select_validation(candidates[::-1], spec)}
    assert math != [r for r in select_validation(candidates, dict(spec, seed=43)) if r["domain"] == "math"]


@pytest.mark.parametrize("corruption", ["test", "duplicate", "insufficient_nemotron"])
def test_validation_selection_rejects_split_leakage_duplicate_and_shortfall(spec, corruption):
    from d4_mopd.assets import select_validation
    rows = validation_candidates()
    if corruption == "test":
        rows[0]["mopd_split"] = "test"
    elif corruption == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows = rows[111:]  # Leave only 127 Nemotron validation candidates.
    with pytest.raises(RuntimeError, match="validation"):
        select_validation(rows, spec)


def test_selection_score_has_four_domains_and_equal_math_sources():
    extra = dict(reward=[0.,1.,1.,1.,.25,.25,.75,.5],
                 mopd_source=["nemotron","deepmath_l8","deepmath_l8","deepmath_l9","if","if","cp","logic"],
                 mopd_domain=["math"]*4+["if","if","conversational_pivot","logic_algorithmic"],
                 mopd_sample_id=list("abcdefgh"))
    metrics = source_metrics(extra)
    assert metrics[SELECTION_KEY] == .5
    assert metrics["val-selection/math"] == .5
    assert metrics[SELECTION_KEY] != sum(extra["reward"]) / len(extra["reward"])
    for key in extra:
        extra[key].pop()
    assert SELECTION_KEY not in source_metrics(extra)


def test_formal_start_requires_arm_specific_gpu_resume_proof(manifest, tmp_path, monkeypatch):
    from d4_mopd import control
    monkeypatch.setattr(control, "verify", lambda p: manifest)
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], native_abi={"status":"PASS"}))
    q = tmp_path / "qualification/domain_mix/receipt.json"
    write(q, dict(status="PASS", identity=manifest["identity"], resumed_from=0))
    with pytest.raises(RuntimeError, match="qualification"):
        control.start(manifest["manifest_path"], "train", ["domain_mix"])
    assert not (tmp_path / "states.json").exists()


def test_glibc_audit_uses_needed_versions_not_definitions():
    from d4_mopd.preflight import glibc_requirements
    text = ("Version definition section '.gnu.version_d' contains 1 entry:\n"
            "  Name: GLIBC_2.99\n"
            "Version needs section '.gnu.version_r' contains 3 entries:\n"
            "  Name: GLIBC_2.9 Flags: none\n  Name: GLIBC_2.34 Flags: none\n"
            "  Name: GLIBCXX_3.4.32\n  Name: GLIBC_PRIVATE\n")
    assert glibc_requirements(text) == [(2, 9), (2, 34)]
    assert glibc_requirements("No version information found in this file.") == []


@pytest.mark.parametrize("host", ["glibc 2.34", "glibc 2.40"])
def test_abi_audit_catches_the_uploaded_failure_before_gpu_import(tmp_path, monkeypatch, host):
    from pathlib import Path
    from d4_mopd import preflight
    paths = {}
    for name in ("transformer-engine", "apex", "flash-attn"):
        path = tmp_path / name / "library.so"
        path.parent.mkdir()
        path.write_bytes(b"\x7fELFsynthetic")
        paths[name] = path
    def dist(name):
        return SimpleNamespace(version="2.20.2", files=[paths[name]], locate_file=lambda f:f)
    monkeypatch.setattr(preflight.importlib.metadata, "distribution", dist)
    monkeypatch.setattr(preflight.os, "confstr", lambda _:host)
    def readelf(args, **kw):
        version = "2.38" if Path(args[-1]) == paths["transformer-engine"] else "2.28"
        return SimpleNamespace(stdout="Version needs section '.gnu.version_r':\n  Name: GLIBC_"+version)
    monkeypatch.setattr(preflight.subprocess, "run", readelf)
    audit = preflight.native_abi()
    assert audit["target_glibc"] == "2.34" and len(audit["incompatible"]) == 1
    with pytest.raises(RuntimeError, match="GLIBC_2.38"):
        preflight.require_native_abi(audit)


def test_failed_repeat_cpu_preflight_invalidates_old_pass(tmp_path, monkeypatch):
    from d4_mopd import preflight
    from d4_teachers.common import read
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity="old"))
    def fail(_): raise OSError("GLIBC_2.38 not found")
    monkeypatch.setattr(preflight, "_cpu", fail)
    with pytest.raises(OSError): preflight.cpu(tmp_path / "manifest.json")
    assert read(tmp_path / "cpu_preflight.json")["status"] == "FAIL"


def test_gpu_import_error_is_persisted(tmp_path, monkeypatch):
    from d4_mopd import preflight
    from d4_teachers.common import read
    def fail(*args): raise OSError("GLIBC_2.38 not found")
    monkeypatch.setattr(preflight, "_gpu_node", fail)
    with pytest.raises(OSError): preflight.gpu_node(tmp_path / "manifest.json", tmp_path, 1)
    result = read(tmp_path / "gpu_node_1.json")
    assert result["status"] == "FAIL" and "GLIBC_2.38" in result["traceback"]


def test_node_shell_records_failure_before_ray(tmp_path):
    from pathlib import Path
    import os
    import subprocess
    folder = tmp_path / "attempt"
    folder.mkdir()
    fake = tmp_path / "python"
    fake.write_text('#!/bin/bash\nif [[ "$3" == paths ]]; then echo /unused; exit 0; fi\nexit 17\n')
    fake.chmod(0o755)
    (folder / "python.txt").write_text(str(fake))
    script = Path(__file__).resolve().parents[2] / "d4_mopd/node.sh"
    p = subprocess.run(["bash", str(script), str(folder / "request.json")],
                       env={**os.environ, "OMPI_COMM_WORLD_RANK":"0", "PBS_LOCALDIR":str(tmp_path)},
                       capture_output=True, text=True, timeout=15)
    assert p.returncode == 17, p.stderr
    assert (folder / "finished").read_text().strip() == "17"
    assert not list(folder.glob("ray_*.log"))
