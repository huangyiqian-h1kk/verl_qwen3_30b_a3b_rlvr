"""Freeze teacher identities, disjoint data, code and the existing scoring environment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from . import DOMAINS, TEACHERS, VERL_REVISION
from .spec import load_spec
from d4_teachers.common import digest, file_sha, files, model_inventory, need, read, verify_files, write

REPO = Path(__file__).resolve().parents[1]


def check_verl(path):
    path = Path(path).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    need(git("rev-parse", "HEAD") == VERL_REVISION, "Use the pinned upstream verl revision: " + VERL_REVISION)
    need(not git("status", "--porcelain"), "Pinned verl has modified/untracked source files")
    return str(path)


def prepare(campaign, destination, verl_src, python, spec_path=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from d4_teachers.source import load_campaign
    from d4_teachers.exporting import validate_export
    from d4_teachers import deepmath_data, math_test_repair
    from d4_training_continuation.common import locked

    root, campaign = Path(destination).resolve(), Path(campaign).resolve()
    spec = load_spec(spec_path)
    with locked(root.parent / (root.name + ".prepare.lock")):
        if (root / "manifest.json").exists():
            m = verify(root / "manifest.json")
            need(m["teacher_campaign"] == str(campaign) and m["spec"] == spec
                 and m["verl_src"] == str(Path(verl_src).resolve())
                 and m["python"] == str(Path(python).absolute()), "Existing experiment has another recipe/path")
            return m
        need(not root.exists(), "Incomplete destination exists; choose a new experiment directory")
        c = load_campaign(campaign)
        check_verl(verl_src)
        need(Path(python).is_file() and os.access(python, os.X_OK), "New verl Python is unavailable")
        teachers = {}
        for domain, (task, step) in TEACHERS.items():
            e = validate_export(read(campaign / "runs" / task / "export.json"))
            need(e["selected"]["step"] == step and e["selected"]["domain"] == domain,
                 "Wrong frozen teacher: " + domain)
            teachers[domain] = dict(task=task, step=step, path=e["path"],
                                    export=e, weights_identity=e["weights_identity"])
        student = dict(path=c["contract"]["base_model"], files=c["contract"]["base_inventory"])
        need(model_inventory(student["path"]) == student["files"], "Base model changed")
        deep = deepmath_data.verify(c)
        revision = math_test_repair.active_revision(c)
        need(revision is not None and deep["verifier_identity"] == revision["identity"],
             "DeepMath and active corrected verifier must have the same identity")
        stage = root.parent / ("." + root.name + ".prepare_" + uuid.uuid4().hex)
        stage.mkdir()
        try:
            data, summaries, rows_by_split = {}, {}, {}
            required = ("domain", "sample_id", "split_group_id", "source", "messages_json", "tools_json",
                        "verifier_json", "prompt_tokens", "prompt_token_hash", "generation_max_tokens")
            for split in ("train", "validation", "test"):
                rows = [r for d in DOMAINS for r in pq.read_table(campaign / "data" / d / (split + ".parquet")).to_pylist()]
                if split != "train":
                    rows += pq.read_table(deepmath_data.directory(c) / (split + ".parquet")).to_pylist()
                rows = [{**{k: r[k] for k in required}, "category": r.get("category", ""),
                         "task": r.get("task", ""), "mopd_split": split} for r in rows]
                need(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate sample IDs in " + split)
                need(all(r["domain"] in DOMAINS and 0 < r["prompt_tokens"] < 65536
                         and 0 < r["generation_max_tokens"] <= 65536-r["prompt_tokens"] for r in rows),
                     "Invalid domain or frozen context budget")
                path = stage / "data" / (split + ".parquet")
                path.parent.mkdir(exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows), path)
                data[split] = [str(root / "data" / path.name)]
                summaries[split] = {"rows": len(rows), "domains": {d: sum(r["domain"] == d for r in rows) for d in DOMAINS},
                                    "sha256": file_sha(path), "ids_sha256": digest([r["sample_id"] for r in rows])}
                rows_by_split[split] = rows
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
                for key in ("sample_id", "split_group_id", "prompt_token_hash"):
                    need(not {r[key] for r in rows_by_split[a]} & {r[key] for r in rows_by_split[b]},
                         f"Split overlap: {a}/{b}/{key}")
            # Copy a small immutable runtime. Model files remain referenced by a validated
            # HF inventory: large shard sizes/mtimes + small-file SHA, NOT a full weight SHA.
            runtime = stage / "runtime"
            for name in ("d4_mopd", "multidomain", "d4_teachers", "d4_training_continuation"):
                shutil.copytree(REPO / name, runtime / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copytree(REPO / "config/mopd", runtime / "config/mopd")
            verifier = stage / "verifier"
            for name in c["contract"]["template_files"]:
                src, dest = campaign / "template" / name, verifier / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            shutil.rmtree(verifier / "runtime/d4_db_mathfix")
            shutil.copytree(math_test_repair.revision_directory(c, revision["identity"]) / "runtime/d4_db_mathfix",
                            verifier / "runtime/d4_db_mathfix", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            entry = verifier / "md_reward_roundoff_0390/__init__.py"
            before = repr(str(Path(c["contract"]["source_job"]) / "math_diagnostics"))
            need(entry.read_text().count(before) == 1, "Unknown scoring diagnostics binding")
            entry.write_text(entry.read_text().replace(before, repr(str(root / "math_diagnostics"))))
            payload = dict(schema="d4-mopd-manifest-v1", teacher_campaign=str(campaign),
                           teacher_campaign_identity=c["identity"], spec=spec,
                           verl_src=str(Path(verl_src).resolve()), verl_revision=VERL_REVISION,
                           python=str(Path(python).absolute()), student=student, teachers=teachers,
                           data=data, data_summary=summaries,
                           max_prompt_tokens=max(r["prompt_tokens"] for rows in rows_by_split.values() for r in rows),
                           runtime_files=files(runtime), verifier_files=files(verifier),
                           verifier_python=str(Path(c["contract"]["overlay"]) / "bin/python"),
                           verifier_identity=revision["identity"],
                           manifest_path=str(root / "manifest.json"))
            manifest = dict(payload, identity=digest(payload))
            write(stage / "manifest.json", manifest)
            stage.rename(root)
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def verify(path, *, models=True):
    path = Path(path).resolve()
    m = read(path)
    need(m["manifest_path"] == str(path) and m["identity"] == digest({k: v for k, v in m.items() if k != "identity"}),
         "Experiment manifest identity changed")
    check_verl(m["verl_src"])
    verify_files(path.parent / "runtime", m["runtime_files"])
    verify_files(path.parent / "verifier", m["verifier_files"])
    for split, paths in m["data"].items():
        need(len(paths) == 1 and file_sha(paths[0]) == m["data_summary"][split]["sha256"], "Data changed: " + split)
    if models:
        from d4_teachers.exporting import validate_export
        need(model_inventory(m["student"]["path"]) == m["student"]["files"], "Student base changed")
        for t in m["teachers"].values():
            validate_export(t["export"])
    return m
