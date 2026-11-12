"""Audited adapter correction before any completed qualification update."""
import hashlib
from pathlib import Path
import shutil
import sys
import uuid

from .assets import verify
from .cuda_repair import revision_plan as cuda_plan
from .job_exit import failure_result, ray_exit_trace, root_sigterm
from .launcher_repair import replace_once
from d4_teachers.common import digest, files, lock, need, now, read, write
from d4_training_continuation.common import locked


SCHEMA = "d4-mopd-batch-metadata-v3"
OLD_IDS = '    ids = list(batch.non_tensor_batch["index"])\n'
NEW_IDS = '''    # verl moves top-level index into the generation request. extra_info stays
    # with the optimizer batch through rollout, repeat and token balancing.
    extra = batch.non_tensor_batch.get("extra_info", ())
    need(len(extra) == b and all(isinstance(v, dict) and isinstance(v.get("sample_id"), str)
                                and v["sample_id"] for v in extra), "Missing stable prompt IDs")
    ids = [v["sample_id"] for v in extra]
'''


def revision_plan(m):
    changes, _, _, _ = cuda_plan(m)  # Keep v1/v2 revisions reproducible and valid.
    original = Path(m["manifest_path"]).parent / "runtime"
    changes["d4_mopd/integration.py"] = replace_once(
        (original / "d4_mopd/integration.py").read_text(), OLD_IDS, NEW_IDS)
    # The driver AND Ray actors must import this checked snapshot. Pointing only
    # entry.py at the corrected code would leave Ray workers on the old adapter.
    changes["d4_mopd/cli.py"] = replace_once((original / "d4_mopd/cli.py").read_text(),
        'print(str(Path(r["manifest"]).parent / "runtime") + os.pathsep + m["verl_src"])',
        'print(str(Path(__file__).resolve().parents[1]) + os.pathsep + m["verl_src"])')
    changes["d4_mopd/job_exit.py"] = replace_once(changes["d4_mopd/job_exit.py"],
        'paths = [str(Path(m["manifest_path"]).parent / "runtime"), m["verl_src"]]',
        'paths = [str(Path(support).resolve().parents[1]), m["verl_src"]]')
    changes["d4_mopd/job_exit.py"] = replace_once(changes["d4_mopd/job_exit.py"],
        'All learner imports still resolve to the original manifest runtime.',
        'Driver and Ray imports resolve to the checked adapter revision.')
    changes["d4_mopd/batch_contract.py"] = Path(__file__).with_name("batch_contract.py").read_text()
    inventory = dict(m["runtime_files"])
    inventory.update({name: hashlib.sha256(text.encode()).hexdigest() for name, text in changes.items()})
    revision = digest(dict(schema=SCHEMA, identity=m["identity"], runtime_files=inventory))
    folder = original.parent / "launcher_revisions" / revision
    return changes, inventory, revision, folder


def failure_evidence(m, state, observations):
    evidence = {}
    for key, s in state.items():
        run = Path(m["manifest_path"]).parent / "qualification" / s["arm"]
        need(not any((run / "attempts").glob("*/update_*.json")),
             "A completed optimizer update exists; metadata repair requires review of training progress")
        attempt = Path(s["job"]["folder"])
        request, result = read(attempt / "request.json"), failure_result(s["job"])
        need(request["identity"] == m["identity"] == result.get("identity")
             and request["phase"] == "qualification" and request["arm"] == s["arm"]
             and Path(request["folder"]).resolve() == attempt.resolve()
             and result.get("status") == "FAIL", "Unbound qualification failure or failed node preflight")
        resume = read(attempt / "resume.json") if (attempt / "resume.json").exists() else None
        if resume is not None:
            need(resume.get("identity") == m["identity"] and resume.get("step") == 0
                 and resume.get("checkpoint") is None
                 and resume.get("model_optimizer_scheduler_rng_and_loader") is False,
                 "Qualification has resumed training progress")
        trace = result.get("traceback", "")
        if (isinstance(trace, str) and trace.rstrip().endswith("KeyError: 'index'")
                and OLD_IDS.strip() in trace and "in check_batch" in trace and "in _update_actor" in trace
                and str(Path(m["manifest_path"]).parent / "runtime/d4_mopd/integration.py") in trace):
            need(resume is not None, "Missing step-zero resume evidence for the failed batch check")
            evidence[key] = "top_level_index_removed_before_first_update"
        else:
            # This is a one-time explicit restart paired with the targeted fix.
            # It does not weaken the controller's automatic signal retry rule.
            need(root_sigterm(observations[key]) and ray_exit_trace(result),
                 "Unrelated qualification failure; inspect result.json and pbs.log: " + str(attempt))
            evidence[key] = "pbs_root_sigterm_explicit_retry_only"
        for prefix in ("cuda_runtime", "gpu_node"):
            for rank in (0, 1):
                proof = read(attempt / f"{prefix}_{rank}.json")
                need(proof.get("status") == "PASS" and proof.get("identity") == m["identity"]
                     and proof.get("rank") == rank, "Missing current per-node preflight proof")
    need("top_level_index_removed_before_first_update" in evidence.values(),
         "No qualification shows the targeted batch metadata failure")
    return evidence


def repair(manifest_path, *, lease_fd=None, on_pbs_unknown=None):
    from . import ARMS, control
    from .launcher_repair import active_runtime
    from .repair_pipeline import environment_lease
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    need(Path(sys.executable).absolute() == Path(m["python"]).absolute(), "Use the manifest interpreter")
    with environment_lease(m, lease_fd), lock(root / "launch.lock"), \
            locked(root / "guardian.lock"), lock(root / "states.lock"):
        need(not (root / "STOP").exists(), "Experiment is stopped")
        state = read(root / "states.json")
        need(set(state) == {"qualification:" + arm for arm in ARMS}, "Both qualifications are required")
        observations = control.quiescent_qualifications(m, state, control.PBS(), on_pbs_unknown=on_pbs_unknown)
        evidence = failure_evidence(m, state, observations)
        selected_before = active_runtime(m)
        changes, inventory, revision, folder = revision_plan(m)
        pointer = root / "launcher_repair.json"
        previous = read(pointer) if pointer.exists() else None
        stage = folder.parent / ("." + revision + "." + uuid.uuid4().hex)
        stage.mkdir(parents=True)
        try:
            shutil.copytree(root / "runtime", stage / "runtime",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name, text in changes.items():
                (stage / "runtime" / name).write_text(text)
            need(files(stage / "runtime") == inventory, "Unexpected batch metadata revision changes")
            receipt = dict(status="PASS", schema=SCHEMA, identity=m["identity"], revision=revision,
                runtime_files=inventory, changed_files=sorted(changes), original_runtime_files=m["runtime_files"],
                previous_pointer=previous, previous_runtime=str(selected_before), before=state, pbs=observations,
                failure_evidence=evidence, at_utc=now(),
                scope="Stable sample ID assertion and runtime import paths; loss, optimizer, sampling and datasets unchanged")
            write(stage / "receipt.json", receipt)
            if folder.exists():
                old = read(folder / "receipt.json")
                need(old.get("status") == "PASS" and old.get("identity") == m["identity"]
                     and old.get("revision") == revision and old.get("runtime_files") == inventory
                     and files(folder / "runtime") == inventory, "Conflicting batch metadata revision")
            else:
                stage.rename(folder)
            write(root / "cpu_preflight.json", dict(status="INVALIDATED", reason="Batch metadata adapter revision", at_utc=now()))
            write(pointer, dict(schema=SCHEMA, identity=m["identity"], revision=revision, folder=str(folder)))
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        selected = active_runtime(m)
        print("Batch metadata revision: " + str(selected), flush=True)
        return dict(status="PASS", identity=m["identity"], revision=revision, runtime=str(selected),
                    changed_files=sorted(changes), failure_evidence=evidence, gpu_execution_tested=False, at_utc=now())
