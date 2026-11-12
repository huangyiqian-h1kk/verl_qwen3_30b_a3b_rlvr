"""Audited signal-recovery revision for qualifications blocked before updates."""
import hashlib
from pathlib import Path
import shutil
import sys
import uuid

from .assets import verify
from .cuda_repair import NEW_ENTRY
from .job_exit_v2 import failure_result, retry_assessment
from .launcher_repair import replace_once
from .metadata_repair import OLD_IDS, revision_plan as metadata_plan
from d4_teachers.common import digest, files, lock, need, now, read, write
from d4_training_continuation.common import locked


SCHEMA = "d4-mopd-signal-recovery-v4"
OLD_FAILURE = '''            write(folder / "result.json", dict(status="FAIL", identity=m["identity"], error=str(exc),
                                                traceback=traceback.format_exc(), at_utc=now()))'''
NEW_FAILURE = '''            write(folder / "result.json", failure_record(exc, m["identity"]))'''
OLD_RETRY = '''                if reason and retry_allowed(row, result, job, m["identity"]):'''
NEW_RETRY = '''                assessment = retry_assessment(row, result, job, m["identity"])
                s["retry_assessment"] = dict(assessment, pbs_reason=reason, job_id=job["job_id"])
                if reason and assessment["allowed"]:'''


def revision_plan(m):
    # Keep earlier content-addressed revisions reproducible. This revision also
    # includes the already-reviewed sample-ID correction for the companion arm.
    changes, _, _, _ = metadata_plan(m)
    ctl = replace_once(changes["d4_mopd/control.py"],
        "from .job_exit import entry_script, failure_result, retry_allowed",
        "from .job_exit_v2 import entry_script, failure_result, retry_assessment")
    ctl = replace_once(ctl, NEW_ENTRY,
        '    (folder / "entry.py").write_text(entry_script(m, runtime=Path(__file__).resolve().parents[1]))\n')
    changes["d4_mopd/control.py"] = replace_once(ctl, OLD_RETRY, NEW_RETRY)
    cli = replace_once(changes["d4_mopd/cli.py"], "from . import control\n",
                      "from . import control\nfrom .job_exit_v2 import failure_record\n")
    changes["d4_mopd/cli.py"] = replace_once(cli, OLD_FAILURE, NEW_FAILURE)
    changes["d4_mopd/job_exit_v2.py"] = Path(__file__).with_name("job_exit_v2.py").read_text()
    inventory = dict(m["runtime_files"])
    inventory.update({name: hashlib.sha256(text.encode()).hexdigest() for name, text in changes.items()})
    revision = digest(dict(schema=SCHEMA, identity=m["identity"], runtime_files=inventory))
    folder = Path(m["manifest_path"]).parent / "launcher_revisions" / revision
    return changes, inventory, revision, folder


def failure_evidence(m, state, observations):
    evidence = {}
    for key, s in state.items():
        run = Path(m["manifest_path"]).parent / "qualification" / s["arm"]
        need(not any((run / "attempts").glob("*/update_*.json")),
             "Completed optimizer update exists; signal migration requires review of training progress")
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
        assessment = retry_assessment(observations[key], result, s["job"], m["identity"])
        if assessment["allowed"]:
            evidence[key] = assessment["reason"]
        else:
            trace = result.get("traceback", "")
            need(isinstance(trace, str) and trace.rstrip().endswith("KeyError: 'index'")
                 and OLD_IDS.strip() in trace and "in check_batch" in trace and "in _update_actor" in trace
                 and str(Path(m["manifest_path"]).parent / "runtime/d4_mopd/integration.py") in trace
                 and resume is not None,
                 "Unrelated qualification failure: " + assessment["reason"] + "; inspect " + str(attempt))
            evidence[key] = "top_level_index_removed_before_first_update"
        for prefix in ("cuda_runtime", "gpu_node"):
            for rank in (0, 1):
                proof = read(attempt / f"{prefix}_{rank}.json")
                need(proof.get("status") == "PASS" and proof.get("identity") == m["identity"]
                     and proof.get("rank") == rank, "Missing current per-node preflight proof")
    need(any(v.startswith("pbs_root_sigterm_") for v in evidence.values()),
         "No qualification shows the targeted PBS/Ray signal exit")
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
            need(files(stage / "runtime") == inventory, "Unexpected signal recovery revision changes")
            receipt = dict(status="PASS", schema=SCHEMA, identity=m["identity"], revision=revision,
                runtime_files=inventory, changed_files=sorted(changes), original_runtime_files=m["runtime_files"],
                previous_pointer=previous, previous_runtime=str(selected_before), before=state, pbs=observations,
                failure_evidence=evidence, at_utc=now(),
                scope="Early driver exit evidence and signal recovery; includes prior batch metadata correction; recipe unchanged")
            write(stage / "receipt.json", receipt)
            if folder.exists():
                old = read(folder / "receipt.json")
                need(old.get("status") == "PASS" and old.get("identity") == m["identity"]
                     and old.get("revision") == revision and old.get("runtime_files") == inventory
                     and files(folder / "runtime") == inventory, "Conflicting signal recovery revision")
            else:
                stage.rename(folder)
            write(root / "cpu_preflight.json", dict(status="INVALIDATED", reason="Signal recovery revision", at_utc=now()))
            write(pointer, dict(schema=SCHEMA, identity=m["identity"], revision=revision, folder=str(folder)))
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        selected = active_runtime(m)
        print("Signal recovery revision: " + str(selected), flush=True)
        return dict(status="PASS", identity=m["identity"], revision=revision, runtime=str(selected),
                    changed_files=sorted(changes), failure_evidence=evidence, gpu_execution_tested=False, at_utc=now())
