"""Audited CUDA launcher revision for qualifications that failed before updates."""
import hashlib
from pathlib import Path
import shutil
import sys
import uuid

from .assets import verify
from .job_exit import ray_exit_trace, root_sigterm
from .launcher_repair import patched_sources, replace_once
from d4_teachers.common import digest, files, lock, need, now, read, write
from d4_training_continuation.common import locked


SCHEMA = "d4-mopd-launcher-cuda-v2"
CUDA_SOURCE = 'source "$(dirname "${BASH_SOURCE[0]}")/cuda_runtime.sh"\n'
OLD_ENTRY = '''    (folder / "entry.py").write_text("import sys\\nsys.path[:0]=" + repr([
        str(Path(m["manifest_path"]).parent / "runtime"), m["verl_src"]])
        + "\\nfrom d4_mopd.cli import main\\nmain()\\n")
'''
NEW_ENTRY = '''    (folder / "entry.py").write_text(entry_script(m, Path(__file__).with_name("job_exit.py"), folder / "request.json"))
'''


def revision_plan(m):
    changes = patched_sources(m)  # Includes the already-reviewed short Ray path.
    changes["d4_mopd/node.sh"] = replace_once(changes["d4_mopd/node.sh"],
        "unset PYTHONHOME\n", "unset PYTHONHOME\n" + CUDA_SOURCE)
    control = changes["d4_mopd/control.py"]
    control = replace_once(control, "from .assets import verify\n",
        "from .assets import verify\nfrom .job_exit import entry_script, failure_result, retry_allowed\n")
    control = replace_once(control, OLD_ENTRY, NEW_ENTRY)
    control = replace_once(control,
        '            result_path = Path(job["folder"]) / "result.json"\n'
        '            result = read(result_path) if result_path.exists() else {}\n',
        '            result = failure_result(job)\n')
    control = replace_once(control, 'if reason and result.get("status") != "FAIL":',
        'if reason and retry_allowed(row, result, job, m["identity"]):')
    changes["d4_mopd/control.py"] = control
    for name in ("cuda_runtime.sh", "cuda_runtime.py", "job_exit.py"):
        changes["d4_mopd/" + name] = Path(__file__).with_name(name).read_text()
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
             "An optimizer update already exists; this repair is for initialization failures only")
        attempt = Path(s["job"]["folder"])
        request, result = read(attempt / "request.json"), read(attempt / "result.json")
        need(request["identity"] == m["identity"] == result.get("identity")
             and request["arm"] == s["arm"] and request["phase"] == "qualification"
             and Path(request["folder"]).resolve() == attempt.resolve()
             and result.get("status") == "FAIL", "Unbound qualification failure")
        need(not (attempt / "resume.json").exists(), "Learner checkpoint loading already began")
        with (attempt / "pbs.log").open("rb") as stream:
            log = stream.read(32 * 1024 * 1024).decode(errors="replace")
        if ("RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist" in log
                and "Engine core initialization failed" in result.get("traceback", "")):
            evidence[key] = "missing_cuda_toolkit_during_vllm_initialization"
        else:
            # The old driver omitted Ray's original termination message. This
            # permits this explicit repair/retry only, never an automatic retry.
            need(root_sigterm(observations[key]) and ray_exit_trace(result),
                 "Unrelated qualification failure; inspect before applying CUDA repair")
            evidence[key] = "legacy_ray_exit_with_pbs_sigterm_explicit_retry_only"
    need("missing_cuda_toolkit_during_vllm_initialization" in evidence.values(),
         "No qualification shows the targeted missing-nvcc failure")
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
        selected_before = active_runtime(m)  # Validate the existing v1/v2 pointer before migration.
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
            need(files(stage / "runtime") == inventory, "Unexpected CUDA launcher changes")
            receipt = dict(status="PASS", schema=SCHEMA, identity=m["identity"], revision=revision,
                runtime_files=inventory, changed_files=sorted(changes), original_runtime_files=m["runtime_files"],
                previous_pointer=previous, previous_runtime=str(selected_before), before=state, pbs=observations,
                failure_evidence=evidence, at_utc=now(),
                scope="Per-node CUDA JIT tools, entry exit evidence and controller retry predicate; learner unchanged")
            write(stage / "receipt.json", receipt)
            if folder.exists():
                old = read(folder / "receipt.json")
                need(old.get("status") == "PASS" and old.get("identity") == m["identity"]
                     and old.get("revision") == revision and old.get("runtime_files") == inventory
                     and files(folder / "runtime") == inventory, "Incomplete/conflicting CUDA launcher revision")
            else:
                stage.rename(folder)
            write(root / "cpu_preflight.json", dict(status="INVALIDATED", reason="CUDA launcher revision", at_utc=now()))
            write(pointer, dict(schema=SCHEMA, identity=m["identity"], revision=revision, folder=str(folder)))
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        selected = active_runtime(m)
        print("CUDA launcher revision: " + str(selected), flush=True)
        return dict(status="PASS", identity=m["identity"], revision=revision, runtime=str(selected),
                    changed_files=sorted(changes), failure_evidence=evidence, gpu_execution_tested=False, at_utc=now())
