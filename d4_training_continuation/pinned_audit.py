"""Job-local checkpoint audit binding for both eight- and sixteen-rank phases.

Compile the contract-bound source bytes in a private package. Ordinary imports
may already refer to another installed d4_two_node_resumefix package. Never
reload that shared package or mutate a live optimizer to repair its audit.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import uuid

VERSION = "continuation-pinned-audit-v1"
POLICY = "fused-adam-empty-local-group-step-v1"
ENV_JOB = "CONT_JOB"
FILES = ("__init__.py", "core.py", "state_audit.py", "audit_offload.py")
_CACHE = {}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def modules():
    job = Path(os.environ[ENV_JOB]).resolve()
    req = read(job / "request.json")
    need(Path(req["job_dir"]).resolve() == job and req["expected_nodes"] in (1, 2),
         "Audit request is not this allocation")
    need(Path(__file__).resolve() == job / "runtime/d4_training_continuation/pinned_audit.py",
         "Audit binder was imported from another snapshot")
    bound = req["contract"]["snapshot_files"]
    need(sha(Path(__file__).read_bytes()) == bound["runtime/d4_training_continuation/pinned_audit.py"],
         "Audit binder differs from the candidate contract")
    directory = job / "audit_source"
    sources = {}
    for name in FILES:
        path = directory / name
        need(not path.is_symlink(), "Audit source cannot be a symlink")
        sources[name] = path.read_bytes()
        need(sha(sources[name]) == bound["audit_source/" + name],
             "Pinned audit source changed: " + name)
    key = sha((str(directory) + json.dumps({n: sha(b) for n, b in sources.items()},
                                         sort_keys=True)).encode())
    if key not in _CACHE:
        package = "_d4_continuation_checkpoint_audit_" + key
        loaded = {}
        try:
            for name, data in sources.items():
                module_name = package if name == "__init__.py" else package + "." + name[:-3]
                need(module_name not in sys.modules, "Audit private namespace already occupied")
                path = directory / name
                module = types.ModuleType(module_name)
                module.__file__ = str(path)
                module.__package__ = package
                module.__spec__ = importlib.util.spec_from_file_location(
                    module_name, path,
                    submodule_search_locations=[str(directory)] if name == "__init__.py" else None)
                if name == "__init__.py":
                    module.__path__ = [str(directory)]
                sys.modules[module_name] = module
                loaded[name] = module
                # Compile the validated bytes, bypassing stale pyc and path precedence.
                exec(compile(data, str(path), "exec"), module.__dict__)
            audit, offload = loaded["state_audit.py"], loaded["audit_offload.py"]
            need(audit.OPTIMIZER_AUDIT_POLICY == POLICY, "Unsupported optimizer audit policy")
            need(offload.checkpoint_engine_state.__globals__["engine_state"] is audit.engine_state,
                 "Offload adapter is bound to another engine_state")
            _CACHE[key] = loaded
        except BaseException:
            for module in loaded.values():
                sys.modules.pop(module.__name__, None)
            raise
    loaded = _CACHE[key]
    audit, offload = loaded["state_audit.py"], loaded["audit_offload.py"]
    need(offload.checkpoint_engine_state.__globals__["engine_state"] is audit.engine_state
         and audit.OPTIMIZER_AUDIT_POLICY == POLICY, "Cached audit binding changed")
    return job, loaded


def require_audit_binding():
    import torch
    job, loaded = modules()
    phase = Path(os.environ["CONT_PHASE_DIR"]).resolve()
    need(phase.parent == job and phase.name in ("A", "B", "C", "D", "E", "train")
         and phase.is_dir(), "Invalid audit phase directory")
    need(torch.distributed.is_initialized() and torch.distributed.get_world_size() == read(phase / "phase.json")["target_world"],
         "Audit actor world differs from the phase contract")
    # The audit recorder must use this explicit candidate phase, never silently skip.
    previous = os.environ.get("MD2_PHASE_DIR")
    need(previous is None or Path(previous).resolve() == phase, "Audit phase variables disagree")
    os.environ["MD2_PHASE_DIR"] = str(phase)
    rank = torch.distributed.get_rank()
    ordinary = sys.modules.get("d4_two_node_resumefix.state_audit")
    report = dict(version=VERSION, rank=rank, pid=os.getpid(), policy=POLICY,
                  bound_sources={n: dict(path=m.__file__, sha256=sha(Path(m.__file__).read_bytes()))
                                 for n, m in loaded.items()},
                  ordinary_module_path=getattr(ordinary, "__file__", None),
                  ordinary_module_policy=getattr(ordinary, "OPTIMIZER_AUDIT_POLICY", None))
    write(phase / f"audit_binding_rank_{rank}.json", report)
    return phase, rank, loaded


def state_digest(value, sampled=False):
    _, loaded = modules()
    return loaded["state_audit.py"].state_digest(value, sampled=sampled)


def checkpoint_engine_state(engine):
    phase, rank, loaded = require_audit_binding()
    result = loaded["audit_offload.py"].checkpoint_engine_state(engine)
    need(result.get("optimizer_audit_policy") == POLICY, "Pinned audit omitted its policy marker")
    # This durable record is independent of the recorder inside state_audit.
    target = phase / f"audit_state_rank_{rank}_{uuid.uuid4().hex}.json"
    write(target, dict(version=VERSION, rank=rank, pid=os.getpid(), state=result))
    return result


def compare_restored_state(saved, actual):
    phase, rank, _ = require_audit_binding()
    need(actual.get("optimizer_audit_policy") == POLICY, "Actual audit policy is missing")
    expected = read(saved) if Path(saved).is_file() else None
    changed = (sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
               if isinstance(expected, dict) else ["missing_saved_state"])
    write(phase / f"restore_comparison_rank_{rank}.json",
          dict(version=VERSION, rank=rank, status="FAIL" if changed else "PASS",
               saved_file=str(saved), changed_sections=changed, expected=expected, actual=actual))
    need(not changed, "Same-topology restored state differs on rank " + str(rank)
         + ": " + ", ".join(changed))
