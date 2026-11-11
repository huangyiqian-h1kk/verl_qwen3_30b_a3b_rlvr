"""Audited Ray socket-path repair for an already-frozen, unstarted campaign."""
import hashlib
from pathlib import Path
import shutil
import sys
import uuid

from .assets import verify
from d4_teachers.common import digest, files, lock, need, now, read, write
from d4_training_continuation.common import locked


OLD_TMP = 'export RAY_TMPDIR="${PBS_LOCALDIR:?}/mopd_${D4_MOPD_ATTEMPT}"\nmkdir -p "$RAY_TMPDIR"\n'
NEW_TMP = '''export RAY_TMPDIR="${PBS_LOCALDIR:?}/r"
mkdir -p "$RAY_TMPDIR"
# PBS_LOCALDIR already isolates each job on each node. Ray adds a session name;
# including the full attempt UUID here would exceed Linux's AF_UNIX limit.
"$mopd_python" -I - "$RAY_TMPDIR" <<'PY'
import os
import sys
path = sys.argv[1] + "/session_9999-12-31_23-59-59_999999_2147483647/sockets/plasma_store"
size = len(os.fsencode(path))
if size > 107:
    raise RuntimeError(f"Ray socket path needs up to {size} bytes (>107); PBS_LOCALDIR is too long: {sys.argv[1]}")
print(f"[0390] Ray temp directory: {sys.argv[1]}; socket path budget: {size}/107 bytes", flush=True)
PY
'''
SCHEMA = "d4-mopd-launcher-repair-v1"


def replace_once(text, before, after):
    need(text.count(before) == 1, "Unknown frozen launcher layout; refusing partial repair")
    return text.replace(before, after, 1)


def patched_sources(m):
    """Keep every learner/configuration file; change only launch path handling."""
    source = Path(m["manifest_path"]).parent / "runtime"
    need(files(source) == m["runtime_files"], "Frozen runtime changed before launcher repair")
    node = replace_once((source / "d4_mopd/node.sh").read_text(), OLD_TMP, NEW_TMP)
    controller = (source / "d4_mopd/control.py").read_text()
    controller = replace_once(controller,
        'repr([str(root / "runtime"), m["verl_src"]])',
        'repr([str(Path(__file__).resolve().parents[1]), m["verl_src"]])')
    controller = replace_once(controller,
        '    runtime = Path(m["manifest_path"]).parent / "runtime"\n',
        '    runtime = Path(__file__).resolve().parents[1]\n')
    return {"d4_mopd/node.sh": node, "d4_mopd/control.py": controller}


def revision_plan(m):
    changes = patched_sources(m)
    inventory = dict(m["runtime_files"])
    inventory.update({name: hashlib.sha256(text.encode()).hexdigest() for name, text in changes.items()})
    revision = digest(dict(schema=SCHEMA, identity=m["identity"], runtime_files=inventory))
    folder = Path(m["manifest_path"]).parent / "launcher_revisions" / revision
    return changes, inventory, revision, folder


def active_runtime(m):
    """Select a checked operational snapshot; GPU Python still uses original runtime."""
    root = Path(m["manifest_path"]).parent
    pointer = root / "launcher_repair.json"
    if not pointer.exists():
        return root / "runtime"
    _, inventory, revision, folder = revision_plan(m)
    selected = read(pointer)
    need(selected == dict(schema=SCHEMA, identity=m["identity"], revision=revision, folder=str(folder)),
         "Launcher repair pointer changed")
    receipt = read(folder / "receipt.json")
    need(receipt.get("status") == "PASS" and receipt.get("identity") == m["identity"]
         and receipt.get("revision") == revision and receipt.get("runtime_files") == inventory,
         "Launcher repair receipt changed")
    need(files(folder / "runtime") == inventory, "Repaired launcher snapshot changed")
    return folder / "runtime"


def repair(manifest_path, *, lease_fd=None, on_pbs_unknown=None):
    from . import ARMS, control
    from .repair_pipeline import environment_lease
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    need(Path(sys.executable).absolute() == Path(m["python"]).absolute(), "Use the manifest interpreter")
    with environment_lease(m, lease_fd), lock(root / "launch.lock"), \
            locked(root / "guardian.lock"), lock(root / "states.lock"):
        need(not (root / "STOP").exists(), "Experiment is stopped")
        state = read(root / "states.json")
        need(set(state) == {"qualification:" + a for a in ARMS}, "Both qualifications are required")
        observations = control.quiescent_qualifications(m, state, control.PBS(), on_pbs_unknown=on_pbs_unknown)
        for s in state.values():
            attempt = Path(s["job"]["folder"])
            need(not (attempt / "result.json").exists(), "Learner execution already began; inspect its result")
            log = (attempt / "ray_start_0.log").read_text(errors="replace")
            need("AF_UNIX path length cannot exceed 107 bytes:" in log,
                 "Qualification does not show the targeted Ray socket-path error")
        changes, inventory, revision, folder = revision_plan(m)
        pointer = root / "launcher_repair.json"
        if pointer.exists():
            active_runtime(m)
        else:
            stage = folder.parent / ("." + revision + "." + uuid.uuid4().hex)
            stage.mkdir(parents=True)
            try:
                shutil.copytree(root / "runtime", stage / "runtime",
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                for name, text in changes.items():
                    (stage / "runtime" / name).write_text(text)
                need(files(stage / "runtime") == inventory, "Unexpected launcher repair changes")
                receipt = dict(status="PASS", schema=SCHEMA, identity=m["identity"], revision=revision,
                               runtime_files=inventory, changed_files=sorted(changes),
                               original_runtime_files=m["runtime_files"], before=state, pbs=observations,
                               scope="Ray path and controller launch locations only; learner runtime unchanged",
                               at_utc=now())
                write(stage / "receipt.json", receipt)
                if folder.exists():
                    old = read(folder / "receipt.json")
                    need(old.get("status") == "PASS" and old.get("revision") == revision
                         and old.get("identity") == m["identity"] and old.get("runtime_files") == inventory
                         and files(folder / "runtime") == inventory, "Incomplete/conflicting launcher revision")
                else:
                    stage.rename(folder)
                write(pointer, dict(schema=SCHEMA, identity=m["identity"], revision=revision, folder=str(folder)))
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        selected = active_runtime(m)
        print("Ray launcher revision: " + str(selected), flush=True)
        return dict(status="PASS", identity=m["identity"], revision=revision, runtime=str(selected),
                    changed_files=sorted(changes), learner_runtime=str(root / "runtime"), at_utc=now())
