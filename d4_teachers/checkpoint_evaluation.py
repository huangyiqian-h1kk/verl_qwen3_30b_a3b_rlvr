"""Explicit final-checkpoint tests, owned by the existing campaign guardian."""

from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import uuid

from .common import (
    below,
    digest,
    files,
    is_fenced,
    lock,
    need,
    read,
    verify_files,
    write,
)
from d4_training_continuation import checkpoints as ck

POLICY = "final-checkpoint-tests-v1"
TARGETS = {"math_step50": ("math", 50), "cp_step80": ("conversational_pivot", 80)}
PLAN = "checkpoint_evaluation.json"


def plan(c):
    path = Path(c["path"]) / PLAN
    if not path.exists():
        return None
    p = read(path)
    need(
        p["identity"] == digest({k: v for k, v in p.items() if k != "identity"})
        and p["campaign_identity"] == c["identity"]
        and p["policy"] == POLICY,
        "Checkpoint evaluation plan changed",
    )
    need(
        set(p["targets"]) == set(TARGETS) and p["include_base"] is True,
        "Unknown checkpoint evaluation targets",
    )
    for task, (domain, step) in TARGETS.items():
        need(
            p["targets"][task]["domain"] == domain
            and p["targets"][task]["step"] == step,
            "Checkpoint target changed",
        )
    return p


def verifier(c):
    from .math_test_repair import active_revision

    revision = active_revision(c)
    return revision["identity"] if revision else c["contract"]["verifier_identity"]


def runtime_directory(c, p):
    identity = digest(p["runtime_files"])
    return below(
        Path(c["path"]) / "checkpoint_evaluation_runtimes" / identity,
        Path(c["path"]) / "checkpoint_evaluation_runtimes",
    )


def apply_runtime(c, task, folder):
    p = plan(c)
    if not p or task not in (*p["targets"], "base"):
        return None
    need(
        verifier(c) == p["verifier_identity"],
        "Test verifier changed since checkpoint evaluation was requested",
    )
    source = runtime_directory(c, p)
    verify_files(source, p["runtime_files"])
    for name in p["runtime_files"]:
        target = Path(folder) / "runtime/d4_teachers" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    return p


def validate_selected(c, task):
    from .selection import validate_best

    p = plan(c)
    need(p and task in p["targets"], "Unregistered checkpoint evaluation")
    run = Path(c["path"]) / "runs" / task
    best = validate_best(read(run / "best.json"))
    target = p["targets"][task]
    need(
        best["selection_source"] == "explicit_checkpoint"
        and best["step"] == target["step"]
        and best["domain"] == target["domain"]
        and best["source_checkpoint"] == target["checkpoint"]
        and best["lineage"] == target["lineage"]
        and read(run / "run.json")["lineage"] == target["lineage"],
        "Explicit checkpoint selection changed",
    )
    need(
        best["path"] == str(run / "best_weights" / ("step_" + str(target["step"])))
        and best["weights_identity"] == digest(best["files"])
        and all(
            target["checkpoint"]["files"].get(n) == v for n, v in best["files"].items()
        ),
        "Pinned weights differ from the requested checkpoint",
    )
    if (run / "export.json").exists():
        from .exporting import validate_export

        need(
            validate_export(read(run / "export.json"))["selected"] == best,
            "HF export belongs to another checkpoint",
        )
    return best


def inspect_sources(c):
    """Read-only checks before handing off the guardian or registering any work."""
    root, targets = Path(c["path"]), {}
    need(not is_fenced(root), "Campaign is stopped")
    for task, (domain, step) in TARGETS.items():
        need(
            not is_fenced(root, domain) and not is_fenced(root, task),
            "Checkpoint evaluation source/target is stopped",
        )
        run = root / "runs" / domain
        info = read(run / "run.json")
        done = read(run / "training_complete.json")
        state = read(run / "state.json")
        need(
            info["campaign_identity"] == c["identity"]
            and done["status"] == "PASS"
            and done["completed_step"] == step
            and done["lineage"] == info["lineage"]
            and state["phase"] != "train",
            "Source teacher has not finalized the requested checkpoint: " + task,
        )
        directory = below(run / "checkpoints" / ("global_step_" + str(step)), run)
        cp = ck.validate(dict(read(directory / ck.MARKER), path=str(directory)))
        need(
            cp["step"] == step and cp["lineage"] == info["lineage"],
            "Source checkpoint step/lineage mismatch",
        )
        targets[task] = dict(
            domain=domain, step=step, lineage=info["lineage"], checkpoint=cp
        )
    need(not is_fenced(root, "base"), "Base test is stopped")
    return targets


def pin_model(run, target):
    """Pin only native model shards; never manufacture a validation selection."""
    cp = ck.validate(target["checkpoint"])
    source = Path(cp["path"])
    rel = Path(read(source / "actor/ckpt_contents.json")["contents"]["model"]["path"])
    below(source / "actor" / rel, source / "actor")
    prefix = str(Path("actor") / rel) + "/"
    names = [
        n
        for n in cp["files"]
        if n.startswith(prefix) or n == "actor/ckpt_contents.json"
    ]
    need(any(n.endswith(".distcp") for n in names), "No native model shards")
    path = run / "best_weights" / ("step_" + str(target["step"]))
    staging = path.with_name(".pin_" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        for name in names:
            dest = staging / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(source / name, dest, follow_symlinks=False)
        inv = ck.inventory(staging)
        need(inv == {n: cp["files"][n] for n in names}, "Source changed while pinning")
        if path.exists():
            need(ck.inventory(path) == inv, "Conflicting checkpoint pin")
        else:
            staging.rename(path)
        return dict(
            step=target["step"],
            domain=target["domain"],
            lineage=target["lineage"],
            score=None,
            selection_source="explicit_checkpoint",
            source_checkpoint=cp,
            path=str(path),
            format="megatron-model-only",
            files=inv,
            weights_identity=digest(inv),
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def install(c):
    """Called only under control/guardian leases; publication is the final write."""
    existing = plan(c)
    if existing:
        return dict(status="ALREADY_REGISTERED", identity=existing["identity"])
    root = Path(c["path"])
    with ExitStack() as stack:
        for domain, _ in TARGETS.values():
            stack.enter_context(lock(root / "runs" / domain / "writer.lock"))
        targets = inspect_sources(c)
        source = Path(__file__).parent
        runtime_files = files(source)
        payload = dict(
            policy=POLICY,
            campaign_identity=c["identity"],
            targets=targets,
            include_base=True,
            verifier_identity=verifier(c),
            runtime_files=runtime_files,
        )
        p = dict(payload, identity=digest(payload))
        runtime = runtime_directory(c, p)
        if not runtime.exists():
            runtime.parent.mkdir(parents=True, exist_ok=True)
            staging = runtime.with_name(".staging_" + uuid.uuid4().hex)
            try:
                shutil.copytree(
                    source,
                    staging,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                verify_files(staging, runtime_files)
                staging.rename(runtime)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        verify_files(runtime, runtime_files)
        for task, target in targets.items():
            run = root / "runs" / task
            intent = dict(
                plan_identity=p["identity"],
                domain=target["domain"],
                lineage=target["lineage"],
                campaign_identity=c["identity"],
            )
            if run.exists():
                need(
                    read(run / "run.json") == intent,
                    "Unrelated evaluation run already exists",
                )
                state = (
                    read(run / "state.json") if (run / "state.json").exists() else {}
                )
                need(
                    not state.get("job") and not state.get("pending"),
                    "Unpublished run owns work",
                )
            write(run / "run.json", intent)
            write(run / "best.json", pin_model(run, target))
            write(
                run / "state.json",
                dict(
                    status="READY",
                    phase="export",
                    job=None,
                    pending=None,
                    no_progress_retries=0,
                    last_progress=0,
                ),
            )
        # Base's existing job, receipt, state and cache remain owned exactly once.
        write(root / PLAN, p)
    return dict(
        status="REGISTERED", identity=p["identity"], targets=list(targets) + ["base"]
    )


def launch(path):
    from .source import load_campaign
    from .retry_repair import repair

    c = load_campaign(path)
    existing = plan(c)
    if not existing:
        inspect_sources(c)
    else:
        need(verifier(c) == existing["verifier_identity"], "Test verifier changed")
        for task in TARGETS:
            validate_selected(c, task)
    result = repair(path, install_under_lease=None if existing else install)
    return dict(
        status="CHECKPOINT_EVALUATIONS_REGISTERED",
        guardian=result,
        targets={
            **{t: dict(domain=d, step=s) for t, (d, s) in TARGETS.items()},
            "base": dict(domain="base", step=0),
        },
        note="Export then four-domain test; base tests directly. Existing jobs are adopted.",
    )
