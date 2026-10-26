"""Validation-only selection, weight-only hard-link pins, and step-based patience."""

import os
from pathlib import Path
import shutil
import uuid
from .common import below, digest, need, read, write
from d4_training_continuation import checkpoints as ck


def should_stop(step, best_step, policy):
    return (
        step >= policy["early_stop_min_step"]
        and step - best_step >= policy["early_stop_patience_steps"]
    )


def validate_best(best):
    if best["step"]:
        if Path(best["path"]).is_dir():
            need(
                ck.inventory(best["path"]) == best["files"], "Selected weights changed"
            )
        else:
            # Final retention replaces the native best pin with its proven HF
            # export; keep the original selection record as provenance.
            from .exporting import validate_export

            export = validate_export(
                read(Path(best["path"]).parent.parent / "export.json")
            )
            need(export["selected"] == best, "Retired best has no matching HF export")
    return best


def select(request, checkpoint, receipt):
    need(
        request["role"] == "train" and receipt["split"] == "validation",
        "Only validation can select teachers",
    )
    need(receipt["step"] == checkpoint["step"], "Selection checkpoint mismatch")
    run = Path(request["run_dir"])
    score = receipt["scores"][request["domain"]]["mean"]
    previous = read(run / "best.json") if (run / "best.json").exists() else None
    if previous and score <= previous["score"]:
        validate_best(previous)  # Ties always retain the earlier checkpoint.
        return previous
    step = checkpoint["step"]
    best = dict(
        step=step,
        score=score,
        domain=request["domain"],
        lineage=request["lineage"],
        validation_receipt=receipt["path"],
        validation_identity=receipt["identity"],
        verifier_identity=request["verifier_identity"],
    )
    if step == 0:
        best.update(
            path=request["base_model"],
            format="huggingface",
            files=None,
            weights_identity=request["base_identity"],
        )
    else:
        ck.validate(checkpoint)
        source = Path(checkpoint["path"])
        spec = read(source / "actor/ckpt_contents.json")
        rel = Path(spec["contents"]["model"]["path"])
        below(source / "actor" / rel, source / "actor")
        target = run / "best_weights" / ("step_" + str(step))
        staging = target.with_name(".pin_" + uuid.uuid4().hex)
        staging.mkdir(parents=True)
        try:
            prefix = str(Path("actor") / rel) + "/"
            names = [
                p
                for p in checkpoint["files"]
                if p.startswith(prefix) or p == "actor/ckpt_contents.json"
            ]
            need(
                any(p.endswith(".distcp") for p in names),
                "Best has no native model shards",
            )
            for name in names:
                dest = staging / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(source / name, dest, follow_symlinks=False)
            inv = ck.inventory(staging)
            if target.exists():
                need(ck.inventory(target) == inv, "Conflicting unpublished best pin")
                shutil.rmtree(staging)
            else:
                staging.rename(target)
            best.update(
                path=str(target),
                format="megatron-model-only",
                files=inv,
                weights_identity=digest(inv),
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    write(run / "best.json", best)
    # Publish the replacement pointer before releasing the previous weights.
    for folder in (run / "best_weights").glob("step_*"):
        if str(folder) != best["path"]:
            shutil.rmtree(below(folder, run / "best_weights"))
    return validate_best(best)
