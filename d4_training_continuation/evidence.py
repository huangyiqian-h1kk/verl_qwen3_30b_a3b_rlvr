"""Validate actual ranks, fresh processes and all five complete update phases."""
from collections import Counter
from pathlib import Path
from . import PHASES
from .common import digest, need, read, sha, write
from . import checkpoints as ck


def topology(rows, world):
    need(world in (8, 16) and len(rows) == world and {r["rank"] for r in rows} == set(range(world)), "Missing actor ranks")
    need(sorted(Counter(r["host"] for r in rows).values()) == [8] * (world // 8),
         "Eight-rank phase must occupy one physical host; sixteen-rank phase must occupy two")
    for r in rows:
        need(r.get("optimizer_repair") == "continuation-optimizer-v1", "Unpatched optimizer loader")
        for key, expected in dict(world_size=world, tp=2, pp=2, ep=2, cp=1, dp=world // 4).items():
            need(r[key] == expected, "Actor topology differs: " + key)
    return rows


def phase_pass(request, phase, resume, checkpoint, step):
    phase = Path(phase)
    execution = read(phase / "phase.json")
    world = execution["target_world"]
    topology(read(phase / "topology.json")["actor_ranks"], world)
    loader = read(phase / "resume.json")
    need(loader["status"] == "PASS" and loader["step"] == resume["step"], "Dataloader restore not proven")
    batch = read(phase / f"step_{step}.json")
    need(step == resume["step"] + 1 and batch["step"] == step and batch["weights_synchronized"]
         and batch["actor_update_completed"] and batch["prompts"] == 256 and batch["responses"] == 2048
         and batch["saved_checkpoint_step"] == step, "Missing real update/save/sync")
    ck.validate(checkpoint)
    evidence = ["phase.json", "topology.json", "resume.json", "resume_source.json", f"step_{step}.json"]
    for rank in range(world):
        restored = read(phase / f"restore_rank_{rank}.json")
        loaded = read(phase / f"optimizer_load_returned_rank_{rank}.json")
        bound = read(phase / f"audit_binding_rank_{rank}.json")
        need(restored["status"] == "PASS" and restored["rank"] == rank
             and restored["token"] == loaded["token"] == execution["token"]
             and loaded["status"] == "LOAD_RETURNED" and loaded["rank"] == rank
             and loaded["world_size"] == restored["target_world"] == world
             and restored["source_world"] == resume["world_size"], "Missing rank restore evidence")
        need(bound["policy"] == "fused-adam-empty-local-group-step-v1" and bound["rank"] == rank,
             "Unbound state audit")
        need(restored["full_model_optimizer_load_returned"] and restored["lr_state_equal"] and restored["rng_restored"],
             "Incomplete checkpoint restore")
        if resume["world_size"] == world and resume.get("lineage"):
            compare = read(phase / f"restore_comparison_rank_{rank}.json")
            need(compare["status"] == "PASS" and not compare["changed_sections"], "Same-layout state mismatch")
            evidence.append(f"restore_comparison_rank_{rank}.json")
        saved = read(Path(checkpoint["path"]) / "actor" / f"continuation_state_rank_{rank}.json")
        write(phase / f"saved_state_rank_{rank}.json", saved)
        evidence += [f"restore_rank_{rank}.json", f"optimizer_load_returned_rank_{rank}.json",
                     f"audit_binding_rank_{rank}.json", f"saved_state_rank_{rank}.json"]
    # Preserve small evidence when intermediate full checkpoints are pruned.
    write(phase / "checkpoint_manifest.json", read(Path(checkpoint["path"]) / ck.MARKER))
    evidence.append("checkpoint_manifest.json")
    result = dict(status="PASS", identity=request["identity"], phase=phase.name,
                  token=execution["token"], source_world=resume["world_size"], target_world=world,
                  source_step=resume["step"], step=step, checkpoint=checkpoint,
                  source_files_digest=digest(resume["files"]),
                  evidence={name: sha(phase / name) for name in evidence})
    write(phase / "qualification.json", result)
    return result


def verify_qualification(folder, ready):
    folder = Path(folder)
    tokens, previous = set(), None
    for name, source_world, target_world in PHASES:
        phase = folder / name
        row = read(phase / "qualification.json")
        execution = read(phase / "phase.json")
        need(row["status"] == "PASS" and row["identity"] == ready["identity"] and row["phase"] == name,
             "Missing qualification phase " + name)
        need((row["source_world"], row["target_world"]) == (source_world, target_world), "Wrong phase transition")
        need(row["token"] == execution["token"] and row["token"] not in tokens, "Phase was not a fresh execution")
        tokens.add(row["token"])
        need(execution["execution_id"] == ready["execution_id"], "Qualification mixed PBS executions")
        for file, expected in row["evidence"].items():
            from .common import below
            need(sha(below(phase / file, phase)) == expected, "Phase evidence changed")
        need(row["step"] == row["source_step"] + 1, "Qualification skipped/repeated a step")
        if previous:
            need(row["source_step"] == previous["step"] and row["source_files_digest"] == digest(previous["checkpoint"]["files"]),
                 "Qualification resumed an unrelated checkpoint")
        previous = row
    need(previous["checkpoint"] == ready["checkpoint"], "READY is not the final qualified checkpoint")
    return True
