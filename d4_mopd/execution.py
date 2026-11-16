"""Immutable execution snapshots, separate from the frozen scientific manifest.

The existing manifest remains the scientific identity, including its original
runtime provenance. A new execution is an additional, explicit provenance edge;
it never edits that manifest or relabels an existing checkpoint. Both profiles
use the same eight-rank student and the same sampler recipe. Qualification has
its own control directory and cannot consume the legacy short-test receipt.
"""
from copy import deepcopy
import math
from pathlib import Path
import shutil
import uuid

from . import ARMS
from . import checkpoints
from .metrics import SELECTION_KEY
from .spec import validate_spec
from d4_teachers.common import digest, file_sha, files, need, read, write


SCHEMA = "d4-mopd-execution-v1"
QUALIFICATION_SCHEMA = "d4-mopd-full-4plus4-v1"
PROFILE_NAMES = ("dual_node", "single_node")
RUNTIME_PACKAGES = ("d4_mopd", "multidomain", "d4_teachers", "d4_training_continuation")


def profile(name):
    """Return a fresh description; physical nodes do not change student ranks."""
    need(name in PROFILE_NAMES, "Unknown MOPD execution profile")
    shared = name == "single_node"
    return dict(name=name, physical_nodes=2 if name == "dual_node" else 1,
                shared_gpus=name == "single_node", student_world_size=8,
                gpus_per_node=8, actor_tp=2, actor_pp=2, actor_ep=2,
                rollout_tp=4, teacher_tp=2,
                teacher_gpu_pairs=[[0, 1], [2, 3], [4, 5], [6, 7]],
                teacher_residency="all_four_fixed_pairs",
                phase_multiplex=shared,
                teacher_scoring_mode="whole_batch_deferred" if shared else "per_sample_streaming",
                phase_order=(["student_generation", "student_sleep", "teachers_wake",
                              "teacher_scoring", "teachers_sleep", "student_old_logprob",
                              "student_update", "checkpoint_if_due", "student_weight_sync"]
                             if shared else ["student_generation_with_streamed_teacher_scoring",
                                             "student_old_logprob", "student_update",
                                             "checkpoint_if_due", "student_weight_sync"]))


def control_root(manifest):
    execution = manifest.get("execution")
    if execution is None:
        return Path(manifest["manifest_path"]).resolve().parent
    return Path(execution["control_root"])


def qualification_contract(spec):
    """Eight full updates; keep the 200-update scientific sampler unchanged."""
    validate_spec(spec)
    need(spec["train_batch_size"] == 2048 and spec["rollout_n"] == 1
         and spec["max_context_tokens"] == 65536 and spec["updates"] == 200,
         "Full qualification must preserve B2048/R1/64K and the 200-update recipe")
    return dict(schema=QUALIFICATION_SCHEMA, updates=8, pause_after=4,
                train_batch_size=2048, rollout_n=1, max_context_tokens=65536,
                scientific_spec_sha256=digest(spec), sampler_updates=200,
                reduced_training_rows=False, response_length_cap=None,
                max_num_seqs=spec["max_num_seqs"],
                max_num_batched_tokens=spec["max_num_batched_tokens"],
                gpu_memory_utilization=spec["gpu_memory_utilization"],
                actor_micro_batch_size_per_gpu=1, actor_max_token_len_per_gpu=65536,
                old_log_prob_max_token_len_per_gpu=65536,
                worst_case_64k_capacity_proven=False)


def _transition(value, target=None):
    need(isinstance(value, (list, tuple)) and len(value) == 2,
         "Choose one explicit four-plus-four qualification transition")
    result = list(value)
    for name in result:
        profile(name)
    if target is not None:
        need(result[1] == target, "Qualification must resume into the formal execution profile")
    return result


def make_record(manifest, name, runtime_files, *, qualification_transition):
    """Build deterministic metadata; no writes and no changes to the manifest."""
    need(isinstance(manifest.get("identity"), str) and manifest["identity"],
         "Missing frozen manifest identity")
    need(isinstance(runtime_files, dict) and runtime_files,
         "Execution runtime inventory is empty")
    need(all(isinstance(k, str) and isinstance(v, str) and len(v) == 64
             and all(c in "0123456789abcdef" for c in v)
             and not Path(k).is_absolute() and ".." not in Path(k).parts
             for k, v in runtime_files.items()), "Invalid execution runtime inventory")
    transition = _transition(qualification_transition, name)
    payload = dict(schema=SCHEMA, identity=manifest["identity"], profile=profile(name),
                   profiles={p: profile(p) for p in PROFILE_NAMES},
                   qualification_transition=transition,
                   qualification_contract=qualification_contract(manifest["spec"]),
                   runtime_identity=digest(runtime_files), runtime_files=deepcopy(runtime_files))
    execution_id = digest(payload)
    root = Path(manifest["manifest_path"]).resolve().parent / "executions" / execution_id
    return dict(payload, execution_identity=execution_id, control_root=str(root),
                record_path=str(root / "record.json"), runtime=str(root / "runtime"))


def verify_record(manifest, path):
    """Check a selected immutable record and the complete runtime inventory."""
    path = Path(path).resolve()
    record = read(path)
    expected = make_record(manifest, record["profile"]["name"], record["runtime_files"],
                           qualification_transition=record["qualification_transition"])
    need(record == expected and str(path) == expected["record_path"],
         "Execution record identity or path changed")
    runtime = Path(record["runtime"])
    need(not runtime.is_symlink() and files(runtime) == record["runtime_files"],
         "Execution runtime changed")
    return record


def attach_execution(manifest, path):
    """Return an augmented in-memory view; the serialized manifest stays intact."""
    result = deepcopy(manifest)
    result["execution"] = verify_record(manifest, path)
    return result


def prepare(manifest, name, source_runtime, *, qualification_transition):
    """Publish only a new snapshot/record. Never deploy, submit, or stop a job.

    The caller verifies the original manifest first. Existing training may keep
    running while this snapshot is prepared; switching writers requires the
    existing campaign locks and a separate, explicit handoff at a saved boundary.
    """
    source = Path(source_runtime).resolve()
    root = Path(manifest["manifest_path"]).resolve().parent / "executions"
    need(not root.is_symlink(), "Execution root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    stage = root / (".prepare_" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        runtime = stage / "runtime"
        for name_to_copy in (*RUNTIME_PACKAGES, "config/mopd"):
            shutil.copytree(source / name_to_copy, runtime / name_to_copy,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        record = make_record(manifest, name, files(runtime),
                             qualification_transition=qualification_transition)
        write(stage / "record.json", record)
        destination = Path(record["control_root"])
        if destination.exists():
            return verify_record(manifest, destination / "record.json")
        stage.rename(destination)
        return verify_record(manifest, destination / "record.json")
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def qualification_fields(manifest):
    record = manifest["execution"]
    need(record["identity"] == manifest["identity"], "Execution belongs to another experiment")
    return dict(execution_identity=record["execution_identity"],
                runtime_identity=record["runtime_identity"],
                execution_profile=record["profile"]["name"],
                qualification_profile_sequence=list(record["qualification_transition"]),
                qualification_contract=deepcopy(record["qualification_contract"]))


def validate_qualification(receipt, manifest, arm):
    """Fail closed for old B8/256-token receipts and a different execution route."""
    need(arm in ARMS and receipt.get("status") == "PASS"
         and receipt.get("identity") == manifest["identity"]
         and receipt.get("scope") == "qualification" and receipt.get("arm") == arm
         and receipt.get("step") == 8 and receipt.get("resumed_from") == 4,
         "Full GPU qualification and fresh-process resume are required")
    expected = qualification_fields(manifest)
    need(all(receipt.get(k) == value for k, value in expected.items()),
         "Qualification belongs to another runtime, profile, route, or budget")
    segments = receipt.get("qualification_segments")
    need(isinstance(segments, list) and len(segments) == 2,
         "Missing actual qualification segment evidence")
    for segment, name, start, end in zip(segments, expected["qualification_profile_sequence"], (0, 4), (4, 8)):
        need(segment.get("profile") == name and segment.get("loaded_step") == start
             and segment.get("completed_step") == end
             and isinstance(segment.get("attempt_id"), str) and segment["attempt_id"],
             "Qualification segment profile or resume boundary changed")
    need(segments[0]["attempt_id"] != segments[1]["attempt_id"],
         "Qualification must reload in a fresh attempt")
    return receipt


def qualification_plan(spec, arms, transitions):
    """Expose the chosen matrix cost, without adding unrequested directions."""
    contract = qualification_contract(spec)
    need(isinstance(arms, (list, tuple)) and arms and len(set(arms)) == len(arms)
         and all(a in ARMS for a in arms), "Choose unique qualification arms")
    need(isinstance(transitions, (list, tuple)) and transitions,
         "Choose an explicit qualification matrix")
    routes = [_transition(route) for route in transitions]
    need(len({tuple(route) for route in routes}) == len(routes), "Duplicate qualification transition")
    cases = [dict(arm=arm, transition=route, updates=8,
                  allocated_node_updates=sum(4 * profile(p)["physical_nodes"] for p in route))
             for route in routes for arm in arms]
    return dict(contract=contract, cases=cases, case_count=len(cases),
                total_optimizer_updates=8 * len(cases),
                total_allocated_node_updates=sum(c["allocated_node_updates"] for c in cases),
                wall_clock_and_node_hours="Require measured phase durations; node-updates are not node-hours",
                worst_case_64k_capacity_proven=False)


def _checked_checkpoint(folder, identity, arm, sampler_identity):
    need(not folder.is_symlink(), "Checkpoint directory must not be a symlink")
    proof = read(folder / "mopd_complete.json")
    step = proof["step"]
    need(type(step) is int and step > 0 and proof.get("status") == "PASS"
         and proof["identity"] == identity and proof["arm"] == arm
         and proof["sampler_identity"] == sampler_identity
         and folder.name == f"global_step_{step}", "Incompatible checkpoint marker")
    checkpoints.validate_native(folder, step)
    need(checkpoints.inventory(folder) == proof["contents"], "Committed checkpoint changed")
    return proof


def inspect_resume(run, manifest, arm, sampler_identity):
    """Read-only evidence for same-root profile handoff, under caller's writer lock.

    This checks existing native proofs; it does not deserialize GPU states or
    claim bitwise vLLM RNG restoration. No marker, selection, or receipt is edited.
    """
    need(arm in ARMS, "Unknown MOPD arm")
    run = Path(run).resolve()
    root = run / "checkpoints"
    need(not root.is_symlink(), "Checkpoint root must not be a symlink")
    identity = manifest["identity"]
    retained = {}
    for marker in root.glob("global_step_*/mopd_complete.json"):
        proof = _checked_checkpoint(marker.parent, identity, arm, sampler_identity)
        retained[proof["step"]] = (marker.parent, proof)
    latest = max(retained) if retained else 0
    retired = {}
    for path in (run / "checkpoint_retention").glob("*.json"):
        event = read(path)
        if event.get("status") != "DELETED":
            continue
        proof = event.get("checkpoint_marker", {})
        need(event.get("identity") == identity and event.get("arm") == arm
             and event.get("sampler_identity") == sampler_identity
             and proof.get("identity") == identity and proof.get("arm") == arm
             and proof.get("sampler_identity") == sampler_identity
             and proof.get("status") == "PASS" and proof.get("step") == event.get("step"),
             "Retired checkpoint lineage changed")
        need(event["source"] == str(root / f"global_step_{event['step']}"),
             "Retired checkpoint source changed")
        retired[event["step"]] = proof
    candidates, validations = [], {}
    for path in sorted((run / "validation").glob("step_*.json")):
        row = read(path)
        step = row.get("step")
        need(type(step) is int and 0 <= step <= latest and path.name == f"step_{step}.json"
             and row.get("status") == "PASS" and row.get("identity") == identity
             and row.get("arm") == arm and row.get("split") == "validation"
             and row.get("test_used_for_selection") is False,
             "Validation receipt identity, split, or step changed")
        if step == 0 or step in retained:
            reference = checkpoints.evaluation_reference(root, identity, arm, step, sampler_identity)
        else:
            need(step in retired, "Validation checkpoint missing without a retirement proof")
            reference = dict(checkpoint=str(root / f"global_step_{step}"),
                             checkpoint_identity=digest(retired[step]))
        need(all(row.get(k) == v for k, v in reference.items()), "Validation checkpoint lineage changed")
        score = row["metrics"][SELECTION_KEY]
        need(isinstance(score, (int, float)) and math.isfinite(score), "Invalid selection score")
        candidates.append(dict(step=step, score=score, **reference,
                               validation_receipt=str(path), validation_sha256=digest(row)))
        validations[str(path)] = file_sha(path)
    best = None
    if candidates:
        best = dict(max(candidates, key=lambda r: (r["score"], -r["step"])),
                    identity=identity, arm=arm, selection_key=SELECTION_KEY,
                    split="validation", test_used_for_selection=False)
        checkpoints.restore_selected(best, root, identity, arm, sampler_identity)
    published = run / "best_validation.json"
    if published.exists():
        checkpoints.restore_selected(read(published), root, identity, arm, sampler_identity)
    completed = run / "receipt.json"
    if completed.exists():
        receipt = read(completed)
        need(receipt.get("status") == "PASS" and receipt.get("identity") == identity
             and receipt.get("arm") == arm, "Completed training receipt changed")
        checkpoints.restore_selected(receipt["selection"], root, identity, arm, sampler_identity)
    selected = retained.get(latest)
    return dict(status="CHECKED", identity=identity, arm=arm, sampler_identity=sampler_identity,
                run_dir=str(run), step=latest, checkpoint=str(selected[0]) if selected else None,
                checkpoint_identity=digest(selected[1]) if selected else None,
                validation_receipts=validations, best_selection=best,
                best_pointer_current=read(published) == best if published.exists() else best is None,
                training_receipt_sha256=file_sha(completed) if completed.exists() else None,
                student_world_size=8, vllm_sampling_rng_bitwise_restored=False,
                proof_scope="Existing native inventory/lineage only; GPU load and cross-profile qualification remain required")
