"""IF70 test, explicit CP80 adoption, and a separate full-state Math continuation.

The original campaign, training runs and validation selections remain immutable.
Only the existing guardian owns submissions, including interruption recovery.
"""

from contextlib import ExitStack
from pathlib import Path
import shutil
import uuid

from . import DOMAINS
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

POLICY = "teacher-followup-math100-if70-cp80-v1"
PLAN = "teacher_followup.json"
MATH = "math_continued"
IF = "if_step70"
TARGETS = {IF: ("if", 70), MATH: ("math", 50)}
MATH_POLICY = dict(
    start_step=50,
    end_step=100,
    early_stop_min_step=80,
    early_stop_patience_steps=25,
    selection="equal_source_mean",
    source_weights={"nemotron": 0.5, "deepmath": 0.5},
)


def plan(c):
    path = Path(c["path"]) / PLAN
    if not path.exists():
        return None
    p = read(path)
    need(
        p["identity"] == digest({k: v for k, v in p.items() if k != "identity"})
        and p["policy"] == POLICY
        and p["campaign_identity"] == c["identity"]
        and p["math_policy"] == MATH_POLICY
        and set(p["targets"]) == set(TARGETS),
        "Teacher follow-up plan changed",
    )
    for task, (domain, step) in TARGETS.items():
        t = p["targets"][task]
        need(t["domain"] == domain and t["step"] == step, "Follow-up source changed")
    choice = p["cp_choice"]
    need(
        choice["task"] == "cp_step80" and choice["step"] == 80,
        "Explicit CP80 choice changed",
    )
    return p


def source_checkpoint(c, domain, step):
    run = Path(c["path"]) / "runs" / domain
    need(not is_fenced(c["path"], domain), "Source teacher is stopped")
    info, done, state = (
        read(run / name)
        for name in ("run.json", "training_complete.json", "state.json")
    )
    need(
        info["campaign_identity"] == c["identity"]
        and done["status"] == "PASS"
        and done["completed_step"] == step
        and done["lineage"] == info["lineage"]
        and state["phase"] != "train",
        "Source teacher is not finalized at the requested step",
    )
    directory = below(run / "checkpoints" / f"global_step_{step}", run)
    cp = ck.validate(dict(read(directory / ck.MARKER), path=str(directory)))
    need(
        cp["step"] == step and cp["lineage"] == info["lineage"],
        "Full source checkpoint has another step/lineage",
    )
    return dict(domain=domain, step=step, lineage=info["lineage"], checkpoint=cp)


def inspect_sources(c):
    from . import deepmath_data
    from .checkpoint_evaluation import verifier, validate_selected
    from .evaluation import read_receipt
    from .exporting import validate_export
    import pyarrow.parquet as pq

    root = Path(c["path"])
    need(not is_fenced(root), "Campaign is stopped")
    for task in (*TARGETS, "cp_step80"):
        need(not is_fenced(root, task), "Follow-up target is stopped")
    targets = {t: source_checkpoint(c, d, s) for t, (d, s) in TARGETS.items()}
    from .sampling import build_plan

    checkpoint = Path(targets[MATH]["checkpoint"]["path"])
    train_rows = pq.read_table(root / "data/math/train.parquet").to_pylist()
    sampling = build_plan(
        train_rows,
        "math",
        seed=c["policy"]["seed"],
        total_steps=c["policy"]["max_steps"],
    ).metadata()
    need(
        read(checkpoint / "sampling.json") == sampling,
        "Math50 sampler/cursor plan differs from the original 200-step stream",
    )
    teacher = read(checkpoint / "teacher.json")
    need(
        teacher["campaign_identity"] == c["identity"]
        and teacher["domain"] == "math"
        and teacher["lineage"] == targets[MATH]["lineage"],
        "Foreign Math50 checkpoint",
    )
    deep = deepmath_data.verify(c)
    need(deep["verifier_identity"] == verifier(c), "DeepMath verifier changed")
    # Confirm the existing Math split really is Nemotron, not a pooled split.
    manifest = read(root / "data/manifest.json")
    for split in ("validation", "test"):
        rows = pq.read_table(root / "data/math" / f"{split}.parquet").to_pylist()
        need(
            rows and all(r["domain"] == "math" and r["source"] == "math" for r in rows),
            "Expected the frozen Nemotron-only Math held-out",
        )
        need(
            [r["sample_id"] for r in rows] == manifest["domains"]["math"][split]["ids"],
            "Nemotron held-out membership changed",
        )
    all_ids = [
        sid
        for d in DOMAINS
        for s in ("train", "validation", "test")
        for sid in manifest["domains"][d][s]["ids"]
    ]
    heldout_ids = [
        sid for s in ("validation", "test") for sid in deep["splits"][s]["ids"]
    ]
    need(
        len(set(heldout_ids)) == len(heldout_ids)
        and not set(heldout_ids).intersection(all_ids),
        "DeepMath held-outs overlap",
    )
    selected = validate_selected(c, "cp_step80")
    run = root / "runs/cp_step80"
    exp = validate_export(read(run / "export.json"))
    receipt = read_receipt(run / "test.json")
    test_rows = [
        row
        for domain in DOMAINS
        for row in pq.read_table(root / "data" / domain / "test.parquet").to_pylist()
    ]
    test_runtime = read(Path(receipt["journal"]["path"]) / "manifest.json")["runtime"]
    need(
        selected["step"] == exp["step"] == receipt["step"] == 80
        and exp["selected"] == selected
        and receipt["status"] == "PASS"
        and receipt["split"] == "test"
        and receipt["binding"]["weights"] == exp["weights_identity"]
        and receipt["binding"]["verifier"] == verifier(c)
        and test_runtime["ids"] == [row["sample_id"] for row in test_rows]
        and test_runtime["dataset_sha256"] == digest(test_rows)
        and all(
            receipt["scores"][d]["n"] == len(manifest["domains"][d]["test"]["ids"])
            for d in DOMAINS
        ),
        "CP80 must have a verified export and complete four-domain test",
    )
    return (
        targets,
        deep,
        dict(
            task="cp_step80",
            step=80,
            weights_identity=exp["weights_identity"],
            test_receipt=receipt["path"],
            test_identity=receipt["identity"],
            selection_source="explicit_user_choice_after_supplementary_test",
        ),
    )


def runtime_directory(c, p):
    root = Path(c["path"]) / "teacher_followup_runtimes"
    return below(root / digest(p["runtime_files"]), root)


def apply_runtime(c, task, folder):
    from .checkpoint_evaluation import verifier

    p = plan(c)
    if not p or task not in p["targets"]:
        return None
    need(verifier(c) == p["verifier_identity"], "Follow-up verifier changed")
    runtime = runtime_directory(c, p)
    verify_files(runtime, p["runtime_files"])
    for name in p["runtime_files"]:
        target = Path(folder) / "runtime/d4_teachers" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime / name, target)
    return p


def install(c):
    from . import deepmath_data
    from .checkpoint_evaluation import pin_model, verifier

    existing = plan(c)
    if existing:
        need(
            deepmath_data.verify(c)["identity"] == existing["data_identity"]
            and verifier(c) == existing["verifier_identity"],
            "Follow-up inputs changed",
        )
        return dict(status="ALREADY_REGISTERED", identity=existing["identity"])
    root = Path(c["path"])
    with ExitStack() as stack:
        for task in ("math", "if", "cp_step80"):
            stack.enter_context(lock(root / "runs" / task / "writer.lock"))
        targets, deep, cp_choice = inspect_sources(c)
        source = Path(__file__).parent
        payload = dict(
            policy=POLICY,
            campaign_identity=c["identity"],
            targets=targets,
            math_policy=MATH_POLICY,
            cp_choice=cp_choice,
            data_identity=deep["identity"],
            verifier_identity=verifier(c),
            runtime_files=files(source),
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
                verify_files(staging, p["runtime_files"])
                staging.rename(runtime)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        verify_files(runtime, p["runtime_files"])
        for task, target in targets.items():
            run = root / "runs" / task
            lineage = (
                target["lineage"]
                if task == IF
                else "math_followup_" + p["identity"][:32]
            )
            info = dict(
                plan_identity=p["identity"],
                campaign_identity=c["identity"],
                domain=target["domain"],
                lineage=lineage,
                source_task=target["domain"],
            )
            if (run / "run.json").exists():
                need(read(run / "run.json") == info, "Unrelated follow-up run exists")
                state = (
                    read(run / "state.json") if (run / "state.json").exists() else {}
                )
                need(
                    not state.get("job") and not state.get("pending"),
                    "Unpublished follow-up run already owns work",
                )
            else:
                need(not run.exists(), "Unowned follow-up directory exists")
                write(run / "run.json", info)
            if task == MATH:
                pinned = ck.pin(target["checkpoint"], run / "initial_checkpoint")
                # This full pin includes model, optimizer, scheduler, RNG and data cursor.
                write(
                    run / "progress.json",
                    dict(
                        step=50, latest_full_step=50, source_checkpoint=pinned["path"]
                    ),
                )
            else:
                write(run / "best.json", pin_model(run, target))
            need(not is_fenced(root, task), "Follow-up stopped during registration")
            write(
                run / "state.json",
                dict(
                    status="READY",
                    phase="train" if task == MATH else "export",
                    job=None,
                    pending=None,
                    no_progress_retries=0,
                    last_progress=0,
                ),
            )
        # This final write publishes both tasks and the CP80 choice together.
        write(root / PLAN, p)
    return dict(
        status="REGISTERED",
        identity=p["identity"],
        tasks=list(targets),
        cp_choice=cp_choice,
        math_policy=MATH_POLICY,
    )


def launch(path):
    from .source import load_campaign
    from .retry_repair import repair

    c = load_campaign(path)
    if not plan(c):
        inspect_sources(c)  # All missing prerequisites fail before guardian handoff.
    else:
        install(c)  # Existing registrations only verify inputs; never reset progress.
    return dict(
        status="TEACHER_FOLLOWUP_REGISTERED",
        guardian=repair(path, install_under_lease=install if not plan(c) else None),
        note="IF70 export/test; Math full step50 continuation with two-source validation; CP80 test reused.",
    )


def initial_resume(c):
    p = plan(c)
    need(p, "Math continuation is not registered")
    pinned = ck.validate(
        read(Path(c["path"]) / "runs" / MATH / "initial_checkpoint/source.json")
    )
    expected = p["targets"][MATH]["checkpoint"]
    need(
        all(
            pinned[k] == expected[k] for k in ("step", "lineage", "files", "world_size")
        ),
        "Math continuation initial checkpoint changed",
    )
    return pinned


def validate_if_selected(c):
    from .selection import validate_best

    p = plan(c)
    need(p, "IF70 is not registered")
    run = Path(c["path"]) / "runs" / IF
    selected = validate_best(read(run / "best.json"))
    target = p["targets"][IF]
    need(
        selected["domain"] == "if"
        and selected["step"] == 70
        and selected["lineage"] == target["lineage"]
        and selected["selection_source"] == "explicit_checkpoint"
        and selected["source_checkpoint"] == target["checkpoint"]
        and selected["path"] == str(run / "best_weights/step_70")
        and selected["weights_identity"] == digest(selected["files"])
        and all(
            target["checkpoint"]["files"].get(k) == v
            for k, v in selected["files"].items()
        ),
        "IF70 selection changed",
    )
    return selected


def files_and_ids(c, split):
    from . import deepmath_data

    need(split in ("validation", "test"), "Unknown continuation evaluation split")
    p, deep = plan(c), deepmath_data.verify(c)
    need(
        p and deep["identity"] == p["data_identity"],
        "Continuation DeepMath data changed",
    )
    manifest = read(Path(c["path"]) / "data/manifest.json")
    domains = ("math",) if split == "validation" else DOMAINS
    paths = [str(Path(c["path"]) / "data" / d / f"{split}.parquet") for d in domains]
    ids = [sid for d in domains for sid in manifest["domains"][d][split]["ids"]]
    paths.append(str(deepmath_data.directory(c) / f"{split}.parquet"))
    ids.extend(deep["splits"][split]["ids"])
    return paths, ids


def source_statistics(samples, rows):
    from .evaluation import statistics

    by_id = dict(samples)
    need(
        len(by_id) == len(samples) and set(by_id) == {r["sample_id"] for r in rows},
        "Continuation evaluation IDs are incomplete/duplicated",
    )
    result = {}
    for label, sources in (
        ("nemotron", {"math"}),
        ("deepmath", {"deepmath_l8", "deepmath_l9"}),
    ):
        subset = [r for r in rows if r["domain"] == "math" and r["source"] in sources]
        need(subset, "Missing Math evaluation source: " + label)
        scores = statistics(
            [(r["sample_id"], by_id[r["sample_id"]]) for r in subset], subset
        )
        result[label] = scores["math"]
        if label == "deepmath":
            result["deepmath_by_level"] = {
                k.removeprefix("math/source/"): v
                for k, v in scores.items()
                if k.startswith("math/source/")
            }
    return result


def selection_score(receipt):
    need(
        receipt["split"] == "validation" and receipt["status"] == "PASS",
        "Only complete validation can select continuation weights",
    )
    return sum(
        MATH_POLICY["source_weights"][s] * receipt["math_sources"][s]["mean"]
        for s in ("nemotron", "deepmath")
    )


def end_step(request):
    return (
        MATH_POLICY["end_step"]
        if request.get("task") == MATH and request.get("teacher_followup")
        else request["policy"]["max_steps"]
    )


def stop_state(request, step, best_step):
    from .selection import should_stop

    policy = (
        MATH_POLICY
        if request.get("task") == MATH and request.get("teacher_followup")
        else request["policy"]
    )
    return dict(
        early_stop=should_stop(step, best_step, policy),
        budget_reached=step >= end_step(request),
    )


def validate_request(c, r):
    from . import deepmath_data

    p = plan(c)
    task = r["task"]
    need(
        p and r["teacher_followup"] == p["identity"] and task in p["targets"],
        "Unregistered follow-up request",
    )
    need(
        r["run_dir"] == str(Path(c["path"]) / "runs" / task)
        and r["domain"] == p["targets"][task]["domain"]
        and r["lineage"] == read(Path(r["run_dir"]) / "run.json")["lineage"]
        and r["policy"] == c["policy"]
        and r["verifier_identity"] == p["verifier_identity"],
        "Follow-up request binding changed",
    )
    if task == IF:
        need(r["role"] in ("export", "test"), "IF70 cannot train")
        selected = validate_if_selected(c)
        if r["role"] == "export":
            need(r["selected"] == selected, "IF70 export selection changed")
        else:
            need(r["evaluation_model"]["step"] == 70, "IF70 test step changed")
    else:
        need(
            r["math_followup_policy"] == MATH_POLICY
            and r["heldout_manifest"] == deepmath_data.verify(c)
            and r["heldout_manifest"]["identity"] == p["data_identity"],
            "Math continuation evaluation/selection policy changed",
        )
        if r["role"] == "train":
            initial_resume(c)


def validate_test(c, task):
    from .evaluation import Journal, read_receipt, statistics
    from .exporting import evaluation_model
    import pyarrow.parquet as pq

    p = plan(c)
    need(p and task in p["targets"], "Unknown follow-up test")
    root = Path(c["path"])
    receipt = read_receipt(root / "runs" / task / "test.json")
    model = evaluation_model(c, task)
    if task == MATH:
        paths, ids = files_and_ids(c, "test")
    else:
        manifest = read(root / "data/manifest.json")
        paths = [str(root / "data" / d / "test.parquet") for d in DOMAINS]
        ids = [sid for d in DOMAINS for sid in manifest["domains"][d]["test"]["ids"]]
    directory = below(receipt["journal"]["path"], root / "runs" / task / "evaluations")
    metadata = read(directory / "manifest.json")
    binding = dict(
        campaign=c["identity"],
        domain=p["targets"][task]["domain"],
        split="test",
        step=model["step"],
        weights=model["weights_identity"],
        verifier=p["verifier_identity"],
        teacher_followup=p["identity"],
    )
    runtime = metadata["runtime"]
    need(
        receipt["status"] == "PASS"
        and receipt["split"] == "test"
        and receipt["step"] == model["step"]
        and receipt["binding"] == binding
        and runtime["ids"] == ids
        and runtime["rows"] == len(ids)
        and receipt["identity"] == digest(dict(binding=binding, runtime=runtime)),
        "Follow-up test identity/data/weights changed",
    )
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    need(runtime["dataset_sha256"] == digest(rows), "Follow-up test dataset changed")
    journal = Journal(directory, binding, runtime)
    journal.index, journal.rows = runtime["batches"], runtime["rows"]
    proof, samples = journal.finish()
    need(
        proof["batch_files"] == receipt["journal"]["batch_files"]
        and receipt["scores"] == statistics(samples, rows),
        "Follow-up test scores changed",
    )
    if task == MATH:
        need(
            receipt["math_sources"] == source_statistics(samples, rows),
            "Math test source scores changed",
        )
    return receipt


def chosen_task(c, domain):
    p = plan(c)
    if not p:
        return domain
    if domain == "conversational_pivot":
        return p["cp_choice"]["task"]
    if domain == "math" and (Path(c["path"]) / "runs" / MATH / "export.json").exists():
        return MATH
    return domain


def selection_metadata(c, domain, task, export):
    p = plan(c)
    original = Path(c["path"]) / "runs" / domain / "best.json"
    result = dict(
        source_task=task,
        selected_step=export["step"],
        original_validation_best=read(original) if original.exists() else None,
    )
    if p and task == "cp_step80":
        need(
            export["step"] == 80
            and export["weights_identity"] == p["cp_choice"]["weights_identity"],
            "Adopted CP80 weights changed",
        )
        result.update(
            selection_split=None, selection_source=p["cp_choice"]["selection_source"]
        )
    elif p and task == MATH:
        result.update(
            selection_source="two_source_validation",
            selection_rule=p["math_policy"]["selection"],
        )
    elif p and domain == "math":
        result["pending_replacement"] = MATH
    return result


def four_domain_scores(receipt):
    result = {d: receipt["scores"][d] for d in DOMAINS}
    if receipt.get("math_sources"):
        result["math"] = receipt["math_sources"]["nemotron"]
    return result


def report(path):
    from .source import load_campaign
    from .controller import status

    c = load_campaign(path, verify=False)
    p = plan(c)
    need(p, "Follow-up has not been registered")
    s = status(path)
    root = Path(c["path"])
    result = {
        k: s[k]
        for k in ("guardian_lock_held", "heartbeat_stale", "heartbeat", "fenced")
    }
    result.update(math_policy=p["math_policy"], cp_choice=p["cp_choice"], tasks={})
    for task in (IF, MATH, "logic_algorithmic"):
        run = root / "runs" / task
        row = dict(s["states"][task])
        row.pop("pbs", None)
        for key in (
            "progress",
            "validation_state",
            "training_complete",
            "export",
            "test",
        ):
            f = run / (key + ".json")
            if f.exists():
                v = read(f)
                row[key] = {
                    k: v[k]
                    for k in (
                        "status",
                        "step",
                        "latest_full_step",
                        "best_step",
                        "best_score",
                        "completed_step",
                        "early_stop",
                        "budget_reached",
                        "path",
                        "scores",
                        "math_sources",
                        "binding",
                    )
                    if k in v
                }
        result["tasks"][task] = row
    return result
