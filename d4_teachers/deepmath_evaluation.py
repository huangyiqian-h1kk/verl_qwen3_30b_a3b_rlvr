"""Independent DeepMath base/20/50 evaluations under the campaign's one guardian."""

from itertools import combinations
from pathlib import Path
import shutil
import uuid

from .common import below, digest, files, is_fenced, need, read, verify_files, write
from . import deepmath_data

POLICY = "deepmath-teacher-comparison-v1"
PLAN = "deepmath_evaluation.json"
TARGETS = {
    "deepmath_base": ("base", 0),
    "deepmath_math20": ("math", 20),
    "deepmath_math50": ("math_step50", 50),
}


def plan(c):
    path = Path(c["path"]) / PLAN
    if not path.exists():
        return None
    p = read(path)
    need(
        p["identity"] == digest({k: v for k, v in p.items() if k != "identity"})
        and p["campaign_identity"] == c["identity"]
        and p["policy"] == POLICY
        and set(p["targets"]) == set(TARGETS),
        "DeepMath evaluation plan changed",
    )
    for task, (source, step) in TARGETS.items():
        need(
            p["targets"][task]["source_task"] == source
            and p["targets"][task]["step"] == step,
            "DeepMath model target changed",
        )
    return p


def inspect_models(c):
    from .checkpoint_evaluation import validate_selected
    from .exporting import validate_export
    from .common import model_inventory

    root = Path(c["path"])
    need(not is_fenced(root), "Campaign is stopped")
    need(
        model_inventory(c["contract"]["base_model"]) == c["contract"]["base_inventory"],
        "Base HF model changed",
    )
    targets = {}
    for task, (source, step) in TARGETS.items():
        need(
            not is_fenced(root, task) and not is_fenced(root, source),
            "DeepMath source/target stopped",
        )
        target = dict(source_task=source, step=step)
        if source == "base":
            target["base_identity"] = c["contract"]["base_identity"]
        else:
            run = root / "runs" / source
            if source == "math_step50":
                selected = validate_selected(c, source)
            else:
                selected = validate_export(read(run / "export.json"))["selected"]
            need(
                selected["step"] == step and selected["domain"] == "math",
                "Expected Math checkpoint unavailable",
            )
            target.update(
                selected_identity=digest(selected), lineage=selected["lineage"]
            )
        targets[task] = target
    return targets


def evaluation_model(c, task):
    from .exporting import evaluation_model as original, validate_export
    from .common import model_inventory

    p = plan(c)
    need(p and task in p["targets"], "Unregistered DeepMath task")
    target = p["targets"][task]
    source = target["source_task"]
    if source == "base":
        need(
            model_inventory(c["contract"]["base_model"])
            == c["contract"]["base_inventory"]
            and target["base_identity"] == c["contract"]["base_identity"],
            "DeepMath base changed",
        )
    else:
        export = validate_export(
            read(Path(c["path"]) / "runs" / source / "export.json")
        )
        need(
            digest(export["selected"]) == target["selected_identity"]
            and export["selected"]["lineage"] == target["lineage"],
            "DeepMath checkpoint selection changed",
        )
    model = original(c, source)
    need(model["step"] == target["step"], "DeepMath checkpoint step changed")
    return model


def ready(c, task, state):
    p = plan(c)
    if not p or task not in p["targets"]:
        return True
    source = p["targets"][task]["source_task"]
    if (
        source != "base"
        and not (Path(c["path"]) / "runs" / source / "export.json").exists()
    ):
        state.update(status="WAITING_MODEL", waiting_for=source + "/export.json")
        return False
    evaluation_model(c, task)
    state.pop("waiting_for", None)
    state["status"] = "READY"
    return True


def runtime_directory(c, p):
    return below(
        Path(c["path"])
        / "deepmath_evaluation"
        / "runtimes"
        / digest(p["runtime_files"]),
        Path(c["path"]) / "deepmath_evaluation" / "runtimes",
    )


def apply_runtime(c, task, folder):
    from .checkpoint_evaluation import verifier

    p = plan(c)
    if not p or task not in p["targets"]:
        return None
    need(verifier(c) == p["verifier_identity"], "DeepMath verifier changed")
    need(
        deepmath_data.verify(c)["identity"] == p["data_identity"],
        "DeepMath dataset changed",
    )
    source = runtime_directory(c, p)
    verify_files(source, p["runtime_files"])
    for name in p["runtime_files"]:
        target = Path(folder) / "runtime/d4_teachers" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    return p


def install(c):
    from .checkpoint_evaluation import verifier

    existing = plan(c)
    if existing:
        need(
            deepmath_data.verify(c)["identity"] == existing["data_identity"]
            and verifier(c) == existing["verifier_identity"],
            "Registered DeepMath inputs changed",
        )
        return dict(status="ALREADY_REGISTERED", identity=existing["identity"])
    root = Path(c["path"])
    data = deepmath_data.verify(c)
    targets = inspect_models(c)
    need(
        data["verifier_identity"] == verifier(c),
        "DeepMath CPU preflight used another verifier",
    )
    source = Path(__file__).parent
    payload = dict(
        policy=POLICY,
        campaign_identity=c["identity"],
        targets=targets,
        data_identity=data["identity"],
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
                source, staging, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
            verify_files(staging, p["runtime_files"])
            staging.rename(runtime)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    verify_files(runtime, p["runtime_files"])
    for task, target in targets.items():
        run = root / "runs" / task
        intent = dict(
            plan_identity=p["identity"],
            campaign_identity=c["identity"],
            domain="math",
            lineage="deepmath_" + digest(target),
            source_task=target["source_task"],
        )
        if run.exists():
            need(
                read(run / "run.json") == intent,
                "Unrelated DeepMath task directory exists",
            )
            state = read(run / "state.json") if (run / "state.json").exists() else {}
            need(
                not state.get("job") and not state.get("pending"),
                "Unpublished DeepMath task owns work",
            )
        need(not is_fenced(root, task), "DeepMath target stopped during registration")
        write(run / "run.json", intent)
        write(
            run / "state.json",
            dict(
                status="READY",
                phase="test",
                job=None,
                pending=None,
                no_progress_retries=0,
                last_progress=0,
            ),
        )
    # A crash before this publication leaves no schedulable new targets.
    write(root / PLAN, p)
    return dict(status="REGISTERED", identity=p["identity"], tasks=list(targets))


def launch(path):
    from .source import load_campaign
    from .retry_repair import repair
    from .checkpoint_evaluation import verifier

    c = load_campaign(path)
    data = deepmath_data.verify(c)
    existing = plan(c)
    if existing:
        need(
            existing["data_identity"] == data["identity"]
            and existing["verifier_identity"] == verifier(c),
            "DeepMath plan no longer matches",
        )
    else:
        inspect_models(c)
    result = repair(path, install_under_lease=None if existing else install)
    return dict(
        status="DEEPMATH_EVALUATIONS_REGISTERED",
        guardian=result,
        note="Three read-only model evaluations; Math50 waits for its existing verified HF export.",
    )


def files_and_ids(c):
    m = deepmath_data.verify(c)
    return (
        [
            str(deepmath_data.directory(c) / (s + ".parquet"))
            for s in deepmath_data.SPLITS
        ],
        [sid for s in deepmath_data.SPLITS for sid in m["splits"][s]["ids"]],
    )


def split_statistics(samples, rows):
    from .evaluation import statistics

    by_id = dict(samples)
    return {
        s: statistics(
            [
                (r["sample_id"], by_id[r["sample_id"]])
                for r in rows
                if r["evaluation_split"] == s
            ],
            [r for r in rows if r["evaluation_split"] == s],
        )
        for s in deepmath_data.SPLITS
    }


def validate_result(c, task):
    from .evaluation import Journal, read_receipt
    import pyarrow.parquet as pq

    p = plan(c)
    need(p and task in p["targets"], "Unregistered DeepMath result")
    receipt = read_receipt(Path(c["path"]) / "runs" / task / "heldout.json")
    model = evaluation_model(c, task)
    paths, ids = files_and_ids(c)
    journal_dir = below(
        receipt["journal"]["path"], Path(c["path"]) / "runs" / task / "evaluations"
    )
    journal = read(journal_dir / "manifest.json")
    binding = dict(
        campaign=c["identity"],
        domain="math",
        split="heldout",
        step=model["step"],
        weights=model["weights_identity"],
        verifier=p["verifier_identity"],
        deepmath_evaluation=p["identity"],
    )
    rows = [r for path in paths for r in pq.read_table(path).to_pylist()]
    runtime = journal["runtime"]
    need(
        receipt["identity"] == digest(dict(binding=binding, runtime=runtime)),
        "DeepMath evaluation identity differs from journal",
    )
    checker = Journal(journal_dir, binding, runtime)
    samples = []
    names = [f"batch_{i:06d}.json" for i in range(1, runtime["batches"] + 1)]
    need(
        sorted(receipt["journal"]["batch_files"]) == names,
        "DeepMath journal batches incomplete",
    )
    for i, name in enumerate(names, 1):
        envelope = read(journal_dir / name)
        need(
            digest(envelope["payload"]) == envelope["sha256"],
            "DeepMath batch payload changed",
        )
        checker.check(envelope["payload"])
        need(envelope["payload"]["batch"] == i, "DeepMath batch order changed")
        values = envelope["payload"]["lists"]
        samples.extend(zip(values["sample_uids"], values["sample_scores"]))
    need(
        receipt["status"] == "PASS"
        and receipt["split"] == "heldout"
        and receipt["binding"] == binding
        and receipt["journal"]["rows"] == runtime["rows"] == len(ids)
        and runtime["ids"] == ids
        and receipt["journal"]["batches"] == runtime["batches"]
        and runtime["dataset_sha256"] == digest(rows)
        and [sid for sid, _ in samples] == ids
        and len(set(ids)) == len(ids),
        "Incomplete/different DeepMath evaluation",
    )
    need(
        receipt["split_scores"] == split_statistics(samples, rows),
        "DeepMath split metrics differ from journal",
    )
    return receipt, rows, samples


def report(path):
    from .source import load_campaign
    import numpy as np

    c = load_campaign(path)
    p = plan(c)
    need(p, "DeepMath evaluations not registered")
    outputs, values, rows, protocol = {}, {}, None, None
    for task in TARGETS:
        run = Path(c["path"]) / "runs" / task
        state = read(run / "state.json")
        outputs[task] = dict(
            status=state["status"],
            phase=state["phase"],
            job_id=(state.get("job") or {}).get("job_id"),
        )
        if (run / "heldout.json").exists():
            receipt, rows, samples = validate_result(c, task)
            current = read(Path(receipt["journal"]["path"]) / "manifest.json")[
                "runtime"
            ]
            need(
                protocol is None or protocol == current,
                "Model evaluation protocols differ; do not pool scores",
            )
            protocol = current
            outputs[task].update(
                scores=receipt["split_scores"],
                receipt=receipt["path"],
                weights_identity=receipt["binding"]["weights"],
            )
            values[task] = dict(samples)
    comparisons = []
    if rows:
        for first, second in combinations(values, 2):
            for split in deepmath_data.SPLITS:
                for level in (None, 8, 9):
                    ids = [
                        r["sample_id"]
                        for r in rows
                        if r["evaluation_split"] == split
                        and (level is None or r["difficulty_level"] == level)
                    ]
                    delta = np.array(
                        [values[second][sid] - values[first][sid] for sid in ids],
                        dtype=float,
                    )
                    rng = np.random.default_rng(42)
                    boot = delta[
                        rng.integers(0, len(ids), size=(10000, len(ids)))
                    ].mean(axis=1)
                    comparisons.append(
                        dict(
                            first=first,
                            second=second,
                            split=split,
                            level=level,
                            n=len(ids),
                            mean_difference=float(delta.mean()),
                            ci95=[float(x) for x in np.quantile(boot, [0.025, 0.975])],
                            improved=int((delta > 0).sum()),
                            worsened=int((delta < 0).sum()),
                            method="paired percentile bootstrap, 10000 replicates, seed42; exploratory unadjusted intervals",
                        )
                    )
    result = dict(
        status="PASS" if len(values) == len(TARGETS) else "INCOMPLETE",
        plan=p["identity"],
        models=outputs,
        paired_comparisons=comparisons,
        note="Supplementary held-out comparison; original validation selection remains recorded separately.",
    )
    write(Path(c["path"]) / "deepmath_evaluation" / "comparison.json", result)
    return result
