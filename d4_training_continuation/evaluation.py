"""Checkpoint-bound validation receipts; interrupted validation is replayed.

Receipts live outside immutable checkpoint inventories. Legacy completion needs
both the four-domain final metrics line and all 1,006 persisted validation rows.
Neither an existing output file nor a completed optimizer step proves evaluation.
"""
import json
import math
from pathlib import Path
import re
import uuid
from . import checkpoints as ck
from .common import below, digest, locked, need, now, read, sha, write

VERSION = "checkpoint-validation-v1"
DOMAINS = ("math", "if", "conversational_pivot", "logic_algorithmic")
CORE = tuple("val-core/" + d + "/acc/mean@1" for d in DOMAINS)
ROWS = 1006
NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SCALAR = re.compile(r"\b(?:np|numpy)\.(?:float(?:16|32|64|128)?|u?int(?:8|16|32|64)?)\(\s*(" + NUMBER + r")\s*\)", re.I)
METRIC = re.compile(r"(?:^|\s+-\s+)([^:\s][^:]*?):\s*(" + NUMBER + r")(?=\s+-\s+|\s*$|\s*\[repeated)", re.I)


def due(step, frequency):
    return step > 0 and frequency > 0 and step % frequency == 0


def checkpoint_key(record):
    return digest({k: record[k] for k in ("lineage", "step", "world_size", "files")})


def protocol(request):
    if request.get("validation_semantics"):
        return digest(request["validation_semantics"])
    config = request["config"]
    # Allocation/save frequency are deliberately absent; validation semantics
    # and the actual frozen reward source must still agree across attempts.
    reward = {p: v for p, v in request["contract"]["snapshot_files"].items()
              if p.startswith("md_reward") and p.endswith(".py")}
    return digest(dict(kind=request["kind"], data_dir=request["data_dir"],
                       config={k: config.get(k) for k in ("data", "domains", "verifiers", "seed", "verl_commit")},
                       validation={k: config["training"].get(k) for k in ("model_path", "val_n", "val_temperature")},
                       reward=reward))


def semantics(request):
    """Stable scoring identity across frozen job directories, not across setups."""
    config = request["config"]
    folder = Path(request["job_dir"])
    reward = {}
    for relative, checksum in request["contract"]["snapshot_files"].items():
        if relative.startswith("md_reward") and relative.endswith(".py"):
            source = folder / relative
            need(sha(source) == checksum, "Reward snapshot changed")
            # Only the bound diagnostics directory changes on each new job.
            reward[relative] = digest(source.read_text().replace(str(folder / "math_diagnostics"), "<diagnostics>"))
        elif relative.startswith("runtime/d4_db_mathfix/"):
            reward[relative] = checksum
    need(reward, "Missing bound verifier snapshot")
    return dict(kind=request["kind"], data_dir=request["data_dir"],
                validation_parquet_sha256=sha(Path(request["data_dir"]) / "validation.parquet"),
                config={k: config.get(k) for k in ("data", "domains", "verifiers", "seed", "verl_commit")},
                validation={k: config["training"].get(k) for k in ("model_path", "val_n", "val_temperature")},
                reward=reward, infrastructure={p: v for p, v in request["contract"]["root_files"].items()
                    if p.startswith(("multidomain/", "config/multidomain/"))})


def receipt_path(request, record):
    return Path(request["control_dir"]) / "evaluations" / (checkpoint_key(record) + "_" + protocol(request) + ".json")


def numeric_metrics(metrics):
    result = {k: float(v) for k, v in metrics.items() if k.startswith(("val-core/", "val-aux/"))}
    need(all(k in result for k in CORE) and all(math.isfinite(v) for v in result.values()),
         "Incomplete/non-finite four-domain evaluation metrics")
    return result


def outputs(path, step):
    path = Path(path)
    count = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            need(row.get("step") == step and isinstance(row.get("input"), str)
                 and isinstance(row.get("output"), str) and math.isfinite(float(row["score"])),
                 "Incomplete validation output row")
            count += 1
    need(count == ROWS, "Incomplete validation output count: " + str(count))
    return dict(path=str(path), rows=count, sha256=sha(path))


def completed(request, record, runtime=None):
    path = receipt_path(request, record)
    if not path.exists():
        return None
    receipt = read(path)
    need(receipt["status"] == "PASS" and receipt["version"] == VERSION
         and receipt["checkpoint_key"] == checkpoint_key(record) and receipt["protocol"] == protocol(request),
         "Evaluation receipt belongs to another checkpoint/protocol")
    numeric_metrics(receipt["metrics"])
    artifact = receipt["outputs"]
    need(outputs(artifact["path"], record["step"]) == artifact, "Completed evaluation output changed")
    proof = receipt.get("origin", {}).get("journal")
    if proof:
        manifest = read(Path(proof["path"]) / "manifest.json")
        need(digest(manifest) == proof["manifest_sha256"], "Validation manifest changed")
        if runtime is not None:
            need(manifest["runtime"] == runtime, "Completed validation runtime/dataset changed")
    return receipt


def publish(request, record, metrics, output, origin):
    ck.validate(record)
    receipt = dict(version=VERSION, status="PASS", checkpoint_key=checkpoint_key(record),
                   step=record["step"], lineage=record["lineage"], protocol=protocol(request),
                   metrics=numeric_metrics(metrics), outputs=outputs(output, record["step"]),
                   origin=origin, at_utc=now())
    write(receipt_path(request, record), receipt)
    return receipt


def import_legacy(job, record, probe=False):
    """Called only after the old writer is fenced; absence means replay, not PASS."""
    if not job:
        return False
    folder = Path(job["folder"])
    request = read(folder / "request.json")
    if not due(record["step"], request.get("config", {}).get("training", {}).get("test_freq", 0)):
        return False
    if not record.get("lineage") or request.get("lineage") != record["lineage"]:
        return False
    if Path(request["run_dir"]).resolve() not in Path(record["path"]).resolve().parents:
        return False  # A process killed before its first save still uses its bootstrap pin.
    need(request["identity"] == digest(request["contract"])
         and request["config"] == request["contract"]["config"], "Legacy evaluation request changed")
    below(record["path"], Path(request["run_dir"]) / "checkpoints")
    previous = completed(request, record)
    if previous:
        if not probe and not request.get("validation_semantics") and request["contract"].get("root_files"):
            publish(dict(request, validation_semantics=semantics(request)), record, previous["metrics"],
                    previous["outputs"]["path"], previous["origin"])
        return True
    output = Path(request["run_dir"]) / "generations/train" / (str(record["step"]) + ".jsonl")
    log = folder / "pbs.log"
    if not output.exists() or not log.exists():
        return False
    metrics, evidence = None, None
    with log.open(errors="replace") as handle:
        for line in handle:
            text = SCALAR.sub(lambda m: m[1], ANSI.sub("", line)).strip()
            match = re.search(r"\bstep:\s*(\d+)\s+-\s+", text)
            if not match or int(match[1]) != record["step"]:
                continue
            parsed = {m[1].strip(): float(m[2]) for m in METRIC.finditer(text[match.end():])}
            if all(k in parsed for k in CORE) and parsed.get("timing_s/testing", 0) > 0:
                metrics = parsed
                evidence = digest(line)
    if metrics is None:
        return False
    try:
        outputs(output, record["step"])
        numeric_metrics(metrics)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    if not probe:
        publish(request, record, metrics, output, dict(mode="legacy-complete", job=job,
                log=str(log), metrics_line_digest=evidence))
        if request["contract"].get("root_files"):
            # Bridge an old completed evaluation to the path-independent
            # semantics used by new jobs. No incomplete output is imported.
            bound = dict(request, validation_semantics=semantics(request))
            publish(bound, record, metrics, output, dict(mode="legacy-complete", job=job,
                    log=str(log), metrics_line_digest=evidence))
    return True


def validate(trainer, evaluate, merged=False):
    """Run on restored weights before the next optimizer update, or reuse proof."""
    need(not merged, "Continuation validation receipts require unmerged metrics")
    request, step = trainer.cont_req, trainer.global_steps
    record = trainer.cont_resume if step == trainer.cont_resume["step"] else ck.latest(request["run_dir"], request["lineage"])
    need(record is not None and record["step"] == step, "Evaluation has no matching full checkpoint")
    ck.validate(record)
    actual_runtime = None
    if request.get("validation_policy"):
        from .validation_journal import POLICY, runtime_contract
        need(request["validation_policy"] == POLICY, "Unknown validation persistence policy")
        actual_runtime = runtime_contract(trainer)
    receipt = completed(request, record, runtime=actual_runtime)
    reused = receipt is not None
    if receipt is None:
        output = Path(trainer.config.trainer.validation_data_dir) / (str(step) + ".jsonl")
        if output.exists():
            archive = output.parent / "replay_archive"
            archive.mkdir(exist_ok=True)
            output.rename(archive / (output.stem + "_" + uuid.uuid4().hex + ".jsonl"))
        journal_proof = None
        if request.get("validation_policy"):
            from .validation_journal import Journal, instrument
            directory = Path(request["control_dir"]) / "validation_batches" / (
                checkpoint_key(record) + "_" + protocol(request))
            with locked(directory / "writer.lock"):
                trainer.cont_journal = Journal(directory, dict(checkpoint=checkpoint_key(record),
                    protocol=protocol(request)), actual_runtime)
                try:
                    metrics = instrument(evaluate)(merged=False)
                    journal_proof = trainer.cont_journal.finish()
                finally:
                    del trainer.cont_journal
        else:
            metrics = evaluate(merged=False)
        # Upstream dumps validation asynchronously. Never publish PASS before
        # every output write finishes and the full 1,006-row artifact is checked.
        for future in trainer._dump_futures:
            future.result()
        trainer._dump_futures.clear()
        need(trainer.global_steps == step, "Validation advanced the training step")
        receipt = publish(request, record, metrics, output,
                          dict(mode="evaluation", job_dir=request["job_dir"], phase=trainer.cont_phase.name,
                               journal=journal_proof))
    write(trainer.cont_phase / ("validation_" + str(step) + ".json"),
          dict(status="PASS", step=step, reused=reused, receipt=str(receipt_path(request, record)),
               receipt_sha256=sha(receipt_path(request, record)), metrics=receipt["metrics"], at_utc=now()))
    return receipt["metrics"]
