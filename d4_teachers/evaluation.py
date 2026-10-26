"""Manifest-sized, checkpoint-bound evaluation with completed-batch recovery."""

from collections import defaultdict
import math
from pathlib import Path
import time
import uuid
from .common import digest, file_sha, need, now, read, write
from . import selection
from d4_training_continuation.validation_journal import (
    Journal as BaseJournal,
    instrument,
    plain,
    dependency_versions,
)


class Journal(BaseJournal):
    def progress(self):
        write(
            self.directory / "progress.json",
            dict(
                completed_batches=self.index,
                completed_rows=self.rows,
                cached_batches_this_process=self.cached,
                expected_rows=self.identity["runtime"]["rows"],
                at_utc=now(),
            ),
        )

    def finish(self):
        runtime = self.identity["runtime"]
        need(
            self.rows == runtime["rows"] and self.index == runtime["batches"],
            "Evaluation incomplete",
        )
        ids, samples, checksums = [], [], {}
        for i in range(1, self.index + 1):
            path = self.directory / f"batch_{i:06d}.json"
            env = read(path)
            need(
                env["sha256"] == digest(env["payload"]),
                "Evaluation batch checksum changed",
            )
            self.check(env["payload"])
            values = env["payload"]["lists"]
            ids.extend(values["sample_uids"])
            samples.extend(zip(values["sample_uids"], values["sample_scores"]))
            checksums[path.name] = file_sha(path)
        need(
            ids == runtime["ids"] and len(set(ids)) == len(ids),
            "Evaluation IDs/order differ from frozen split",
        )
        return dict(
            path=str(self.directory),
            rows=self.rows,
            batches=self.index,
            reused_batches=self.cached,
            manifest_sha256=digest(self.identity),
            batch_files=checksums,
        ), samples


def read_receipt(path):
    receipt = read(path)
    need(
        receipt["content_sha256"]
        == digest({k: v for k, v in receipt.items() if k != "content_sha256"}),
        "Evaluation receipt checksum changed",
    )
    proof = receipt["journal"]
    directory = Path(proof["path"])
    need(
        digest(read(directory / "manifest.json")) == proof["manifest_sha256"],
        "Evaluation manifest changed",
    )
    for name, expected in proof["batch_files"].items():
        need(file_sha(directory / name) == expected, "Evaluation batch changed")
    return receipt


def statistics(samples, rows):
    lookup = {r["sample_id"]: r for r in rows}
    groups = defaultdict(list)
    for sid, score in samples:
        score = float(score)
        need(math.isfinite(score) and 0 <= score <= 1, "Invalid local verifier reward")
        r = lookup[sid]
        groups[r["domain"]].append(score)
        for key in ("source", "category", "task"):
            groups[r["domain"] + "/" + key + "/" + str(r.get(key, ""))].append(score)
    return {k: dict(n=len(v), mean=sum(v) / len(v)) for k, v in sorted(groups.items())}


def validate(trainer, original, merged=False):
    req, cfg = trainer.teacher_request, trainer.config
    split = "test" if req["role"] == "test" else "validation"
    step = trainer.global_steps
    cp = trainer.current_checkpoint(step)
    from omegaconf import OmegaConf

    kw = cfg.actor_rollout_ref.rollout.val_kwargs
    need(
        kw.n == 1
        and not kw.do_sample
        and kw.temperature == 0
        and not cfg.data.validation_shuffle,
        "Evaluation must remain ordered greedy n=1",
    )
    runtime = dict(
        rows=len(trainer.val_dataset),
        batches=len(trainer.val_dataloader),
        ids=[r["sample_id"] for r in trainer.val_dataset.rows],
        dataset_sha256=digest(plain(trainer.val_dataset.rows)),
        dependencies=dependency_versions(),
        decoding=OmegaConf.to_container(kw, resolve=True),
        batch_size=cfg.data.val_batch_size,
        context=req["config"]["data"],
    )
    binding = dict(
        campaign=req["campaign_identity"],
        domain=req["domain"],
        split=split,
        step=step,
        weights=trainer.weights_identity(cp),
        verifier=req["verifier_identity"],
    )
    key = digest(dict(binding=binding, runtime=runtime))
    directory = (
        Path(req["run_dir"]) / "evaluations" / (split + "_" + str(step) + "_" + key)
    )
    receipt_path = directory / "receipt.json"
    trainer.cont_journal = Journal(directory, binding, runtime)
    if receipt_path.exists():
        receipt = read_receipt(receipt_path)
        need(
            receipt["identity"] == key and receipt["binding"] == binding,
            "Evaluation receipt changed",
        )
        proof = receipt["journal"]
        need(
            proof["manifest_sha256"] == digest(trainer.cont_journal.identity),
            "Journal manifest changed",
        )
        for name, expected in proof["batch_files"].items():
            need(
                file_sha(directory / name) == expected,
                "Completed evaluation output changed",
            )
    else:
        start = time.monotonic()
        attempt = uuid.uuid4().hex
        completed = False
        previous = cfg.trainer.validation_data_dir
        cfg.trainer.validation_data_dir = str(directory / "outputs")
        try:
            metrics = plain(instrument(original)(merged=merged))
            for future in getattr(trainer, "_dump_futures", []):
                future.result()
            proof, samples = trainer.cont_journal.finish()
            receipt = dict(
                status="PASS",
                identity=key,
                binding=binding,
                split=split,
                step=step,
                path=str(receipt_path),
                metrics=metrics,
                scores=statistics(samples, trainer.val_dataset.rows),
                journal=proof,
                elapsed_seconds=time.monotonic() - start,
                at_utc=now(),
            )
            receipt["content_sha256"] = digest(receipt)
            write(receipt_path, receipt)
            completed = True
        finally:
            cfg.trainer.validation_data_dir = previous
            write(
                directory / "attempts" / (attempt + ".json"),
                dict(
                    completed=completed,
                    seconds=time.monotonic() - start,
                    job=req["job_dir"],
                    at_utc=now(),
                ),
            )
    if req["role"] == "train":
        best = selection.select(req, cp, receipt)
        trainer.teacher_stop = selection.should_stop(step, best["step"], req["policy"])
        write(
            Path(req["run_dir"]) / "validation_state.json",
            dict(
                step=step,
                best_step=best["step"],
                best_score=best["score"],
                early_stop=trainer.teacher_stop,
                receipt=str(receipt_path),
            ),
        )
    else:
        write(Path(req["run_dir"]) / "test.json", receipt)
    return receipt["metrics"]
