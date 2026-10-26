"""Finite deterministic shuffled stream; only the StatefulDataLoader cursor mutates."""

from dataclasses import dataclass
import random
from .common import digest, need


@dataclass(frozen=True)
class Plan:
    indices: tuple
    domain: str
    seed: int
    batch_size: int
    row_fingerprint: str

    def batch(self, step):
        need(
            1 <= step <= len(self.indices) // self.batch_size,
            "Sampling step outside budget",
        )
        return self.indices[(step - 1) * self.batch_size : step * self.batch_size]

    def metadata(self):
        return dict(
            version="teacher-shuffled-pools-v1",
            domain=self.domain,
            seed=self.seed,
            batch_size=self.batch_size,
            total_steps=len(self.indices) // self.batch_size,
            row_fingerprint=self.row_fingerprint,
            plan_sha256=digest(self.indices),
            tail_policy="discard a short tail, reshuffle, then draw a full unique batch",
        )


def build_plan(rows, domain, seed=42, batch_size=256, total_steps=200):
    need(
        len(rows) >= batch_size > 0 and total_steps > 0,
        "Insufficient teacher data/budget",
    )
    ids = [r["sample_id"] for r in rows]
    need(
        len(set(ids)) == len(ids) and all(r["domain"] == domain for r in rows),
        "Duplicate IDs or foreign domain in teacher sampler",
    )
    rng = random.Random(int(digest(["domain-pool", seed, domain]), 16))
    pool, cursor, stream = [], 0, []
    for _ in range(total_steps):
        if len(pool) - cursor < batch_size:
            pool = list(range(len(rows)))
            rng.shuffle(pool)
            cursor = 0
        stream.extend(pool[cursor : cursor + batch_size])
        cursor += batch_size
    return Plan(
        tuple(stream),
        domain,
        seed,
        batch_size,
        digest([(r["sample_id"], r["domain"]) for r in rows]),
    )


class PlannedDataset:
    def __init__(self, original, plan):
        self.original, self.plan = original, plan

    def __len__(self):
        return len(self.plan.indices)

    def __getitem__(self, index):
        return self.original[self.plan.indices[index]]

    def resume_dataset_state(self):
        if hasattr(self.original, "resume_dataset_state"):
            self.original.resume_dataset_state()


def verify_batch(batch, domain, ids, repeats):
    from collections import Counter

    data = batch.non_tensor_batch
    got = [str(x["sample_id"]) for x in data["extra_info"]]
    need(
        set(data["data_source"]) == {domain}
        and Counter(got) == Counter({i: repeats for i in ids}),
        "Actual GRPO batch differs from the frozen teacher sampling plan",
    )


def loader_restart_check():
    """Exercise real torchdata save/load on CPU inside each formal allocation."""
    import io
    import torch
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    rows = [dict(sample_id=str(i), domain="math") for i in range(23)]
    plan = build_plan(rows, "math", batch_size=4, total_steps=60)

    def loader():
        ds = PlannedDataset(list(range(len(rows))), plan)
        return StatefulDataLoader(
            ds,
            batch_size=4,
            sampler=SequentialSampler(ds),
            num_workers=0,
            drop_last=True,
        )

    original = loader()
    it = iter(original)
    for step in range(1, 60):
        need(
            next(it).tolist() == list(plan.batch(step)), "Loader changed sampling order"
        )
        if step in (1, 5, 10, 39, 40, 50, 59):
            buffer = io.BytesIO()
            torch.save(original.state_dict(), buffer)
            buffer.seek(0)
            restored = loader()
            restored.load_state_dict(torch.load(buffer, weights_only=False))
            need(
                next(iter(restored)).tolist() == list(plan.batch(step + 1)),
                "Loader repeated/skipped a batch",
            )
    return dict(status="PASS", scope="real CPU torchdata resume; no pilot training")
