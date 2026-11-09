"""Immutable recipe and reproducible domain sampling; no GPU imports."""

from pathlib import Path
from . import ARMS, DOMAINS
from d4_teachers.common import digest, need, read


def load_spec(path=None):
    path = path or Path(__file__).resolve().parents[1] / "config/mopd/baseline.json"
    return validate_spec(read(path))


def validate_spec(s):
    need(s["schema"] == "d4-plain-mopd-v1", "Unknown MOPD recipe")
    need(s["rollout_n"] == 1, "This baseline requires rollout_n=1")
    need(s["domain_only_schedule"] == "iid_uniform", "Domain-only requires independent uniform draws")
    need(s["selection_score"] == "four_domain_macro_math_source_equal", "Unexpected selection score")
    for key in ("train_batch_size", "updates", "save_every", "validate_every"):
        need(type(s[key]) is int and s[key] > 0, key + " must be a positive integer")
    need(s["train_batch_size"] % 4 == 0, "Mixed batch size must be divisible by four")
    need(type(s["validation_per_domain"]) is int and s["validation_per_domain"] > 0
         and s["validation_per_domain"] % 2 == 0,
         "Validation size per domain must be positive and even for 1:1 Math sources")
    need(s["loss_reduction"] == "seq-mean-token-mean", "Unexpected loss reduction")
    need(s["pg_signal_clip"] == 5.0, "Paper PG signal clipping must be +/-5")
    need(s["max_context_tokens"] == 65536, "Frozen project prompt budgets require 64K")
    need(s["learning_rate"] > 0 and s["weight_decay"] >= 0, "Invalid optimizer")
    need(s["adam_epsilon"] > 0 and len(s["adam_betas"]) == 2
         and all(0 <= b < 1 for b in s["adam_betas"]), "Invalid Adam parameters")
    need(s["validate_every"] % s["save_every"] == 0,
         "Every scheduled validation must have a complete checkpoint")
    need(type(s["checkpoint_keep_recent"]) is int and s["checkpoint_keep_recent"] >= 2,
         "Keep at least two complete checkpoints for recovery, plus the validation best")
    need(s["student_nodes"] == s["teacher_nodes"] == 1 and s["gpus_per_node"] == 8,
         "Qualified topology is one 8-GPU student node plus one 8-GPU teacher node")
    need((s["actor_tp"], s["actor_pp"], s["actor_ep"], s["rollout_tp"], s["teacher_tp"])
         == (2, 2, 2, 4, 2), "Topology changes require a new qualification")
    # EP shares the model's data-parallel groups; it is not another factor in TP*PP*DP.
    need(0 < s["gpu_memory_utilization"] < 1, "Invalid memory fraction")
    return s


def domain_order(spec):
    """One independent uniform draw per update, separate from all other RNGs."""
    import random
    rng = random.Random(int(digest([spec["seed"], "domain_only_iid_uniform_v1"])[:16], 16))
    return [rng.choice(DOMAINS) for _ in range(spec["updates"])]


def plan_indices(rows, spec, arm):
    """Both arms use the same shuffled stream within each domain.

    Mix consumes equal quotas each update; only chooses a domain independently.
    Only has equal domain exposure in expectation, not in every four-update block.
    """
    from collections import Counter
    import random
    need(arm in ARMS, "Unknown batching arm")
    groups = {d: [i for i, r in enumerate(rows) if r["domain"] == d] for d in DOMAINS}
    need(all(groups.values()), "Every domain must contain training prompts")
    need(sum(map(len, groups.values())) == len(rows), "Unknown training domain")
    order = domain_order(spec)
    counts = Counter(order)
    b = spec["train_batch_size"]
    streams = {}
    for d, indices in groups.items():
        rng = random.Random(int(digest([spec["seed"], d])[:16], 16))
        stream = []
        required = spec["updates"] * b // 4 if arm == "domain_mix" else counts[d] * b
        while len(stream) < required:
            block = indices.copy()
            rng.shuffle(block)
            stream.extend(block)
        streams[d] = stream
    output, cursor = [], dict.fromkeys(DOMAINS, 0)
    for step in range(spec["updates"]):
        if arm == "domain_mix":
            batch = [idx for d in DOMAINS for idx in streams[d][cursor[d]:cursor[d]+b//4]]
            random.Random(spec["seed"] + step).shuffle(batch)
            output.extend(batch)
            for d in DOMAINS:
                cursor[d] += b//4
        else:
            d = order[step]
            output.extend(streams[d][cursor[d]:cursor[d]+b])
            cursor[d] += b
    return output


class PlannedSampler:
    """torchdata stateful protocol; cursor is saved with the DataLoader state."""
    def __init__(self, rows, spec, arm):
        self.indices = plan_indices(rows, spec, arm)
        self.identity = digest([spec, arm, [r["sample_id"] for r in rows], self.indices])
        self.cursor = 0

    def __iter__(self):
        while self.cursor < len(self.indices):
            index = self.indices[self.cursor]
            self.cursor += 1
            yield index

    def __len__(self):
        return len(self.indices)

    def state_dict(self):
        return {"identity": self.identity, "cursor": self.cursor}

    def load_state_dict(self, state):
        need(state["identity"] == self.identity, "Sampler/data/recipe changed on resume")
        need(0 <= state["cursor"] <= len(self.indices), "Invalid sampler cursor")
        self.cursor = state["cursor"]
