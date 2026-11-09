"""Immutable experiment recipe and equal-budget schedule; no GPU imports."""

from pathlib import Path
from . import ARMS, DOMAINS
from d4_teachers.common import digest, need, read


def load_spec(path=None):
    path = path or Path(__file__).resolve().parents[1] / "config/mopd/baseline.json"
    return validate_spec(read(path))


def validate_spec(s):
    need(s["schema"] == "d4-plain-mopd-v1", "Unknown MOPD recipe")
    need(s["rollout_n"] == 1, "This baseline requires rollout_n=1")
    for key in ("train_batch_size", "updates", "save_every", "validate_every"):
        need(isinstance(s[key], int) and s[key] > 0 and s[key] % 4 == 0,
             key + " must be a positive multiple of four")
    need(s["loss_reduction"] == "seq-mean-token-mean", "Unexpected loss reduction")
    need(s["pg_signal_clip"] == 5.0, "Paper PG signal clipping must be +/-5")
    need(s["max_context_tokens"] == 65536, "Frozen project prompt budgets require 64K")
    need(s["learning_rate"] > 0 and s["weight_decay"] >= 0, "Invalid optimizer")
    need(s["student_nodes"] == s["teacher_nodes"] == 1 and s["gpus_per_node"] == 8,
         "Qualified topology is one 8-GPU student node plus one 8-GPU teacher node")
    need((s["actor_tp"], s["actor_pp"], s["actor_ep"], s["rollout_tp"], s["teacher_tp"])
         == (2, 2, 2, 4, 2), "Topology changes require a new qualification")
    # EP shares the model's data-parallel groups; it is not another factor in TP*PP*DP.
    need(0 < s["gpu_memory_utilization"] < 1, "Invalid memory fraction")
    return s


def plan_indices(rows, spec, arm):
    """Same per-domain shuffled streams and multiset in every four-update cycle.

    Domain-only rotates domain order each cycle (a four-cycle Latin rotation).
    It is homogeneous at optimizer-update granularity, not just microbatch granularity.
    """
    import random
    need(arm in ARMS, "Unknown batching arm")
    groups = {d: [i for i, r in enumerate(rows) if r["domain"] == d] for d in DOMAINS}
    need(all(groups.values()), "Every domain must contain training prompts")
    need(sum(map(len, groups.values())) == len(rows), "Unknown training domain")
    streams = {}
    for d, indices in groups.items():
        rng = random.Random(int(digest([spec["seed"], d])[:16], 16))
        stream = []
        while len(stream) < spec["updates"] // 4 * spec["train_batch_size"]:
            block = indices.copy()
            rng.shuffle(block)
            stream.extend(block)
        streams[d] = stream
    b, output = spec["train_batch_size"], []
    for cycle in range(spec["updates"] // 4):
        chunk = {d: streams[d][cycle*b:(cycle+1)*b] for d in DOMAINS}
        if arm == "domain_mix":
            for j in range(4):
                batch = [idx for d in DOMAINS for idx in chunk[d][j*b//4:(j+1)*b//4]]
                random.Random(spec["seed"] + cycle*4 + j).shuffle(batch)
                output.extend(batch)
        else:
            order = DOMAINS[cycle % 4:] + DOMAINS[:cycle % 4]
            for d in order:
                output.extend(chunk[d])
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
