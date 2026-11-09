"""Permit the upstream unclipped PG loss with upstream distillation's k1 signal.

Pinned verl already dispatches arbitrary registered policy losses, but its config
constructor has a vanilla-only guard. This narrow config subclass relaxes that
guard for exactly one baseline; it changes neither the loss nor its backward.
"""
from dataclasses import dataclass
from verl.workers.config.distillation import DistillationLossConfig


@dataclass
class BasicPGConfig(DistillationLossConfig):
    def __post_init__(self):
        if not (self.loss_mode == "k1" and self.use_policy_gradient
                and self.policy_loss_mode == "gpg" and not self.use_task_rewards
                and self.loss_max_clamp == 5.0 and self.log_prob_min_clamp is None):
            raise ValueError("BasicPGConfig only supports the fixed plain MOPD recipe")
        object.__setattr__(self, "policy_loss_mode", "vanilla")
        super().__post_init__()
        object.__setattr__(self, "policy_loss_mode", "gpg")
