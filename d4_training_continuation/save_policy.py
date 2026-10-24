"""The established four-domain persistence policy, separate from evaluation."""
import copy
from .common import need

POLICY = dict(version="2026-10-22.save5-eval25.1", save_freq=5, test_freq=25,
              resume_evaluation="complete-or-replay-before-next-update")


def training_config(config):
    result = copy.deepcopy(config)
    training = result["training"]
    need(training.get("save_freq") in (5, 25) and training.get("test_freq") == 25,
         "Unexpected checkpoint/evaluation policy; inspect before normalizing")
    training["save_freq"] = POLICY["save_freq"]
    return result


def check_request(request):
    policy = request.get("training_policy")
    if policy is None:  # Existing immutable jobs remain readable.
        return False
    need(policy == POLICY and policy == request["contract"].get("training_policy"),
         "Training persistence policy changed")
    training = request["config"]["training"]
    need(all(training[k] == POLICY[k] for k in ("save_freq", "test_freq")),
         "Effective save/evaluation interval differs from the bound policy")
    return True
