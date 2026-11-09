"""Use native student generation and native teacher scoring, not teacher rollouts."""
from multidomain.agent_loop import MultiDomainAgentLoop


class MOPDAgentLoop(MultiDomainAgentLoop):
    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        if not output.response_ids:
            raise RuntimeError("Empty response: refuse silently dropping a prompt from the loss")
        if kwargs["mopd_split"] == "train":
            output.reward_score = 0.0  # no verifier calls or RL rewards in training
            output.extra_fields["reward_extra_info"] = {"task_reward_disabled": 1.0}
        return output
