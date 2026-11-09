from collections import Counter
import copy
import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from d4_mopd import DOMAINS
from d4_mopd.spec import PlannedSampler, plan_indices, validate_spec


def test_equal_prompt_exposure_and_update_composition(rows, spec):
    spec = dict(spec, updates=16, train_batch_size=20)
    mix = plan_indices(rows, spec, "domain_mix")
    only = plan_indices(rows, spec, "domain_only")
    for start in range(0, len(mix), 80):
        assert Counter(mix[start:start+80]) == Counter(only[start:start+80])
    for step, offset in enumerate(range(0, len(mix), 20)):
        assert Counter(rows[i]["domain"] for i in mix[offset:offset+20]) == {d: 5 for d in DOMAINS}
        target = DOMAINS[(step//4 + step%4) % 4]
        assert Counter(rows[i]["domain"] for i in only[offset:offset+20]) == {target: 20}
    assert plan_indices(rows, spec, "domain_mix") == mix


@pytest.mark.parametrize("arm", ["domain_mix", "domain_only"])
@pytest.mark.parametrize("checkpoint_batch", [1, 4, 7])
def test_actual_torchdata_resume_no_repeat_or_skip(rows, spec, arm, checkpoint_batch):
    spec = dict(spec, updates=8, train_batch_size=8)
    def make():
        return StatefulDataLoader(list(range(len(rows))), batch_size=8, drop_last=True, num_workers=0,
                                  sampler=PlannedSampler(rows, spec, arm))
    uninterrupted = [b.tolist() for b in make()]
    loader = make()
    iterator = iter(loader)
    prefix = [next(iterator).tolist() for _ in range(checkpoint_batch)]
    state = copy.deepcopy(loader.state_dict())
    restarted = make()
    restarted.load_state_dict(state)
    assert prefix + [b.tolist() for b in restarted] == uninterrupted


def test_changed_dataset_or_arm_refuses_resume(rows, spec):
    sampler = PlannedSampler(rows, spec, "domain_mix")
    next(iter(sampler))
    state = sampler.state_dict()
    for changed_rows, arm in ((rows[::-1], "domain_mix"), (rows, "domain_only")):
        with pytest.raises(RuntimeError, match="changed"):
            PlannedSampler(changed_rows, spec, arm).load_state_dict(state)


@pytest.mark.parametrize("patch", [{"rollout_n": 8}, {"updates": 201}, {"train_batch_size": 15}, {"pg_signal_clip": 10}])
def test_invalid_baseline_rejected(spec, patch):
    with pytest.raises(RuntimeError):
        validate_spec(dict(spec, **patch))
