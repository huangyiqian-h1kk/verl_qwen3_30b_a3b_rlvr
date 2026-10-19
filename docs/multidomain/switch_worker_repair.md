# Repair candidate worker construction

Both the random and domain two-node candidates failed in the pinned verl
`create_colocated_worker_cls`, before restoring a checkpoint or training an update:

```text
TypeError: ActorRolloutRefWorker.__init__() missing 2 required positional arguments: 'config' and 'role'
```

The pinned upstream version is
[`62ac6d6e9fccc09ab6af2171217cee6e290b799b`](https://github.com/verl-project/verl/blob/62ac6d6e9fccc09ab6af2171217cee6e290b799b/verl/single_controller/ray/base.py).
Its `_determine_fsdp_megatron_base_class` walks the first actor's MRO and returns
the first class whose **name** is `Worker` or `MegatronWorker`. Both candidate
packages define `class Worker(TwoNodeWorker)`. That name causes verl to select
the application actor as the container base, so `WorkerDict.__init__` calls the
actor initializer without its required arguments. The public check uses
`TwoNodeWorker`, which has no such name collision. Inheritance depth alone is
not the problem.

The repair changes the definitions to `RandomSwitchWorker` and
`DomainSwitchWorker`, respectively, and retains `Worker = RandomSwitchWorker`
or `Worker = DomainSwitchWorker` as the module import interface. An alias does
not change `__name__`. Python still compiles each method's `super()` closure
against the correct class. No initializer defaults, shared verl edits, model
configuration changes, checkpoint conversion or verifier changes are needed.

## Apply from Git without changing the dirty cluster checkout

The deployment has controller and job snapshots that are newer than the remote
research branch. The repair entry updates only the two existing controller
snapshots. It does not check out the older Git tree over the working project.
It first verifies known controller implementations, the exact failed job IDs,
PBS exit 1 in `R9920261000`, the matching constructor traceback, unchanged
snapshots and common check evidence, and absence of handoff/progress or
ambiguous submissions. An unavailable PBS history is a blocking observation;
the repair does not assume that an absent job has finished.

Fetch the repair branch and use its reviewed commit SHA:

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d
git fetch --no-tags https://github.com/huangyiqian-h1kk/verl_qwen3_30b_a3b_rlvr.git \
  refs/heads/codex/fix-switch-worker-name
```

The entry is standalone, so no ZIP, extraction, checkout or worktree overwrite is
needed. Use the failed candidate IDs being repaired, not the running single-node
job IDs. Omit `--apply` for a read-only preflight:

```bash
set -o pipefail
git show REPAIR_COMMIT:scripts/multidomain/repair_switch_worker.py |
  ./.venv-multidomain/bin/python -I - --root "$PWD" \
    --random-job RANDOM_FAILED_CANDIDATE --domain-job DOMAIN_FAILED_CANDIDATE --apply
```

The entry cooperatively pauses the affected login-node controllers, holds their
locks, verifies the final state again, and journals the update under each run's
`handoff/worker_name_repair/`. It updates the worker hash, candidate contract
identity, state identity and the existing limits adapter's bound hashes together.
The old candidate snapshot, selected checkpoint and failure log remain unchanged.
An interrupted publication is recovered from the journal before retrying; an
independent file change is never overwritten.

After publication, only the matching blocked failure is rearmed to
`WAIT_CHECKPOINT`. Original controllers resume for 30 hours, use their normal
duplicate and pending-submission protections, and submit new candidates to the
reserved queue. Future random training retains the existing 24-hour walltime
adapter; domain training keeps its existing 24-hour request. Submission is
asynchronous, so an immediate `qstat` may precede the new job IDs.

Repeated execution recognizes the repair receipt and preserves any newer
candidate or completed handoff. It does not reset a running replacement.
Unknown exit-1 failures remain blocked by the original monitor policy.

The next candidates still have to pass their original GPU acceptance gates.
The domain candidate must perform the 8-to-16 restore, original sampler's next
update and complete save, then a fresh-process 16-to-16 restore and subsequent
update with sampling continuity. Single-node cancellation and fallback retention
remain controlled by the existing handoff evidence rules.

## Validation and limits

```bash
python -m unittest discover -s tests/multidomain -p test_switch_worker_repair.py -v
```

CPU tests execute the actual pinned upstream factory with stand-ins for Ray and
model initialization. They reproduce the old error for both candidate classes,
show the common check path succeeding, and verify argument forwarding, RPC
metadata and `super()` after the rename. On-disk migration tests cover contract
rebinding, preserved checkpoints/history, partial-write recovery, idempotence,
and rejection of active jobs, other queues/errors, pending submissions, changed
evidence and handoff progress. The upstream excerpt retains its Apache-2.0 notice.

These tests establish the constructor fix and migration behavior. They do not
claim a real Ray/GPU run or a completed single-to-dual training handoff. Those
remain to be verified by the new candidate jobs on ABCI.
