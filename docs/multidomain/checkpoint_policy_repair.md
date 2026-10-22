# Random continuation: restore save5/eval25 and resume interrupted validation

The legacy random request contained `save_freq=25`, but its recovery launcher
overrode the effective setting to 5. Continuation copied the request instead of
the effective setting. Both continuation singles and duals consequently saved
every 25 updates. Domain requests already contained 5.

New continuation requests now bind `save_freq=5`, `test_freq=25` and the replay
policy in their immutable contract. The original request is retained verbatim
for provenance. Dataset, sampler, batch/rollout sizes, optimizer/LR schedule,
reward implementation and topology qualification are preserved. Unknown source
intervals are rejected rather than silently normalized. The composed GPU runtime
also checks the two effective intervals before creating a model.

## Deployment

Use the reviewed commit on `codex/d4-training-continuation` in the existing
detached operations worktree. Do not change the experiment-root checkout or a
submitted job's files. Run on the current random controller's login host (the
latest supplied snapshot identifies `login3`).

```bash
"$D4_PY" -I "$D4_SCRIPT" self-test
"$D4_PY" -I "$D4_SCRIPT" repair-save-policy --root "$D4_ROOT" --kind random
```

This default updates/restarts only the random login-node controller. Current PBS
training keeps its already-loaded interval until its normal termination. The
next eligible retry uses the corrected job snapshot. Domain's controller/jobs,
deadlines, promotion state and existing opportunity policy are unchanged.

To request one controlled restart after the current writer has published a full
checkpoint at step 125 or later, use the following instead (or add it after the
default repair):

```bash
"$D4_PY" -I "$D4_SCRIPT" repair-save-policy --root "$D4_ROOT" --kind random --restart-after-step 125
```

The command returns after controller adoption; the background controller waits
for the checkpoint. Repeating the identical command does not extend a deadline,
cancel the replacement, or submit another job. A different minimum step is
rejected once a restart has been recorded. Omitting the option later does not
cancel an already armed restart.

Success requires `SAVE_POLICY_MONITOR_RUNNING`, verified actual imported runtime
modules and a completed first monitoring pass. `...UNVERIFIED` exits 2: inspect
the error and repeat the same command. A staged hash alone is not success.
The repair refuses expired monitoring, unrecognized frozen source, a foreign
controller host, an unresolved qsub or an in-flight topology handoff.

## Controlled restart sequence

1. Keep the current allocation while waiting. Ignore staging directories and
   directories without a complete continuation marker.
2. Validate checkpoint components and inventory; hard-link a durable protected
   copy before any qdel. Bind the intent to this setup's exact active job and
   checkpoint authority.
3. Use the existing PBS owner/queue/name/submission-path checks to cancel only
   that writer. Confirm it ended and acquire its stage lock. Re-read the latest
   full checkpoint and preserve any additional complete progress.
4. Submit one formal continuation through the existing submission journal.
   Preserve the existing promoted retry policy: >=1 eligible free whole node
   selects 2 nodes, zero or unknown capacity selects 1. Queue directly in
   `R9920261000` for 24 hours. There is no new qualification or extra admission
   threshold for an already promoted run.
5. Retain the repair's protected pins until the replacement has saved a later
   full checkpoint. Cleanup failure is reported without blocking monitoring.

An earlier natural 271/confirmed-walltime termination uses the normal latest
complete checkpoint and the repaired configuration; it does not wait for an
unreachable step 125. Application errors remain blocked. Unknown PBS results
and ambiguous qsub never justify duplicate submission. If the monitor expires
before cancellation is attempted, the live allocation is left running. A
previously journaled intentional cancellation is reconciled to one replacement
even if the deadline expires while it is in flight.

## Evaluation recovery

The pinned training loop saves before evaluation. Restoring step 125 and starting
with 126 would therefore skip evaluation if the old job died during that work.
For a scheduled source step (a positive multiple of 25), a new process evaluates
the restored weights **before the next optimizer update**, or reuses verified
complete results for those exact weights and validation settings.

Completion receipts bind checkpoint lineage, step, world and file inventory;
validation/data settings; and the frozen reward source. They contain all four
domain metrics and the digest of the full 1,006-row output artifact. The trainer
waits for asynchronous output writes before publishing PASS. Receipts live in
`control/evaluations/`, outside immutable checkpoints. Each new phase writes a
small `validation_<step>.json` receipt and returns metrics to the usual logger.

For the old, already-running snapshot, reuse requires both a final console
metrics line containing all four domains and completed testing time, and the
full matching validation JSONL. A step record, directory, partial output or
`validation generation end` message alone is insufficient. When proof is absent,
validation is rerun from the checkpoint; existing partial outputs are archived
under `generations/train/replay_archive/`. A kill during replay publishes no PASS
and the next retry tries again. Re-evaluation preserves the training step and
does not consume the training sampler. Bitwise numerical equivalence across
allocations is not claimed.

## Status and limits

```bash
"$D4_PY" -I "$D4_SCRIPT" status --root "$D4_ROOT"
```

Inspect `training_policy`, `active_job_policy`, `controller_runtime` and
`checkpoint_restart`. The old live job can correctly still show 25 while the
new monitor is ready to generate jobs with 5. Restart phases are
`WAIT_CHECKPOINT`, `PINNING`, `CANCELLING`, `SUBMITTING`, `COMPLETE`;
`NATURAL_EXIT`, `SUPERSEDED`, and `EXPIRED` explain alternative outcomes.

The published tests use CPU, local processes, synthetic checkpoint payloads and
local PBS fixtures. They do not claim to have submitted to ABCI or run new GPU
restoration/validation. The running experiment must supply that final evidence.
No running GPU process is hot-patched, and no training checkpoint is created by
the login-node repair. Unsaved updates before the next full checkpoint remain
exposed to administrator termination.

The upstream save-before-validation order, startup validation hook and async
dump-future interface were checked against
[`verl/trainer/ppo/ray_trainer.py` at the pinned commit](https://github.com/verl-project/verl/blob/62ac6d6e9fccc09ab6af2171217cee6e290b799b/verl/trainer/ppo/ray_trainer.py).
