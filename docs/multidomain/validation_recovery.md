# Validation recovery and continuation deployment

`manage_training_continuation.py repair-validation --root ROOT --operation
20261023-validation-recovery-plus24 --extra-hours 24` performs one journaled
deployment. It does not require a particular login host when the old owners have
expired and released their leases. Repeating the same operation does not add
another 24 hours or submit another initial evaluation job.

## Deployment order

1. Validate and hard-link random step 125 (16-rank source) and domain step 100
   (8-rank source) into `outputs/multidomain/validation_backfill_sources`.
   These copies survive rolling retention and temporary resume-source cleanup.
   They are bound to the recorded historical job requests, not merely step numbers.
2. Verify the installed verl validation loop against the pinned upstream AST;
   stage both immutable runtimes, preserving each setup's verifier and sampling
   code. Unsupported source changes stop before restarting owners or submitting.
3. Acquire both owner leases, publish plans with save=5, eval=25, batch-persistent
   validation and opportunistic handoff. Extend each deadline once to
   `max(previous deadline, deployment time) + 24 hours`.
4. Restart the owners from their actual new frozen templates and attest the
   imported modules. Previously expired owners resume normal training monitoring.
5. Directly submit one `0390_d4_valfix` allocation to `R9920261000`, one node,
   eight GPUs, 24-hour walltime. No idle-node threshold, host blacklist or hold
   is applied to this submission.

The source worktree is not the active controller runtime. `status` reports both
the plan policies and `controller_runtime.verified`, plus the policy of the
current immutable training job. A new plan does not hot-patch an old GPU process.

## Historical validation job

The allocation runs random step 125, then domain step 100, in separate fresh
Python/Ray driver processes. It restores full state using the existing qualified
restore code, sets `val_only=True`, runs the original 1,006-row validation with
the setup's own scorer and greedy n=1 decoding, and exits. Actor updates and new
checkpoint writes are explicitly forbidden for this role. The evaluation output
directory is isolated from the authoritative training lineage.

The domain controller supervises this independent campaign during the extended
monitor window. Confirmed exit 271 or PBS-confirmed exceeded-walltime -29 permits
one new attempt. Queued/running jobs, unknown PBS observations and ambiguous qsub
returns never cause duplicate submissions. Application failures block the
campaign and remain visible; they do not stop either training owner.

A fully completed setup is skipped on retry without loading its model again.
The old interrupted jobs did not persist their partial validation batches;
their historical points must initially be evaluated in full. New attempts can
reuse the durable batches written by this implementation.

## Validation persistence

Two hooks wrap the exact pinned upstream validation loop: before each batch,
verify and reuse a complete batch record; after generation and scoring, atomically
write that batch. Upstream generation, reward extraction and metric aggregation
are retained. No installed verl file is edited and no global monkey patch is used.

The journal binds checkpoint inventory/lineage, setup, verifier source and isolated
backend archive, full validation dataset digest, ordered tokenized batch inputs,
decoding settings and dependency versions. Placement and diagnostics paths are
excluded so a resubmission can change nodes without changing scoring semantics.
Unfinished batches rerun. Corrupt records or changed inputs cannot be counted as
completed. The final receipt requires all 1,006 distinct prompt IDs, all output
rows, completed asynchronous writes and finite metrics for all four domains.

On training resume from a scheduled evaluation checkpoint, validation completes
or reuses its verified receipt **before the next optimizer update**. This also
supports a walltime interruption during final step 468 evaluation: resume performs
evaluation only, then finishes, without an extra training step.

## Save policy and topology

Both newly submitted setups use save every 5 and evaluation every 25. The old
random save=25 process receives a controlled restart request at the current or
next checkpoint boundary, derived from its observed progress, never hard-coded
to historical step 125. A scheduled validation at that checkpoint must finish
before intentional cancellation. Natural walltime/271 exits use normal retry
policy and restore the latest full checkpoint. Existing domain training can
finish its allocation; its next resume or qualified candidate uses the new code.

Opportunistic admission requires two observations five minutes apart of at least
two available whole-node slots; domain has priority. Each attempt gets independent
qualification, lineage, cancellation and rollback journals. A pending random
save-policy restart suppresses a concurrent opportunistic transition. Keep the
single running while the candidate qualifies, protect its highest fallback,
then fence/catch up/commit using the established handoff protocol. A failed or
interrupted transition cannot promote the other setup. Genuine validation failures
remain blocked; externally interrupted candidates may retry after cooldown.

After a previous successful promotion, normal retry still requests two nodes
when at least one free slot is observed, otherwise one; requests may queue.
Before a successful promotion, ordinary training retry remains single-node.

## Inspection

Run `manage_training_continuation.py status --root ROOT` and `qstat`.

- Training: `training_policy`, `validation_policy`, `opportunistic_policy`,
  `active_job_policy`, `checkpoint_restart`, `controller_runtime`, deadlines.
- Campaign: `validation_backfill.phase`, `job`, `attempts`, `observation`, `result`.
  `QUEUED` means submitted; consult `observation.job_state` or qstat for Q/R.
- Durable batch progress: each setup's `control/validation_batches/*/progress.json`.
- Completed metrics and output SHA: `control/evaluations/*.json`.
- Campaign result: `outputs/multidomain/validation_backfills/OPERATION/jobs/ID/result.json`.
- Child evaluation logs: paths in the campaign job's `request.json` under
  `children.random.folder` and `children.domain.folder`, then `pbs.log`.

Protected historical checkpoints remain available after PASS for review; this
deployment does not silently delete them. All generated operational files stay
under `outputs/multidomain`; no ZIP bundles or top-level patch directories.

## Validation of this change

CPU fault tests exercise the pinned upstream loop with 1,006 synthetic rows,
interruption after one committed batch, replay, input/checksum/protocol changes,
full-output receipts, two fresh evaluation child processes, duplicate-qsub
prevention, source retention, safe restart timing and final-step evaluation mode.
Isolated Python owner processes verify actual frozen imports and one reserved-queue
submission using synthetic PBS commands. These tests do not replace cluster GPU
acceptance of checkpoint loading and the resulting validation PASS receipts.
