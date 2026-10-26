# 0390 independent training continuation

Entry: `scripts/multidomain/manage_training_continuation.py`.
Implementation: `d4_training_continuation/`.

This is the successor to the old capacity/check/handoff owners. The same code
serves random-batch and domain-batch, with separate leases, deadlines, pending
submissions, checkpoint lineages and handoff commits. A failed domain transition
does not disable random, and random does not wait for domain to succeed.

## Operational contract

The table below is the original default policy. The opt-in
[`enable-opportunistic` upgrade](training_opportunities.md) additionally observes
running singles and admits a fresh two-node candidate after two capacity checks.
It can retry an administrator-interrupted qualification after cooldown; actual
application/verification failures remain blocked. Existing duals and the formal
termination/retry policy are preserved. Each new attempt has a separate journal
and commit rather than overwriting the original handoff record.

| Situation | Action |
| --- | --- |
| Initial migration | Adopt both existing singles; extend each deadline once by 24 hours; retire only the verified old candidates |
| New candidate | Submit directly to `R9920261000`, no hold, host blacklist, idle-node threshold, or two-job-per-user gate |
| Candidate fails before durable handoff commit, including 271 | Latch that setup to single-only; never retry that candidate automatically |
| Single is already fenced when candidate fails | Confirm candidate and old single terminal; restart from the highest protected complete single checkpoint |
| Formal training ends after successful handoff, exit 271 or explicitly confirmed walltime -29 | Re-evaluate eligible whole-node availability for this retry |
| At least one eligible free node | Submit a two-node formal continuation; PBS may queue it awaiting a second node |
| Zero eligible free nodes | Submit a one-node formal continuation |
| Job termination unknown | Query again; do not submit another job |
| Termination confirmed, capacity lookup fails twice | Submit one node with explicit `capacity_unknown` reason |
| Application failure | Keep evidence and report a blocked training restart; no blind retry |
| Capacity changes while formal job is Q/H/R | Leave the job alone; do not cancel/requeue |
| Deadline | Stop scheduling; retain active formal training; retire an uncommitted candidate and complete an already-needed single rollback |

All new allocation scripts request 24-hour walltime, the reserved queue and
`RTYPE=rt_HF`. Automatic PBS rerun is disabled; the controller owns retry lineage.
No `qhold` or `qrls` is used. An administrator's unrelated hold is not overridden.
Capacity uses the queue's validated project/`nii` profile, physical 192-CPU/8-GPU
hosts, vnode resource assignments and queue CPU headroom. It does not infer
availability from the cluster-wide total, or call an unavailable reservation ID.

## What runs in the candidate allocation

One two-node PBS job contains all five fresh Python/Ray training processes:

| Phase | Restore | Real work |
| --- | --- | --- |
| A | 8 → 16 | Latest complete single checkpoint captured at actual startup; next GRPO batch; synchronous full save; rollout sync |
| B | 16 → 16 | Restore A; next original batch; full save and sync |
| C | 16 → 8 | Restore B on one physical host; next original batch; full save and sync |
| D | 8 → 8 | Fresh-process restore C; next original batch; full save and sync |
| E | 8 → 16 | Restore D; next original batch; full save and sync; wait for handoff; continue training in the same process/allocation |

The process boundary is a real subprocess boundary, not a reinitialization of
the same trainer. All allocated Ray GPUs must become free between phases. The
eight-rank phases use one eight-GPU resource-pool bundle and additionally verify
eight actor ranks on exactly one physical host. Sixteen-rank phases verify eight
ranks on each of two hosts. TP2/PP2/EP2/CP1 and TP4 rollout replicas are checked.

Global batch 256, eight responses per prompt and eight PPO minibatches remain
unchanged. Domain plan metadata, exact prompt IDs at generation/update and the
saved dataloader cursor are checked. Random uses the original sampler and pinned
verl's dataloader/RNG and exhausted-epoch policy. The baseline evaluation schedule
and metrics logging remain enabled; A–D finish their current iteration before
exiting before generation of the next training batch. At least five unfinished
steps must remain. A candidate cannot replace a single that has already finished
the full 468-step experiment.

The isolated domain math backend/archive and repaired scorer are copied from the
accepted domain snapshot without changing scoring policy. Only its diagnostic
output path moves to the new job. Random retains its own existing reward code.

## Restore evidence and the audit repair

The optimizer loader remains pinned to Megatron Core 0.13.1 and the checked verl
checkpoint-manager source. Loading rebuilds descriptors from live model
parameters before optimizer restore; it does not deep-copy parameters or change
checkpoint tensors. Model, optimizer, LR and RNG restore are all mandatory.

`pinned_audit.py` compiles hash-bound audit source in a private namespace. It checks
that the offload adapter calls that exact audit function and requires policy
`fused-adam-empty-local-group-step-v1`. A normal import of an older audit package
cannot silently replace it. Every phase has its own expected world size, token,
driver/actor evidence and all-rank load records. Mismatches write expected/actual
comparison records before raising. Legacy summaries are not relabeled as passing.

Same-layout phases compare sampled model/optimizer state and full LR state.
Cross-layout phases verify full backend restore, initialized optimizer audit,
full LR equality, RNG restore, dataloader/sampler continuity and real update/save.
This is not a claim of elementwise equality for every tensor across different
distributed layouts. Large checkpoint inventory entries use size/mtime; small
metadata also uses SHA256. The actual GPU phases remain the acceptance gate.

## Safe handoff and checkpoint authority

`ready.json` is insufficient to stop a single on its own. The controller verifies
all five phase records, evidence hashes, the live PBS execution and the final
complete checkpoint. It fences the single retry owner, pins a fallback, cancels
only that setup's verified current single, waits for PBS terminal state and its
stage lock, and then pins the highest final complete single checkpoint.

E receives the exact catch-up target. It force-saves that target before waking
rollout, synchronizes weights, writes `matched.json` and waits. The controller
durably writes `control/handoff_commit.json`, changes authority, then acknowledges
E. Only this durable commit enables adaptive formal retries. No second train job
is submitted at handoff.

Before the commit, failed-candidate checkpoints cannot become training authority.
After it, the committed lineage remains authority even through 16→8→16 retries;
the old pre-switch single is never resurrected. A crash after the commit resumes
the committed lineage. A crash before it remains eligible only for the protected
single rollback.

Checkpoint payloads are pinned using hard links on the same filesystem. No large
copy/reflink fallback is silently attempted. The active lineage keeps two complete
checkpoints plus a currently consumed source pin. Intermediate phase pins are
retired when their processes finish, while small audit records remain. Once the
dual has caught up and the handoff commit is durable, completed checkpoints of
the fenced previous single and the fallback pins can be retired. Cleanup failures
are reported separately and never disable formal monitoring. Old failed-candidate
evidence and unrelated workspace files are not broadly deleted.

## Deploy from a pinned operations worktree

Use the exact commit supplied in the GitHub handoff, not a moving branch head.
Keep one detached operations worktree outside the experiment's working tree.
The deployed owners freeze their code under `outputs/.../control/operations/`;
editing the operations checkout later does not replace running controller code.

Run with the established training Python. `self-test` is CPU-only and submits no
jobs. `preflight` reads current ownership/identities/checkpoint/source evidence and
PBS state. It rejects a legacy handoff already in progress, unknown pending
submissions, source drift or incompatible installed checkpoint manager.

```bash
"$D4_PY" -I "$D4_SCRIPT" self-test
"$D4_PY" -I "$D4_SCRIPT" preflight --root "$D4_ROOT"
"$D4_PY" -I "$D4_SCRIPT" apply --root "$D4_ROOT" \
  --operation 20261021-continuation-plus24 --extra-hours 24
"$D4_PY" -I "$D4_SCRIPT" status --root "$D4_ROOT"
qstat
```

`apply` prepares both templates before requesting old owners to stop. New owners
hold both old controller/monitor leases and the old submission leases, preventing
two owners from submitting for the same stream. Both single monitors must be
confirmed adopted before either new candidate is admitted. Registered current
single IDs are refreshed after the old leases are released. Candidate submission
is asynchronous; the CLI waits a bounded interval for receipts and explicitly
distinguishes `CANDIDATES_SUBMITTED` from pending/blocked admission.

Re-run **the same operation ID** if interrupted. Its deadline targets and pending
submission records are reused; another 24 hours or another candidate is not added.
The deadline is `max(existing deadline, now at first application) + 24h` separately
for each setup. UTC is stored; status also prints Asia/Shanghai time.

New names (candidate and formal continuation share the same name per setup):

| Setup | Job name | New output run suffix |
| --- | --- | --- |
| Random | `0390_d4r_cont` | `_continuation_v1` |
| Domain | `0390_d4db_cont` | `_db_continuation_v1` |

Use `request.json`'s `role`/`expected_nodes` and `qstat -xf -F json JOB_ID` to
distinguish candidate/formal and one/two-node allocations. Do not infer topology
or success from a truncated job name. `Q` means queued; `R` means running;
`promoted: true` plus `handoff_commit.json` means the handoff committed.

For a later **additional** extension, use a new extension operation ID:

```bash
"$D4_PY" -I "$D4_SCRIPT" extend --root "$D4_ROOT" \
  --operation YOUR_NEW_EXTENSION_ID --extra-hours 24
```

`resume --root "$D4_ROOT"` restarts stopped owners without extending deadlines
or rearming a failed handoff. The shared-filesystem lock, not a PID on another
login host, is the ownership check. Detached owners survive SSH logout. If their
login host/process is killed, `status` exposes the inactive lease/stale heartbeat;
run `resume` on a healthy login node. No external service or cron was installed.

Do not restart the old ZIP/capacity/check owners after migration. New status and
controller logs are the authoritative operational interface.

## Repair an ended old candidate without an exit status

A deleted held/queued PBS job can have `job_state=F` and no `Exit_status`.
The initial controller incorrectly required a numeric exit code even when merely
confirming that an old candidate had ended. This produced
`Unknown PBS cancellation state`, leaving the new candidate unsubmitted while
single-node monitoring continued. `QUALIFYING` with `candidate: null` does not
mean a GPU qualification is running.

Cancellation now accepts an identity-verified `F`/`C` observation without an exit
code. The training retry policy still requires a numeric code and permits only
271 or explicitly confirmed walltime -29. Missing records, query failures and
unrecognized states remain unknown; they are not treated as successful cleanup.
An unknown-state error now includes the actual job state and substate.

Updating a Git checkout alone cannot fix an existing frozen owner. From the login
host shown in that owner's `status`, use the patched operations checkout:

```bash
"$D4_PY" -I "$D4_SCRIPT" self-test
"$D4_PY" -I "$D4_SCRIPT" repair-cancellation --root "$D4_ROOT" --kind random
"$D4_PY" -I "$D4_SCRIPT" status --root "$D4_ROOT"
```

The v2 repair stages a new template, changes **only `scheduler.py` and `cli.py`**
within that template, verifies all other hashes, and gracefully stops only the selected
login-node controller. It requires the same host, process owner and exact command
line, and uses a Linux pidfd to avoid signalling a reused PID. It never force-kills
the controller. Under the controller's ownership lock it atomically updates the
operation's plan pointer, then restarts the owner from the repaired snapshot.
Existing submitted job snapshots and the original template are retained.

The first repair published a corrected scheduler but its launcher still imported
`operation/template/runtime` instead of `plan["template"]/runtime`. A restarted
process could therefore continue using the original scheduler. V2 fixes the
entry path and adds a startup binding recorded by the new owner under its lease:
actual imported module paths/hashes, template manifest digest, PID and host.
Startup rejects a module loaded from another snapshot before any PBS action.
`status.controller_runtime.verified` requires that binding to match the current
plan and running owner; it is not inferred from a staged file hash or a lock alone.
An unchanged plan with an old/unbound owner is also restarted. A repeated repair
with a correctly bound running owner does not restart it again.

Some Python builds omit `os.pidfd_open` and/or `signal.pidfd_send_signal` even
though the running Linux kernel provides the operations. `process_signal.py`
uses an available Python wrapper, otherwise the same kernel syscall through
stdlib `ctypes` on Linux x86-64/AArch64 LP64. No Python/environment upgrade is
needed. The syscall numbers are bound to the Linux UAPI sources linked in that
module. There is no fallback to a plain numeric-PID kill or a process-group signal.
Before stopping an owner, the repair probes opening its own pidfd and sending
signal 0, which delivers no signal. If the kernel lacks support or access is
denied, it stops before signalling an owner or publishing a replacement plan.

Deadlines, submission journals, checkpoint authority and handoff failure latches
are preserved. In particular a domain transition already marked `SINGLE_ONLY`
after 271 is not rearmed by repairing random. The repair itself sends no PBS
submission/cancellation commands; the restarted controller resumes its existing
policy. Once old-candidate retirement succeeds, an admitted, never-submitted
random candidate is submitted once, directly to the reserved queue.

The command is repeatable after interruption. If graceful shutdown takes longer
than its bounded wait, rerun the same command on that host. If restarting failed
after plan publication, rerunning resumes the repaired snapshot. The result
`REPAIRED_CONTROLLER_RUNNING` now confirms the current process's runtime binding,
not a candidate's qsub receipt or a successful training handoff. If the bounded
startup wait ends without that proof, the result is `REPAIRED_CONTROLLER_UNVERIFIED`;
inspect the reported binding reason and `controller.log`, then rerun the same
repair. `controller_runtime.first_check_completed` distinguishes a completed
new-owner polling pass from an error left over before restart. Inspect `candidate.job_id`,
`old_candidate_retired`, `pbs`, and `promoted` separately. The scheduling policy
and 24-hour training walltime are unchanged.

The launch regression tests execute the real `python -I controller_entry.py`
chain and daemon against local fixture PBS executables. They reproduce both the
v1 updated-plan/stale-entry state and an already-current plan with an unbound
owner, verify fresh module origins, retire an F/null-exit candidate, and record
exactly one dual submission without cancelling either single. Repeating the
repair preserves the PID, deadline, submission receipt and independent domain
failure latch. A wrong import root is rejected before any PBS command. These
are CPU/process integration checks, not GPU qualification.

## Diagnostics and acceptance

For each run, inspect `control/state.json`, `control/controller.log`,
`control/events/`, and each job's `pbs.log`, `request.json`, `allocation.json`,
`A`…`E`/`train` evidence. All-rank audit binding/comparison files identify the
actual source and any changed state sections. `actor_metrics` preserves entropy
and actor loss alongside per-step reward/output-length/prompt-domain summaries.

The controller journals before `qsub`. If its result is ambiguous, it tries the
durable stdout/ID record and exact PBS owner/name/submission-path matching. It
never guesses that an absent/expired scheduler record means no job was submitted.
Inspect that pending folder and the scheduler before resolving an ambiguous ID;
do not delete the intent to force a new submission.

Local validation for the initial implementation: 40 CPU fault/integration tests,
13 existing audit-binding tests and 14 existing worker-repair tests passed.
They cover queue policy, per-setup failure latches, fallback after qdel, five-phase
evidence, exact save-before-sync order, source isolation, deadline idempotence,
submission ambiguity and durable commit recovery. PBS/GPU operations in these
tests are synthetic. No ABCI submission or real 8↔16 checkpoint restore has been
performed by the development environment; those are verified by the new candidate
jobs before they can replace the current singles.

The cancellation repair adds 13 regression/fault tests; all 53 continuation tests
pass. They include the real PBS adapter's F-without-exit path, unchanged strict
retry/identity checks, exactly-once candidate submission after cleanup, immutable
snapshot publication, restart interruption/repetition, separate setup failure
latches, and graceful controller lock handoff using a real local child process.
No cluster job or GPU is used by these tests.

The Python-binding compatibility follow-up expands the suite to 60 tests. The
full suite passes both normally and with **both Python pidfd attributes removed**,
without skips. It exercises real kernel pidfd operations and graceful local-child
shutdown in the missing-wrapper case, as well as unsupported-kernel/permission
errors that must leave the controller and operation plan intact. These local runs
do not establish the target login host's kernel support; its self-test and the
pre-stop capability probe check that host before deployment.
