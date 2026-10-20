# 0390: extend the two owners and coordinate node capacity

`scripts/multidomain/manage_training_capacity.py` is an additive operational
adapter for the **already registered** `d4_managed_switch` and `d4_db_switch`
owners, with the installed `2026-10-19.monitor30-training24.1` limits adapter.
It does not replace training, sampling, verifier, checkpoint, qualification, or
handoff implementations. It requires the existing owner/limits source hashes
and experiment contracts to remain valid. An unknown entry is rejected.

## Capacity and order

Only queue `R9920261000` is considered. The actual reservation ID and server
are discovered from `qrstat` and confirmed against a detailed record's `queue`
attribute, rather than assuming the queue name plus `.pbs1` is the correct ID.
The script reads `qrstat -f`/`-F`, `qstat -Qf`, `pbsnodes -av -F json`, and current
`qstat -f -F json`. Two NUMA vnodes of one 192-CPU/8-GPU host count as **one**
physical node. Being part of this reservation is allowed. Other reservations,
partial allocations, offline nodes, jobs on an apparently `free` vnode, and
assigned resources prevent a host from being counted as free. No static host
blacklist is added. PBS still chooses the actual hosts.

Available capacity means **free eligible whole nodes plus nodes occupied by
these four registered jobs**, not the total reservation size, other users'
nodes, or number of queued jobs. Queue CPU headroom caps the free count.

| Available capacity / state | Action |
|---|---|
| 0–2 nodes | Keep both single-node primary jobs eligible for PBS queueing/retry. Hold unstarted dual candidates. |
| 3 nodes | Same; retaining both singles plus a two-node candidate needs four. |
| At least 4 nodes | Domain qualification first, only if two free nodes remain after reserving capacity for any primary jobs still queued/retrying. |
| Domain qualified, single stopped, domain caught up and running | Permit random qualification when another two free nodes remain after primary demand. |
| A candidate already started | Let the existing owner perform qualification/handoff; do not preempt it to reorder priorities. |
| Both dual mainlines active | Existing owners continue their normal full-checkpoint retry policy. No extra capacity threshold for mainline resubmission. |

With both singles running, the first handoff uses 1+1+2 = **4** nodes. After
domain has switched, the random handoff temporarily uses 2+1+2 = **5** nodes.
Both dual mainlines then use **4** nodes. A snapshot is not a reservation: PBS
can allocate to other jobs before ours starts, so this policy cannot guarantee
instant dispatch or prevent administrator termination.

The two owners share a filesystem lock and durable hold journal. Existing queued
candidates receive a **user hold**. New qualification scripts contain `#PBS -h`
before their original snapshot hashes are calculated, so there is no gap in
which a newly submitted candidate competes with the primary jobs. They appear
as `H` in qstat until permitted. Only user holds acquired by this policy are
released; preexisting user/system/operator holds are left alone. Running
candidates are not deliberately held. PBS query/hold has a dispatch race; if
the post-command observation is running, the policy removes its own hold.
New qualification scripts also use `#PBS -c n` to disable **PBS-level process
checkpointing**, which is unrelated to application model/optimizer checkpoints.

An unavailable resource query blocks candidate release, not single-node owner
maintenance. `capacity_error` and the shared status report record the reason.
The adapter contains no `qdel`, does not cancel or downgrade an active primary,
and does not delete checkpoints. The original owner remains responsible for:

- Retry/adoption of interrupted jobs, without duplicate qsub after an ambiguous response.
- The existing common check and candidate identity guards.
- Domain's 8→16 restore/update/full-save and fresh-process 16→16 restore/update/
  sampling-continuity qualification.
- Verified readiness before cancelling the corresponding single, retaining the
  highest fallback checkpoint until the dual catches up.

## Extend monitoring

Use the training environment's Python. `--root` is the live experiment directory;
the script itself can run from a separate Git worktree. `preflight` checks the
installed entry/source guards and reports current scheduler observations. It
does not change jobs or processes.

```bash
python -I scripts/multidomain/manage_training_capacity.py preflight --root "$D4_ROOT"
python -I scripts/multidomain/manage_training_capacity.py apply \
  --root "$D4_ROOT" --extra-hours 24 --operation 20261020-plus24
python -I scripts/multidomain/manage_training_capacity.py status --root "$D4_ROOT"
```

The target is `max(current deadline, now) + 24 hours` for each owner. Thus the
reported Oct 21 00:27 UTC deadlines become Oct 22 00:27 UTC (Beijing 08:27),
if the command is applied before expiry. The same operation ID is idempotent,
including interrupted installations; repeating it does not add another day.
Use a new operation ID for a later, separately requested extension.

The installer asks the existing owners to stop cooperatively, waits for their
filesystem locks, backs up their entries/state/plan under the output tree, and
freezes this adapter into each controller's `runtime_capacity`. It then resumes
both existing owners with the actual remaining duration to the saved target and
checks their locks/deadlines. Editing only `state.json` would not extend the
daemon's monotonic timer. A failure to start one owner does not skip the other.
If interrupted, rerun **the same command/operation ID**, not an independent
single-node monitor. No old package directories need to be deleted.

Current and future training walltime remains governed by the existing 24-hour
adapter. No `qalter` is used. At monitor expiry, running/queued primaries are
left alone; unreleased candidates remain held until this owner is resumed.

Reports and backups are under:
`outputs/multidomain/qwen30b_d4_10k_uniform_ctx64k_seed42_v1_capacity/`.
`operations/20261020-plus24.json` records both targets and partial startup errors.
`status.json` records capacity/decisions; `holds.json` records hold ownership.

## Upgrade an already extended monitor (capacity.1 → capacity.2)

The first deployed run confirmed both new owners and their extended deadlines,
but exposed two operational defects: read-only `status` unnecessarily acquired
the policy mutation lock (`Errno 11` during a background update), and an empty
failure from the assumed `qrstat -f R9920261000.pbs1` lookup. That output alone
does not identify why the site query failed.

Version `2026-10-20.capacity.2` makes status independent of that lock, preserves
monitor status even when PBS is unavailable, discovers the actual queue/reservation
binding, and records command return codes/stdout/stderr on failed queries. It
accepts only the exact previously published adapter SHA for upgrade, retains
existing hold ownership and deadlines, and does not add another day.

```bash
python -I scripts/multidomain/manage_training_capacity.py upgrade \
  --root "$D4_ROOT" --operation 20261020-plus24
python -I scripts/multidomain/manage_training_capacity.py status --root "$D4_ROOT"
```

Use this with the original operation ID. Missing operations are rejected instead
of silently creating another extension. Repeated upgrade is idempotent. Both
owners restart cooperatively only when the frozen adapter actually needs updating.

The report now separates:

- `monitoring_active` (status) / `monitors_verified` (installation): both owners
  and their saved deadlines have been checked.
- `capacity_ready`: a valid capacity observation is available. This can be true
  with zero free nodes; it indicates query health, not permission to start a dual.
- `RUNNING_CAPACITY_WAIT`: monitoring is running but the resource query is still
  unavailable or another owner is updating the policy. Check the warning/current
  status; do not create a new 24-hour operation to fix a query.

If all documented query forms remain unavailable, dual release stays blocked
and the single-node retry loops continue. The detailed error needs cluster-side
inspection; mocked tests cannot establish why a live reservation command failed.

## Validation and limitations

```bash
python -m unittest discover -s tests/multidomain -p test_training_capacity.py -v
```

Tests cover physical-node counting, reservation membership, indirect resources,
scarcity, queued-primary headroom, domain priority, the second handoff overlap,
hold adoption/ownership, unknown-query behavior, preserved owner callbacks,
initial-only held submission, deadline extension, repeat installation, and
pending-qsub recovery across restart. These are CPU/mock tests, not an ABCI
GPU or live scheduling test. Validate the live installation with `preflight`,
then confirm both `controller_active` fields, deadlines, and `capacity_error`.

ABCI documents the reserved-node lookup in its
[job guide](https://docs.abci.ai/v3/en/job-execution/#cautions-for-using-reserved-node)
and its restriction on user `qalter` in
[system updates](https://docs.abci.ai/v3/en/system-updates/#2025-07-25).
