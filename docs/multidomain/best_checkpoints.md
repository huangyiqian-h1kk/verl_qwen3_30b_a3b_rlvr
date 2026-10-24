# Best checkpoint rescue and retention

The previous continuation writer saved every 5 steps, evaluated every 25, and
removed all but its two latest complete checkpoints without consulting scores.
This change adds a separate archive for each setup. It does not change the data,
sampler, batch sizes, GRPO, scoring, save/evaluation frequencies, PBS resources,
monitor deadlines or topology-admission policy.

## Install from the operations checkout

Use the pinned commit in the accompanying GitHub handoff, then run on **any ABCI
login node**, with the existing experiment root and training Python:

```bash
D4_ROOT=/groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d
D4_OPS="$D4_ROOT/../data_mixture_rl_4d_ops_capacity"
D4_PY=/groups/gcg51557/experiments/0390_rlsd/envs/verl_qwen3_moe_megatron_py312_cu128/bin/python
D4_BEST="$D4_OPS/scripts/multidomain/manage_best_checkpoints.py"
"$D4_PY" -B -I "$D4_BEST" self-test
"$D4_PY" -B -I "$D4_BEST" install --root "$D4_ROOT"
"$D4_PY" -B -I "$D4_BEST" status --root "$D4_ROOT"
```

`install` is repeatable. It rescues existing checkpoints **before** any owner
upgrade, starts one detached archive bridge with a frozen code snapshot and a
shared filesystem lock, then stages/installs native hooks separately for the two
setups. Disconnecting SSH does not intentionally stop it. The process can still
be killed by the login host/administrator; inspect its heartbeat after outages.
The bridge remains until `stop`; training-monitor expiry does not remove saved
archives or silently stop archiving a still-running GPU job.

The archive bridge itself does not invoke `qsub`, `qdel`, `qhold` or `qrls`.
Installing a new controller template can gracefully restart a **same-host**
controller, which then continues its existing scheduling policy. GPU jobs are
not stopped or restarted to install archive hooks. A remote login PID is never
signalled. The native template is adopted automatically when that controller's
lease becomes free, or by repeating `install` on its reported host. A handoff or
submission transaction may defer native adoption. Deadlines are not extended.

## What is retained

For each setup and validation-protocol identity:

* The best overall model, selected by the arithmetic mean of the four
  `val-core/<domain>/acc/mean@1` values. Ties keep the earlier step.
* The best available **full** checkpoint. This is separately reported as
  `overall_full` if the highest-scoring recovered model has only weights.
* Each domain's best weights. The same checkpoint selected multiple times has
  one payload; while also needed as the latest/best full checkpoint it stays full.
* Two recent full archive checkpoints for each live writer lineage, and pending
  validation points until a complete evaluation is available.

Historical rescue holds are deliberately sticky: random steps 50/100/125/150 and
domain steps 75/100/125/150 are checked across known rolling, recovery, phase and
backfill sources. Any payload recovered for those points is kept for review,
including a lower-scoring fallback. This initial rescue can therefore retain
more than the steady-state best-model set. The program does not delete those
holds, old recovery sources or backfill protection directories.

Only full, checkpoint-bound validation receipts with all four finite metrics and
the complete, SHA-verified 1,006-row output may select best. Backfilled evaluations
are eligible. Checkpoint and scorer/data/decoding protocol identities are kept
separate. Training reward, partial console lines, check-only scores and an
uncommitted candidate cannot replace a formal setup's best. Scores from an
accepted historical formal lineage remain eligible after later handoffs.

New jobs archive a save **before** rolling deletion, and archive a completed
validation before returning to training. If native protection fails (for example
quota exhaustion), the exception prevents rolling deletion/continued training;
the earlier best remains. Fix the archive error before resuming such a job.

## Current jobs versus future jobs

Already-running GPU workers keep their immutable old imports. They are protected
by the independent bridge, which scans approximately every 15 seconds plus scan
time. This is a **polling bridge**, not the writer's transactional guarantee: a
bridge outage or a retention race can miss a checkpoint. The scanner detects
missing/partial payloads and never reports them as successfully rescued. Native
hooks in subsequent jobs remove this polling gap. Do not remove the bridge while
an old job without native hooks is still running.

Read these statuses separately:

* `bridge_active: true`: lock plus fresh RUNNING heartbeat, not proof that every
  payload was captured. Also inspect `service.errors` and each setup's scan errors.
* `BRIDGE_ACTIVE_NATIVE_PENDING_OWNER`: native upgrade is waiting for the
  controller host/lease; the old GPU job uses the bridge.
* `BRIDGE_ACTIVE_NATIVE_PENDING_TRANSITION`: finish the existing handoff/submission
  before native adoption; no transition was cancelled.
* `NATIVE_PLAN_INSTALLED`: future jobs use the new plan. Check
  `controller_runtime.verified` for a live owner's loaded snapshot, and
  `active_job_native_policy` for the active job's bound policy. A null active-job
  policy means that job still depends on the bridge.
* `rescue_targets["125"]: []`: no matching checkpoint was rescued. This is never
  represented as a successful recovery, nor does it prove every external backup
  location has been searched.

`install` exits 2 if the bridge is not running or a setup scan fatally failed.
Deferred native adoption is explicitly reported and is not a bridge failure.

## Storage and integrity

Archives are below each continuation run's `checkpoint_archive/`, outside
`lineages/*/checkpoints` and all temporary source cleanup paths. `index.json`
maps scores to checkpoint keys and payloads; `rescue_report.json` records initial
rescue results; `last_scan.json` records recent verification/errors. The service
is at `outputs/multidomain/checkpoint_archive_service/`.

Payloads use hard links in the same filesystem. Linking does not copy tensor
blocks, but retaining the links prevents those blocks from being freed. This is
protection against retention/unlink, not a separate disk backup. No source file
is rewritten or chmod-ed. New manifests are published atomically, and winners
are verified and indexed before old archive-owned links are compacted. Explicit
rescue holds and pending evaluations are not automatically expired.

Verification checks every file in the writer's recorded inventory: size/mtime
for large shards and SHA256 for small metadata. It does not deserialize tensors,
run GPU restore, or claim a full checksum of huge model files. A missing model
shard rejects recovery even if some other model shards remain.

`weights` contains the original Megatron distributed model shards, distributed
checkpoint metadata and transformer configuration. It is **not** a ready-made
Hugging Face export and is not accepted by the full-state continuation loader.
The original checkpoint metadata and a manifest state that optimizer/dataloader
restore is unavailable; model-only loading/export must use the matching pinned
Megatron/verl stack and should be GPU-tested before inference use.

Legacy paths without an authoritative file inventory are only rescued as
`unverified_weights`. Their existing shard set and metadata are preserved, but
no completeness or model-load success is claimed and they cannot select best
without further verification. Console-only historical peaks, including domain
step 125 if found, are not silently upgraded to verified best checkpoints.

## Operations and validation

`rescue` performs a one-time historical rescue. `scan` refreshes without starting
a background process. `status` reports the archive/plan states. `stop` requests
only the archive bridge to exit; it does not change training jobs or delete data.

CPU tests use real filesystem links and injected failures: retention removal,
missing optimizer/loader, missing model shards, symlinks, quota/partial linking,
incomplete or changed validation output, ties, separate scoring protocols,
candidate promotion, setup isolation, frozen job-policy binding, remote owner
deferral, expired owner adoption, repeated installs and a real isolated bridge
process. They do not submit GPU jobs or claim that the historical cluster files
still exist. Run `self-test` in the pinned operations checkout before installation.
