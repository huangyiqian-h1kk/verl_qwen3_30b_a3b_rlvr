# Retired D4 checkpoint cleanup (approved October 27, 2026)

Use `scripts/multidomain/cleanup_finished_training.py` **on ABCI**, after the
successful `finish_training.py` stop. This script does not stop jobs, submit
jobs, extend monitors, or select a new retention policy. It implements the
reviewed policy from `d4_finish_20261026T181855_ecf148e6`.

## Retention

| Setup | Step | Final representation |
| --- | --- | --- |
| Random | 150, 200 | Full model/optimizer/LR/RNG/dataloader checkpoints |
| Random | 125 | Model shards and loading configuration |
| Random | 50 | Existing unverified historical weights; preserve for review |
| Domain | 175, 225 | Full model/optimizer/LR/RNG/dataloader checkpoints |
| Domain | 100, 200 | Model shards and loading configuration |

These are exact archive object identities, not a wildcard over step numbers.
Random 125 and domain 100 are converted from full state to native Megatron
weights. Domain 200 is already weights-only. Random 195 and domain 220 archive
objects are retired. Other reviewed source pins, checks, staging and rolling
checkpoint tensor payloads are removed, including redundant aliases of retained
objects. Their logs, configs, JSON receipts, sampling evidence and code remain.
This is not an HF export or a new GPU restore validation.

The reviewed inventory contains 62 checkpoint paths, including duplicates and
10 paths with only a small residual dataloader file. It proposes 54 payload
retirements plus two full-to-weights conversions, retaining eight canonical
objects. Expected scope: 12.522 TiB before, about 10.740 TiB reclaimable, about
1.782 TiB afterward. Numbers use allocated bytes, deduplicated by device/inode.
Directory and audit-record overhead can make the final measured figure differ
slightly. Backend quota, open files and storage snapshots are separate concerns.

## Run in two explicit steps

Run the published pinned checkout or extract the required files with `git
archive` into `outputs/analysis/d4_cleanup_tools_<revision>`. No checkout change
or patch ZIP is needed. Required files are the two finish/cleanup scripts and
these modules from that SAME revision: `__init__.py`, `common.py`,
`checkpoints.py`, `best_checkpoints.py`, and `evaluation.py` under
`d4_training_continuation`.

```bash
python -B -I scripts/multidomain/cleanup_finished_training.py plan \
  --root /absolute/experiment/root \
  --review-dir /absolute/experiment/root/outputs/analysis/d4_finish_20261026T181855_ecf148e6 \
  --operation 20261027-best-final
```

`plan` does not remove checkpoint payloads or change archive indexes. It takes
leases to obtain a consistent view, validates the stopped owners and the
protected archives, and writes a per-file manifest and metadata backups under
`outputs/analysis/d4_cleanup_20261027-best-final`. It prints `PLAN_SHA256` and
space estimates. The three original reviewed report files are SHA256-bound;
new checkpoint paths, changed archive selection, unreviewed holds, unknown PBS
state, active jobs or writers, and unexpected external hardlinks block cleanup.

```bash
python -B -I scripts/multidomain/cleanup_finished_training.py apply \
  --root /absolute/experiment/root \
  --operation 20261027-best-final \
  --confirm-plan SHA256_FROM_PREVIEW

python -B -I scripts/multidomain/cleanup_finished_training.py verify \
  --root /absolute/experiment/root \
  --operation 20261027-best-final
```

**Apply permanently unlinks the approved payload files.** The metadata backups
are not backups of deleted tensor data. Reuse the SAME operation ID, plan hash
and code revision after interruption. Do not create another operation to bypass
an INCOMPLETE result. Rerunning a completed apply is idempotent and verifies the
retained data again. A successful final result is `COMPLETE`, followed by
`VERIFIED`; `RESULT_BUNDLE` identifies the small audit bundle.

## Consistency and recovery

1. Acquire the shared deployment lock, archive watcher/install locks, both
   continuation/archive/legacy leases and all recorded stage/writer leases.
   Verify persistent retirement markers and the hashes/content of the frozen
   spawn/daemon guards. A stopped owner can be checked from any login node.
2. Query PBS afresh; no scoped jobs may exist. Unknown scheduler results fail
   closed. The script never sends qdel or signals an owner.
3. Verify all protected files, full-state component inventories and lineage
   markers. Large shards are size/mtime checked; small metadata is hashed.
   This is not a GPU tensor-load or full-content hashing claim.
4. Before any payload unlink, publish archive indexes with final modes,
   atomically reduce the two model-only manifests, retire obsolete manifests,
   and rename completion markers/stale latest pointers to `.before_cleanup`.
   Every affected directory receives `PAYLOAD_CLEANUP.json`. Historical source
   references stay as evidence, not live resume sources. Each setup's
   `control/final_checkpoints.json` lists the retained canonical objects.
5. Delete only paths in the confirmed plan, rechecking inode, size and mtime
   immediately before unlink. Never recursively remove job/control/archive
   directories. Directory traversal uses O_NOFOLLOW file descriptors. Hardlink
   removal does not overwrite the inode shared by a retained archive.
6. Revalidate preserved archives, final metadata, absent deleted paths, and PBS.
   Recompute the output scope's unique allocated storage.

Metadata transitions precede payload removal, and are individually restartable.
An interruption can leave a partially processed cleanup, marked INCOMPLETE;
retained full checkpoints are never mutated, and the same command completes the
remaining transitions. Permanent retirement guards stay installed. Resuming
training in future requires a separately reviewed plan using a preserved full
archive, not the deleted old source pins. Keep the archive watcher stopped;
starting historical rescue again is not part of this cleanup.

## CPU tests

```bash
python3 -B -m unittest discover -s tests/multidomain -p test_training_cleanup.py -v
```

Tests use tiny synthetic checkpoint files and mocked PBS. They cover dry-run,
full/weights retention, shared-inode survival, phase interruptions and repeated
apply, external links, symlinks, changed files/manifests/plans, new checkpoints,
busy writers, changed retirement state, and unknown/active PBS observations.
No model payload from ABCI is downloaded for testing.
