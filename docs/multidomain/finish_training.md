# Finish the two D4 training setups

`scripts/multidomain/finish_training.py stop-and-inventory --root ROOT --ops OPS`
ends both setups and writes a storage/retention review bundle. It never deletes
checkpoints. Run it on ABCI; running it against an extracted history bundle does
not measure cluster storage or stop cluster jobs.

## Stop semantics

The Oct 21 continuation owner has **no stop-file implementation**. Touching the
old monitor's `stop.request` alone is insufficient. This command:

1. Takes the shared deployment lock. Requests the archive watcher and old
   monitor/handoff owners to stop, then takes their leases.
2. Acquires both continuation owner leases. An already stopped owner can be
   retired from any login node. A live owner is signalled only on its own host,
   using its exact UID/command line and a pidfd. No remote PID or SIGKILL is used.
   If the live owner is on another host, it reports that host and stops before
   cancelling any PBS jobs. Run the same command on that host.
3. Preserves the state and plan in the report. Writes persistent `retired.json`
   records and stages a new frozen template changing **only cli.py**: its spawn
   and daemon entry points return immediately while the retirement marker exists.
   Old resume/extend CLIs read this new template from the plan and therefore
   cannot silently revive training. The original template and submitted scripts
   remain unchanged. Deliberate future reactivation needs a reviewed plan change.
4. Queries PBS afresh. It cancels only the current user's jobs in R9920261000
   whose Submit_arguments, job_id.txt and PBS script job name bind them to these
   D4 output trees (including related check/candidate/backfill jobs). It does
   not cancel by job-name prefix alone or use previously pasted job IDs.
5. Verifies there are no remaining scoped PBS jobs and no held training stage
   leases. Query failure is UNKNOWN, not success. A failure produces INCOMPLETE
   and an evidence bundle; rerunning is safe. No checkpoint or tensor is removed.

This stops existing autonomous owners and prevents their normal frozen-runtime
restart paths. It does not disable arbitrary future manual qsub commands, other
experiments, or account-level services. No training process is required to save
an additional step: existing best/final complete checkpoints are retained.

## Storage and checkpoint review

`inventory` can be run independently and is read-only apart from its new report.
The scan covers every run output directory with the exact D4 base name or its
underscore suffixes, both backfill source areas, shared validation jobs and
controller/archive records. Shared base models, input datasets, root-level
caches/wandb and earlier analysis bundles are excluded; they cannot be reliably
attributed to one setup. Symlinks are recorded but not followed.

The report includes per-setup unique allocated/logical bytes and the union of
both setups plus shared check/backfill/operations. Files are deduplicated by
`(st_dev, st_ino)`. Allocated bytes are `st_blocks * 512`, not an estimate from
model parameter counts and not a claim about backend snapshot/compression quota.
Per-setup totals may overlap; the union counts shared inodes once. Model,
optimizer and other training state are also separated.

The **review-only** proposal retains each protocol's overall best full archive
and the latest complete authoritative checkpoint as full state. Per-domain
winners keep model shards/metadata. Unverified historical weight rescues stay
protected for manual review. Other checkpoint payloads, and training-state
payloads of model-only winners, are proposed for review. Metadata, receipts,
logs and code are never proposed for deletion.

The reclaim estimate counts an inode only when *all* its observed links are
proposed for removal **and** that count equals `st_nlink`. External hard links
therefore cannot inflate the estimate. Protection/inventory errors disable the
estimate and mark the scan partial. This is not an executable deletion manifest:
archive modes/indexes and remaining references need a separate reviewed cleanup
operation before removal or conversion.

Outputs:

- `stop.json`: owner fencing, per-job cancellation and final stop verification.
- `storage.json`: scope, unique byte totals, components and scan errors.
- `checkpoint_retention_review.json`: every observed checkpoint path and its
  proposed retention, including old restore pins and failed candidates.
- `*_state_before.json`, `*_plan_before.json`: pre-retirement provenance.
- `RESULT_BUNDLE=...tar.gz`: these small records only, no tensor payloads.

CPU verification:

```bash
python3 -B -m unittest discover -s tests/multidomain -p test_training_finish.py -v
```

Tests exercise PBS H/R cancellation, unknown scheduler observations, strict
identity binding, persistent retirement guards through old restart paths,
writer leases, modified source rejection, hard-link accounting across setups,
external hard links, incomplete protection and symlink exclusion.
