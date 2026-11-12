# MOPD qualification signal recovery

The driver can write `FAIL / SystemExit: 1` when PBS terminates it with SIGTERM.
In the pinned Ray implementation, the driver signal handler first exits with 15;
the Cython signal bridge rethrows `SystemExit(1)` with the original termination
message attached as `ray_terminate_msg`. This is different from a model or adapter
exception, and the numeric exit code alone does not distinguish them.

The earlier entry wrapper added `termination_evidence` only after the driver's
`finally: ray.shutdown()` returned. A FAIL receipt without that extra field was
therefore possible even when PBS explicitly reported termination by root and the
traceback contained Ray's signal bridge. The old controller blocked that case.
Missing evidence alone does not prove exactly where the process stopped during
cleanup.

## Revised behavior

- The driver saves structured `exit_details` together with its original FAIL
  receipt, before calling `ray.shutdown()`.
- After PBS owner, queue, job name and submission-path verification, a finished
  job with exit 271 and an explicit `terminated by root@...` comment can retry
  when its identity-bound traceback is exclusively the known Ray signal exit.
  Historical receipts do not require the late marker. A direct exit 15 through
  Ray's `sigterm_handler` is also recognized.
- Conflicting exception details, mismatched identities/attempts, application
  exceptions, chained failures and failed node preflights remain blocked.
- The controller records `retry_assessment` with a reason and the relevant job ID.
  PBS observation failures still wait; they do not authorize new submissions.
- Only the interrupted arm is automatically requeued. The existing barrier still
  requires both arms to finish fresh steps 1–4 and resume to step 8 before formal
  training. Loss, sampling, optimizer, validation and checkpoint policies are unchanged.

## Applying to a frozen campaign

```bash
"$mopd_python" -u -I scripts/mopd.py repair-and-retry --manifest "$mopd_manifest" --repair-kind pbs-sigterm
"$mopd_python" -u -I scripts/mopd.py status --manifest "$mopd_manifest"
```

This is a detached, audited migration for two BLOCKED qualification tasks before
any completed optimizer update. It requires fresh terminal PBS observations,
bound failure receipts, both nodes' successful preflight evidence, and at least
one matching PBS/Ray signal exit. The companion arm may have the already-known
top-level `index` failure before its first update; the migration includes that
previously reviewed sample-ID correction. Other failures and existing training
progress require investigation rather than automatic migration.

The new `d4-mopd-signal-recovery-v4` snapshot preserves the original manifest and
all earlier runtime snapshots. It works from the checked CUDA or batch-metadata
revision, reruns CPU preflight, then registers both qualifications. It does not
rebuild Transformer Engine or replace installed dependencies. Updating Git alone
does not replace an existing frozen runtime.

Regression coverage includes the installed Ray Cython exception constructor,
the actual driver catch/finally path with interrupted cleanup, legacy receipts
without markers, negative failure cases, revision migration and fresh-process
driver/worker imports. These CPU checks do not establish GPU qualification success.

## Login-node C++ runtime for CPU replay

The metadata replay imports the real selected trainer. Its transitive PEFT
import loads Transformer Engine even on a login node. TE was built with
`gcc/13.2.0`; a plain login Python can instead resolve the older system
`libstdc++.so.6` and fail with `GLIBCXX_3.4.32 not found` before replay begins.

CPU preflight now starts this child through `cpu_replay.sh`, loads the same GCC
module, locates its C++ library through that compiler, and sets the loader path
before starting Python. The child checks the actually mapped library before and
after native imports, and records its path and hash in the replay receipt.
Missing modules, a shadowed library, or native import/replay failures still fail
preflight. No GPU dependency is mocked or silently skipped.

This correction belongs to the checkout-side preflight runner. Existing v4
runtime snapshots remain byte-identical. After updating Git, rerun the same
`repair-and-retry --repair-kind pbs-sigterm` command for a failed preflight; it
reuses the verified v4 revision, reruns CPU checks, and registers both arms only
after those checks pass. It does not rebuild TE. Actual GPU qualification remains
required before formal training.
