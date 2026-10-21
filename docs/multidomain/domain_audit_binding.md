# Domain candidate audit binding repair

The candidate request can pin the newer empty-local-group audit source while
saved and restored summaries still have the legacy schema. In the observed
failure all saved rank summaries lacked `optimizer_audit_policy`, twelve ranks
had matching legacy restore summaries, and the second phase had no detailed
optimizer audit files. The newer `engine_state` always returns its policy marker,
even when its optional recorder skips a write. Missing phase environment alone
therefore does not explain the output schema.

This establishes a mismatch between the intended audit implementation and the
one producing the summaries. It does **not** identify the historical worker's
actual import path or prove that optimizer values are correct. The old failure
does not retain those details. A global package collision, an earlier bound
function, or another old executable artifact must not be silently accepted.

`domain_pinned_audit.py` loads the request-bound `__init__.py`, `core.py`,
`state_audit.py`, and `audit_offload.py` source bytes under a private namespace.
It bypasses Python path precedence and pyc files without reloading or changing
ordinary shared modules. The offload adapter's `engine_state` binding is checked.
Every actor records the private source paths/hashes, policy, and any already
imported ordinary audit path. Save/load requires the expected policy marker;
same-topology comparisons write expected/actual summaries before raising.

The existing normalization policy and numerical comparisons are unchanged.
Model/optimizer comparison remains sampled tensor values plus all metadata, and
LR comparison remains full state. Model and optimizer loading, LR and RNG load
checks, sampler continuity, original check acceptance, and the two-process domain
qualification remain required. No saved legacy summary is relabeled as new.

## Deployment

Use a detached operations worktree of this Git branch. Do not replace the
dirty experiment checkout with the older repository base. Run with the existing
training Python, so the original controller can resume its usual environment:

```bash
python -I scripts/multidomain/repair_domain_audit_binding.py \
  --root /path/to/data_mixture_rl_4d --domain-job FAILED_DOMAIN_CANDIDATE
python -I scripts/multidomain/repair_domain_audit_binding.py \
  --root /path/to/data_mixture_rl_4d --domain-job FAILED_DOMAIN_CANDIDATE --apply
```

Preflight is read-only. Apply requires PBS confirmation of the bound failed
candidate, the diagnosed error and legacy schema, unchanged code/contracts and
common check evidence, an unexpired owner deadline, and no handoff or pending
submission. It cooperatively pauses only the domain owner and uses controller,
submission, execution and shared capacity-policy locks.

The existing unpromoted dual checkpoint directory is atomically renamed into
`handoff/audit_binding_repair/old_checkpoints/`. Its bytes and old job snapshot
are preserved; single-node checkpoints are untouched. The old absolute path in
historical evidence now refers to its archived counterpart. A journal supports
rollback after interrupted publication and refuses independent edits or a newly
written checkpoint. New code, contract hashes, limits adapter hashes and owner
identity are published together. Existing capacity entry/policy files remain
unchanged. The original controller resumes with the remaining monitor time,
without another 24/30-hour extension, and submits a new candidate through its
normal capacity policy and reserved queue.

The replacement reruns 8-to-16 restore, the next planned domain update and full
save, then fresh-process 16-to-16 restore and the following update. It captures
the latest complete single-node checkpoint when it actually starts. A passed
first phase alone never authorizes stopping the single-node job. Unknown/new
application failures remain blocked for inspection.

Repeated apply recognizes its receipt and does not clear a replacement candidate.
If the controller deadline has expired, extend that owner through its existing
operations entry before applying. If preflight detects another active candidate,
changed checkpoint set or prior handoff evidence, inspect current state rather
than editing state files to force this repair.

## Verification

```bash
python -m unittest discover -s tests/multidomain -p test_domain_audit_binding.py -v
```

CPU tests cover poisoned ordinary imports, changed source/function bindings,
phase and world-size guards, rejected legacy summaries, model/optimizer/LR
negative controls, failure evidence before raising, unchanged monitor deadline
and single owner, checkpoint-preserving rollback, and idempotence. They use
synthetic state and scheduler fixtures, not real training data. The real pinned
audit sources were also checked for isolated import and function binding without
loading torch or a model. This is not a claim that the next sixteen-GPU restore
has passed; that must be established by the replacement candidate on ABCI.
