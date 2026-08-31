# Qwen3-30B Legal Entropy Curve — ABCI Serial v2

This version submits **exactly one PBS job**. Inside that one 8-GPU allocation,
checkpoints are processed strictly in order:

```text
step 0 generation
→ step 0 full-vocabulary entropy
→ step 100 generation
→ step 100 full-vocabulary entropy
→ CPU aggregation
```

The formal run uses the same sequence for steps
`0, 10, 20, ..., 100`. Only the `none` legal-hint condition is evaluated.

The DeepMath training data is not read during this evaluation. Step 0 uses the
original Qwen3-30B-A3B-Instruct-2507 model; nonzero steps use the model-only
snapshots produced by the earlier RLVR run. Every checkpoint evaluates the
same frozen legal probe.

## Files

User-facing entry points:

```text
scripts/submit_legal_rlvr_entropy_curve_serial_v2.sh
jobs/q3_legal_rlvr_entropy_curve_serial_v2.pbs
```

Internal components called by the serial job:

```text
scripts/run_legal_rlvr_entropy_step_v2.sh
scripts/run_legal_rlvr_entropy_aggregate_v2.sh
scripts/prepare_legal_entropy_probe_v1.py
scripts/legal_rlvr_entropy_probe_v1.py
scripts/aggregate_legal_rlvr_entropy_curve_v1.py
scripts/aggregate_legal_keyword_entropy_curve_v1.py
```

Do not `qsub` either internal runner.

## 1. Install

Upload the ZIP to the exact ABCI project directory:

```text
/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr/
```

Then run:

```bash
PROJ=/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr
cd "$PROJ"

unzip -o legal_rlvr_entropy_curve_v2_serial.zip

chmod +x \
  scripts/submit_legal_rlvr_entropy_curve_serial_v2.sh \
  scripts/run_legal_rlvr_entropy_step_v2.sh \
  scripts/run_legal_rlvr_entropy_aggregate_v2.sh \
  jobs/q3_legal_rlvr_entropy_curve_serial_v2.pbs

bash -n \
  scripts/submit_legal_rlvr_entropy_curve_serial_v2.sh \
  scripts/run_legal_rlvr_entropy_step_v2.sh \
  scripts/run_legal_rlvr_entropy_aggregate_v2.sh \
  jobs/q3_legal_rlvr_entropy_curve_serial_v2.pbs
```

No output from `bash -n` means the shell syntax is valid.

## 2. Set the two experiment paths

```bash
PROJ=/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr
WORK=/groups/gcg51557/experiments/0390_rlsd

export LEGAL_DATA=/absolute/path/to/the/original/legal_reasoning_data.jsonl
export SNAPSHOT_ROOT="$PROJ/ckpts/model_snapshots/<deepmath-run-name>"
```

`LEGAL_DATA` is the original JSONL used by
`legal_reasoning_probe_vllm_two_stage_entropy_noChunk.py`.

`SNAPSHOT_ROOT` is the directory containing:

```text
global_step_10/
global_step_20/
...
global_step_100/
```

Each nonzero step must have a `.complete` marker.

## 3. Smoke test first

The smoke test uses two frozen legal cases, `none` hint only, and steps 0 and
100. This command submits exactly one job:

```bash
MODE=smoke \
SAMPLE_SIZE=2 \
STEPS="0 100" \
SEED=1 \
PROBE_SEED=1 \
MAX_NEW_TOKENS=4096 \
bash scripts/submit_legal_rlvr_entropy_curve_serial_v2.sh
```

Monitor:

```bash
qstat -u "$USER"
tail -f "$PROJ/logs/legal_entropy_serial_smoke.pbs.log"
```

Smoke succeeds only when this exists:

```bash
test -f \
  "$PROJ/outputs/legal_entropy_curve_qwen3_30b_smoke/.serial_complete" \
  && echo "[PASS] smoke complete"
```

It should also contain:

```text
step_000/seed_1/.complete
step_100/seed_1/.complete
curves/.complete
```

Do not submit the formal job before the smoke job reaches `.serial_complete`.

## 4. Formal curve

After smoke passes, submit the formal run:

```bash
MODE=full \
SAMPLE_SIZE=100 \
STEPS="0 10 20 30 40 50 60 70 80 90 100" \
SEED=1 \
PROBE_SEED=1 \
MAX_NEW_TOKENS=4096 \
bash scripts/submit_legal_rlvr_entropy_curve_serial_v2.sh
```

This also submits exactly one job. It uses one frozen 100-case probe and only
the `none` hint for every checkpoint.

## 5. What “aggregate” means

Aggregation is CPU-only post-processing:

- read completed `trajectory_entropy.jsonl` and `token_entropy.jsonl`;
- merge checkpoint, case, correctness, and region statistics;
- compute curve tables and confidence intervals;
- create CSV, PNG, and PDF outputs;
- compute exact legal-keyword entropy and coverage curves.

It does not load the 30B model and does not need a GPU. In serial v2 it runs at
the end of the existing GPU job, so no second PBS job is submitted. The GPUs
remain idle briefly during plotting, which is preferable here to requesting
another scheduled allocation.

Main outputs:

```text
outputs/legal_entropy_curve_qwen3_30b/curves/
├── legal_entropy_curve_summary.csv
├── legal_entropy_trajectory_merged.csv
├── legal_entropy_curve.png
├── legal_entropy_curve.pdf
├── legal_performance_curve.png
├── legal_performance_curve.pdf
├── legal_entropy_diagnostics.png
├── legal_entropy_diagnostics.pdf
└── keywords/
```

## 6. Resume behavior

The formal 11-checkpoint run may exceed one reservation or walltime window.
If it stops, submit the exact same `MODE=full ...` command again only after the
old job is in state `F`.

The new job:

- skips any step whose run directory already contains `.complete`;
- reuses complete generation output when only entropy scoring remains;
- continues with the first unfinished checkpoint;
- regenerates all curves after every requested step completes.

Do not use `OVERWRITE=1` for an ordinary retry.

If a process was externally terminated while publishing a paired output and
the runner explicitly reports `partial ... output exists`, inspect that one
step before using `OVERWRITE=1`.

## 7. Old jobs

Do not use the old multi-job launcher:

```text
scripts/submit_legal_rlvr_entropy_curve_v1.sh
```

Do not submit:

```text
jobs/q3_legal_rlvr_entropy_aggregate_v1.pbs
```

They are not included in this serial package.
