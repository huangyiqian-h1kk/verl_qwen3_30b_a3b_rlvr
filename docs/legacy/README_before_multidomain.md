# Qwen3-30B-A3B RLVR → Legal Reasoning Entropy Curve (ABCI)

This package evaluates the **same legal reasoning task** used by
`legal_reasoning_probe_vllm_two_stage_entropy_noChunk.py` on a sequence of
Qwen3-30B-A3B-Instruct-2507 model-only snapshots produced by the DeepMath
RLVR run.

DeepMath data is not used by this probe. It is only the training provenance of
the checkpoints.

## What the package measures

For step 0, 10, ..., 100 and each legal hint condition
`none / weak / strong`, it produces:

- the original legal metrics:
  - `acc_source_law`
  - `acc_source_article`
  - `acc_all_law`
  - `acc_all_article`
- full-vocabulary token entropy in nats;
- full-response, reasoning-region, and answer-region entropy;
- correct/incorrect entropy, response length, truncation, and region coverage;
- exact legal-keyword entropy and keyword coverage.

The score stage uses the exact generated token IDs and the same checkpoint that
generated them. It performs a local HF forward pass and computes:

```text
H_t = -sum_v p(v | prompt, generated_prefix) log p(v | prompt, generated_prefix)
```

This is full-vocabulary entropy, not a top-k-logprob approximation.

## Files installed into the project

```text
scripts/prepare_legal_entropy_probe_v1.py
scripts/legal_rlvr_entropy_probe_v1.py
scripts/legal_reasoning_probe_vllm_two_stage_entropy_noChunk.py
scripts/aggregate_legal_rlvr_entropy_curve_v1.py
scripts/aggregate_legal_keyword_entropy_curve_v1.py
scripts/submit_legal_rlvr_entropy_curve_v1.sh
jobs/q3_legal_rlvr_entropy_probe_v1.pbs
jobs/q3_legal_rlvr_entropy_aggregate_v1.pbs
config/legal_entropy_keywords_zh.txt
notebooks/Entropy_Probe_Qwen3_30B_RLVR_Legal.ipynb
```

## 1. Install

Upload the ZIP to:

```text
/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr/
```

Then:

```bash
PROJ=/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr
cd "$PROJ"

unzip -o legal_rlvr_entropy_curve_v1.zip

chmod +x \
  scripts/prepare_legal_entropy_probe_v1.py \
  scripts/legal_rlvr_entropy_probe_v1.py \
  scripts/aggregate_legal_rlvr_entropy_curve_v1.py \
  scripts/aggregate_legal_keyword_entropy_curve_v1.py \
  scripts/submit_legal_rlvr_entropy_curve_v1.sh

bash -n \
  scripts/submit_legal_rlvr_entropy_curve_v1.sh \
  jobs/q3_legal_rlvr_entropy_probe_v1.pbs \
  jobs/q3_legal_rlvr_entropy_aggregate_v1.pbs

python -m py_compile \
  scripts/prepare_legal_entropy_probe_v1.py \
  scripts/legal_rlvr_entropy_probe_v1.py \
  scripts/aggregate_legal_rlvr_entropy_curve_v1.py \
  scripts/aggregate_legal_keyword_entropy_curve_v1.py
```

No output from `bash -n` means the shell syntax is valid.

## 2. Resolve the two experiment-specific paths

### Legal JSONL

`LEGAL_DATA` must be the same original legal JSONL previously supplied to
`legal_reasoning_probe_vllm_two_stage_entropy_noChunk.py`. It must contain:

```text
incident
argue
law_apply_dict
source_folder
```

### RLVR snapshot root

Find the step-100 model-only snapshot:

```bash
find "$PROJ/ckpts/model_snapshots" \
  -mindepth 2 -maxdepth 2 -type d -name global_step_100 -print
```

`SNAPSHOT_ROOT` is the parent of `global_step_100`. For example:

```text
$PROJ/ckpts/model_snapshots/<deepmath-experiment-name>
```

Each nonzero snapshot must contain `.complete`, `config.json`, tokenizer files,
and safetensors weights.

## 3. Submit the main curve

Set the two exact paths:

```bash
PROJ=/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr
WORK=/groups/gcg51557/experiments/0390_rlsd

export LEGAL_DATA=/absolute/path/to/the/legal_probe_source.jsonl
export SNAPSHOT_ROOT="$PROJ/ckpts/model_snapshots/<deepmath-experiment-name>"
```

Then submit:

```bash
cd "$PROJ"

STEPS="0 10 20 30 40 50 60 70 80 90 100" \
SEED=1 \
PROBE_SEED=1 \
SAMPLE_SIZE=100 \
HINT_LEVELS=none:weak:strong \
MAX_NEW_TOKENS=4096 \
TEMPERATURE=1.0 \
TOP_P=1.0 \
TOP_K=-1 \
SCORER_MAX_MEMORY_GIB_PER_GPU=16 \
SUBMIT_AGGREGATE=1 \
bash scripts/submit_legal_rlvr_entropy_curve_v1.sh
```

This creates one GPU job per checkpoint. Each checkpoint job:

1. loads vLLM once and generates all 300 trajectories
   (`100 cases × 3 hints`);
2. exits the vLLM Python process;
3. loads the same snapshot with HF once;
4. computes full-vocabulary token entropy for all 300 trajectories.

The final aggregate job is submitted with `afterok` dependencies on all
checkpoint jobs.

The default 4096-token limit matches the supplied legal probe. If the earlier
legal runs used another value, set `MAX_NEW_TOKENS` to that exact value for
comparability.

`SCORER_MAX_MEMORY_GIB_PER_GPU=16` is an Accelerate device-map cap, not an
allocation limit for the PBS job. It prevents the 30B HF scorer from placing
almost all weights on one 80 GiB GPU and leaves room for full-vocabulary logits.

## 4. Monitor

```bash
qstat -u "$USER"
```

One run directory:

```text
outputs/legal_entropy_curve_qwen3_30b/step_050/seed_1/
```

is complete only when it contains:

```text
.complete
case_tests.jsonl
case_evals.jsonl
trajectory_entropy.jsonl
token_entropy.jsonl
generation_metadata.json
entropy_metadata.json
```

The four JSONL files preserve the interface used by the existing entropy
Notebook. `cot_entropy_*` remains as an alias of `reasoning_entropy_*`.

## 5. Results

After the aggregate job succeeds:

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
    ├── keyword_occurrences.csv
    ├── keyword_entropy_curve_summary.csv
    ├── keyword_plot_key.csv
    ├── legal_keyword_entropy_curve.png
    ├── legal_keyword_entropy_curve.pdf
    ├── legal_keyword_coverage_curve.png
    └── legal_keyword_coverage_curve.pdf
```

The recommended performance columns use all 100 cases. Columns ending in
`legacy_effective_only` reproduce the old Notebook's behavior of excluding
empty answers and are provided only for comparison.

## 6. Resume a failed checkpoint job

The PBS script supports stage-level restart:

- if complete generation files exist, generation is reused;
- if entropy files are missing, only HF scoring is rerun;
- partial file pairs fail closed.

Resubmit one step manually:

```bash
qsub \
  -v STEP=50,SNAPSHOT_ROOT="$SNAPSHOT_ROOT",BASE_MODEL="$WORK/models/Qwen3-30B-A3B-Instruct-2507",PROBE="$PROJ/data/legal_entropy_probe/legal_probe_n100_seed1.jsonl",OUT_ROOT="$PROJ/outputs/legal_entropy_curve_qwen3_30b",SEED=1,HINT_LEVELS=none:weak:strong \
  jobs/q3_legal_rlvr_entropy_probe_v1.pbs
```

If a partial output must intentionally be replaced, add:

```text
,OVERWRITE=1
```

Do not use `OVERWRITE=1` on a valid completed run.

## 7. Additional generation seeds

Keep the frozen probe unchanged while changing the generation seed:

```bash
SEED=2 PROBE_SEED=1 SUBMIT_AGGREGATE=1 \
bash scripts/submit_legal_rlvr_entropy_curve_v1.sh
```

The aggregator automatically combines seed directories and clusters confidence
intervals by legal case. Start with seed 1; add seeds 2 and 3 only if needed.

## 8. Interpretation

- `full_entropy_mean`: macro average of per-trajectory token entropy.
- `full_entropy_micro`: token-weighted entropy, closer to training
  `actor/entropy`.
- `reasoning_entropy_mean`: only tokens inside `<think>` or
  `《reasoning》`.
- `answer_entropy_mean`: only tokens inside `《answer》`, or after
  `</think>`.
- `reasoning_coverage`: fraction of trajectories with a recognized reasoning
  region.

For this Instruct-2507 model, generation defaults to
`enable_thinking=False`, matching RLVR training. If the model emits neither
`<think>` nor Unicode reasoning tags, reasoning entropy is undefined for that
trajectory, while full entropy remains valid.

Always inspect keyword coverage together with keyword entropy. A keyword curve
conditioned on a shrinking set of occurrences is not evidence of a global
entropy change.
