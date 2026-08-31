# Nemotron-Math-v2 baseline v1

This package evaluates whether `nvidia/Nemotron-Math-v2` has a useful GRPO
difficulty distribution for the existing ABCI experiment:

- model: `Qwen3-30B-A3B-Instruct-2507`
- framework environment: existing verl/vLLM environment
- prompt: existing Unicode explicit-reasoning system prompt
- verifier: existing `rewards/unicode_math_reward_v4.py`
- sample: random, reproducible problem-level sample
- rollout group: `K=8`
- maximum response length: `8192`

It deliberately does not use the old baseline evaluator's `head(256)`,
normalized string equality, or unconditional format success.

## Files

- `scripts/prepare_nemotron_math_v2_baseline.py`
  - runs on an ABCI login node with network access;
  - samples random pages across all five trajectory splits through the
    Hugging Face Dataset Viewer API;
  - avoids downloading the approximately 143 GB complete repository;
  - corrects row-level sampling bias with inverse trajectory-multiplicity
    acceptance, then deduplicates problem UUIDs;
  - rejects rows whose answer was replaced by model majority vote;
  - checks the exact rendered prompt length with the local tokenizer.
- `scripts/eval_nemotron_math_v2_baseline.py`
  - runs offline on a compute node;
  - requests eight generations per prompt;
  - calls the same custom reward module used by training;
  - records every rollout and produces prompt-group statistics.
- `jobs/q3_baseline_nemotron_math_v2_v1.pbs`
  - ABCI PBS job for the offline evaluation.

## 1. Install the package in the project root

Upload and unzip this package under:

```text
/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr/
```

The resulting files must be:

```text
scripts/prepare_nemotron_math_v2_baseline.py
scripts/eval_nemotron_math_v2_baseline.py
jobs/q3_baseline_nemotron_math_v2_v1.pbs
```

Set executable bits and check syntax:

```bash
PROJ=/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr
cd "$PROJ"

chmod +x \
  scripts/prepare_nemotron_math_v2_baseline.py \
  scripts/eval_nemotron_math_v2_baseline.py

python -m py_compile \
  scripts/prepare_nemotron_math_v2_baseline.py \
  scripts/eval_nemotron_math_v2_baseline.py

bash -n jobs/q3_baseline_nemotron_math_v2_v1.pbs
```

## 2. Prepare the sample on a login node

Do not submit this step through PBS. ABCI compute nodes do not have network
access.

```bash
PROJ=/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr
WORK=/groups/gcg51557/experiments/0390_rlsd

source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate "$WORK/envs/verl_qwen3_moe_megatron_py312_cu128"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

python "$PROJ/scripts/prepare_nemotron_math_v2_baseline.py" \
  --output "$PROJ/data/nemotron_math_v2/nemotron_math_v2_baseline_n256_seed20260727.parquet" \
  --dataset nvidia/Nemotron-Math-v2 \
  --split all \
  --sample-size 256 \
  --sample-reserve 64 \
  --seed 20260727 \
  --tokenizer-path "$WORK/models/Qwen3-30B-A3B-Instruct-2507" \
  --system-prompt-file "$PROJ/config/unicode_system_prompt.txt" \
  --max-prompt-tokens 3072
```

Expected files:

```text
data/nemotron_math_v2/nemotron_math_v2_baseline_n256_seed20260727.parquet
data/nemotron_math_v2/nemotron_math_v2_baseline_n256_seed20260727.audit.json
```

The audit must show:

```json
"written_sample_size": 256,
"changed_answer_to_majority_count": 0
```

`low`, `medium`, and `high` name generation-budget regimes, not problem
difficulty strata. The rows API stores one row per retained correct trajectory,
so naive row sampling would overrepresent problems with many successful
trajectories. The preparer samples across all five physical splits, accepts
each trajectory with probability `1 / total_retained_trajectories_for_uuid`,
and then deduplicates by UUID. This produces an approximately uniform
problem-level sample rather than an easy-biased trajectory sample.

## 3. Submit the offline baseline

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr
qsub jobs/q3_baseline_nemotron_math_v2_v1.pbs
```

Do not submit two copies concurrently. The PBS script uses a lock and will
reject the second copy.

## 4. Outputs

The job writes:

```text
outputs/baseline_eval/nemotron_math_v2/*.jsonl
outputs/baseline_eval/nemotron_math_v2/*.summary.json
outputs/baseline_eval/nemotron_math_v2/*.groups.parquet
```

The JSONL contains every rollout, including:

```text
uid, problem, ground_truth, prediction, correct, format_ok,
prompt_tokens, response_tokens, finish_reason, truncated, reward, text
```

The summary reports:

```text
accuracy_mean_over_NK
first_sample_accuracy
pass_at_K
all_correct_ratio
all_wrong_ratio
mixed_group_ratio
format_rate
mean_reward
response-token percentiles
truncation_rate
success_hist
per-source metrics
```

`accuracy_mean_over_NK` is the appropriate empirical pass@1 estimate: it uses
all `N × K` sampled completions. `pass_at_K` is the fraction of prompts with at
least one correct completion.

The groups parquet adds:

```text
success_count
correct_rate
format_count
difficulty_class
```

Difficulty classes are:

- `primary_eligible`: 1–6 correct out of 8;
- `borderline_easy`: 7 correct out of 8;
- `all_wrong`: 0 correct;
- `all_correct`: 8 correct.

## 5. Decision rule

The script prints one of:

- `SUITABLE`
- `USABLE_AFTER_DIFFICULTY_FILTERING`
- `TOO_EASY_UNFILTERED`
- `TOO_HARD_UNFILTERED`
- `BORDERLINE`

The primary target is:

```text
accuracy_mean_over_NK: 0.10–0.65
mixed_group_ratio: at least 0.60
all_correct_ratio: at most 0.25
all_wrong_ratio: at most 0.50
```

This is a screening rule, not a statistical law. The actual decision should
also inspect:

- the success histogram;
- AoPS versus StackExchange source metrics;
- reward/parser false positives;
- response truncation;
- a manual sample of correct and incorrect outputs.

Do not start a 500-step run from the 256-row sample. If this baseline is
suitable, the next step is to construct a larger problem-level pool and run
the same model-aware difficulty filtering before creating train/holdout data.
