#!/bin/bash
# Four-domain entrypoint. All paths are anchored to this worktree.
set -euo pipefail
MD_FOUR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MD_STAGE=${1:-accept-four}
if [[ $# -gt 0 ]]; then shift; fi
case "$MD_STAGE" in
  accept-four|prepare-data|check-verifiers|check-infra|smoke|backward|baseline|train|evaluate) ;;
  *) echo "Unsupported four-domain stage: $MD_STAGE" >&2; exit 2 ;;
esac
cd "$MD_FOUR_ROOT"
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate /groups/gcg51557/experiments/0390_rlsd/envs/verl_qwen3_moe_megatron_py312_cu128
source "$MD_FOUR_ROOT/scripts/multidomain/env.sh"
# Preserve the original parent pool. The installer writes its absolute location.
MD_PARENT_DATA=/groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl/data/multidomain/all_six_v2_ep8
python -m multidomain.submit \
  --stage "$MD_STAGE" \
  --config config/multidomain/four_domain.yaml \
  --data-id four_domains_v1 \
  --run-id qwen30b_d4_uniform_ctx64k_seed42_v1 \
  --source-data-dir "$MD_PARENT_DATA" \
  --source-manifest-sha256 4c3f7a46c51f85167d88300dfd2aee2f8f20c4cfd388aa75acf8afb56b8bdf23 \
  "$@"
