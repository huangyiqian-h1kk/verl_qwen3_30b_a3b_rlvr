#!/bin/bash
# Separate upstream installation: do not upgrade the deployed teacher environment.
set -euo pipefail
mopd_upstream=${1:?Usage: bash scripts/mopd_bootstrap.sh NEW_VERL_DIRECTORY}
mopd_revision=482f86fd09959912c99c0acf2b90329f9e061da7
if [[ ! -e "$mopd_upstream" ]]; then
  git clone https://github.com/verl-project/verl.git "$mopd_upstream"
fi
[[ -d "$mopd_upstream/.git" ]]
[[ -z $(git -C "$mopd_upstream" status --porcelain) ]]
git -C "$mopd_upstream" fetch origin "$mopd_revision"
git -C "$mopd_upstream" checkout --detach "$mopd_revision"
cd "$mopd_upstream"
# CUDA 13 build in this exact lock. Driver/extension compatibility is checked in
# gpu-node; this command does not claim compatibility with the old CUDA 12.8 env.
uv sync --python 3.12 --frozen --all-packages --extra vllm --extra megatron
printf 'New interpreter: %s/.venv/bin/python\n' "$PWD"
