#!/bin/bash
# Compare helper vs leader captions for the whole CoMinD v1 dataset on DeltaAI: download the Qwen3 model on
# the login node (compute nodes run offline), then submit compare_views_comind.sbatch as an NSHARDS task
# array. Each task re-queues itself until its shard is done.
#   bash sbatch_script/submit_compare_views_comind.sh                         # Qwen3-8B, 8 tasks
#   MODEL=32b-awq bash sbatch_script/submit_compare_views_comind.sh           # Qwen3-32B-AWQ
#   MODEL=32b NSHARDS=16 EXTRA_ARGS="--think" bash sbatch_script/submit_compare_views_comind.sh
# Add DRY_RUN=1 to only print the sbatch command (skips the download). Monitor: squeue -u $USER -n compare-views
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/slurm

NSHARDS=${NSHARDS:-8}
MODEL=${MODEL:-8b}
CHAIN_LEFT=${CHAIN_LEFT:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}
CAPTION_ROOT=${CAPTION_ROOT:-/projects/bivz/hnguyen24/multiviewagent/outputs/comind_v1}
PYTHON=/projects/bivz/hnguyen24/envs/qwen3vl/bin/python
HF_HOME_DIR=/work/hdd/bivz/hnguyen24/hf

if [[ ${DRY_RUN:-0} != 1 ]]; then
    MODEL=$MODEL HF_HOME_DIR=$HF_HOME_DIR "$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download
MODELS = {"8b": "Qwen/Qwen3-8B", "32b": "Qwen/Qwen3-32B", "32b-nf4": "Qwen/Qwen3-32B", "32b-awq": "Qwen/Qwen3-32B-AWQ"}
repo_id = MODELS.get(os.environ["MODEL"], os.environ["MODEL"])
print("model ready:", snapshot_download(repo_id, cache_dir=os.path.join(os.environ["HF_HOME_DIR"], "hub")))
PY
fi

cmd=(sbatch --array="0-$(( NSHARDS - 1 ))" --job-name=compare-views
     --export="ALL,NSHARDS=$NSHARDS,MODEL=$MODEL,CHAIN_LEFT=$CHAIN_LEFT,BATCH_SIZE=$BATCH_SIZE,EXTRA_ARGS=$EXTRA_ARGS,CAPTION_ROOT=$CAPTION_ROOT"
     sbatch_script/compare_views_comind.sbatch)
echo "${cmd[*]}"
[[ ${DRY_RUN:-0} == 1 ]] || "${cmd[@]}"
