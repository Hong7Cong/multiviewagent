#!/bin/bash
# Smoke test for compare_views_qwen3.py: compare helper vs leader captions of one recording with Qwen3-8B.
#
# Recording: RECORDING=<id> (default: first recording, sorted, that has both helper and leader captions).
# Output:    <CAPTION_ROOT>/recordings/<id>/mp4s/helper_vs_leader_qwen3-8b_30s-events.json
#   sbatch sbatch_script/compare_views_smoke.sbatch
#   sbatch --export=ALL,RECORDING=01d63ce5-f734-4647-93c8-ea4af07b8c7f sbatch_script/compare_views_smoke.sbatch
#   sbatch --export=ALL,EXTRA_ARGS="--max-windows 8 --think" sbatch_script/compare_views_smoke.sbatch
# Submit from the multiviewagent/ directory. The script resumes per window, so a rerun only does what is missing.
#
#SBATCH --job-name=compare-views-smoke
#SBATCH --account=ele260072-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail

CAPTION_ROOT=${CAPTION_ROOT:-/anvil/scratch/x-hnguyen24/multiviewagent/outputs/comind_v1}
RECORDING=${RECORDING:-}
BATCH_SIZE=${BATCH_SIZE:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}      # anything else, e.g. "--max-windows 4", "--think", "--save-raw"

REPO=/anvil/scratch/x-hnguyen24/multiviewagent
ENV=/anvil/scratch/x-hnguyen24/conda/envs/qwen3
cd "$REPO"

export HF_HOME=/anvil/scratch/x-hnguyen24/foundation_model_checkpoints
export HF_HUB_OFFLINE=1         # model is pre-downloaded; never hit the Hub from compute nodes

if [[ -z $RECORDING ]]; then
    for d in $(find "$CAPTION_ROOT/recordings" -path "*/mp4s/helper_trimmed_sync_qwen3vl_30s-captions.json" | sort); do
        d=$(dirname "$d")
        if [[ -f $d/leader_trimmed_sync_qwen3vl_30s-captions.json ]]; then
            RECORDING=$(basename "$(dirname "$d")")
            break
        fi
    done
fi
mp4s_dir=$CAPTION_ROOT/recordings/$RECORDING/mp4s

echo "job ${SLURM_JOB_ID:-local} on $(hostname) | recording $RECORDING | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
srun "$ENV/bin/python" compare_views_qwen3.py "$mp4s_dir" --batch-size "$BATCH_SIZE" $EXTRA_ARGS

echo "done $(date)"
