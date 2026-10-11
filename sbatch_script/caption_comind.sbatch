#!/bin/bash
# Caption CoMinD v1 videos in 30 s windows with Qwen3-VL-30B-A3B-Thinking (caption_videos_qwen3vl.py).
#
# Videos: <DATA_ROOT>/recordings/<id>/mp4s/{helper_trimmed_sync,leader_trimmed_sync,gopro_front_sync}.mp4
# Output: <OUT_ROOT>/recordings/<id>/mp4s/<stem>_qwen3vl_30s-captions.json (same folder structure)
#
# One array task = one view x one shard of the recordings, on one H100. Task i handles
# view VIEWS[i / NSHARDS], videos i % NSHARDS, i % NSHARDS + NSHARDS, ... (sorted by path).
# The array size must be 3 * NSHARDS:
#   sbatch sbatch_script/caption_comind.sbatch                                  # NSHARDS=8, 24 tasks
#   sbatch --array=0-11 --export=ALL,NSHARDS=4 sbatch_script/caption_comind.sbatch
#   sbatch --array=0 --export=ALL,EXTRA_ARGS="--max-windows 2" sbatch_script/caption_comind.sbatch  # test
# Submit from the multiviewagent/ directory (or use submit_caption_comind.sh). The script resumes per
# window, so a rerun only captions what is missing.
#
# Each array task chains itself: at start it queues a successor for the same task index (afterany, so it
# starts once this task ends or times out) and cancels it again if the captioning finishes. CHAIN_LEFT
# caps how many successors one task may queue (default 10). Runs with --max-windows do not chain.
#
#SBATCH --job-name=caption-comind
#SBATCH --account=ele260072-ai
#SBATCH --partition=ai
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=1:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out

set -euo pipefail

NSHARDS=${NSHARDS:-8}
DATA_ROOT=${DATA_ROOT:-/anvil/projects/x-ele260072/comind_v1}
OUT_ROOT=${OUT_ROOT:-/anvil/scratch/x-hnguyen24/multiviewagent/outputs/comind_v1}
BATCH_SIZE=${BATCH_SIZE:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}      # anything else, e.g. "--max-windows 2" or "--no-save-thinking"
CHAIN_LEFT=${CHAIN_LEFT:-10}    # successor jobs this task may still queue

# file stem -> --view for the prompt
VIEWS=(helper_trimmed_sync leader_trimmed_sync gopro_front_sync)
declare -A VIEW_TYPE=([helper_trimmed_sync]=ego [leader_trimmed_sync]=ego [gopro_front_sync]=exo)

REPO=/anvil/scratch/x-hnguyen24/multiviewagent
ENV=/anvil/scratch/x-hnguyen24/conda/envs/qwen3
cd "$REPO"

# torchcodec needs FFmpeg shared libs and CUDA NPP at load time
module load ffmpeg
export LD_LIBRARY_PATH=$ENV/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/anvil/scratch/x-hnguyen24/foundation_model_checkpoints
export HF_HUB_OFFLINE=1         # model is pre-downloaded; never hit the Hub from compute nodes

task=${SLURM_ARRAY_TASK_ID:-0}
view_idx=$(( task / NSHARDS ))
shard=$(( task % NSHARDS ))
if (( view_idx >= ${#VIEWS[@]} )); then
    echo "task $task out of range for NSHARDS=$NSHARDS (array must be 0-$(( 3 * NSHARDS - 1 )))" >&2
    exit 1
fi
name=${VIEWS[$view_idx]}

mapfile -t all_videos < <(find "$DATA_ROOT/recordings" -path "*/mp4s/$name.mp4" | sort)
videos=()
for (( i = shard; i < ${#all_videos[@]}; i += NSHARDS )); do
    videos+=("${all_videos[$i]}")
done

echo "job ${SLURM_JOB_ID:-local} task $task on $(hostname) | $name (${VIEW_TYPE[$name]}) | shard $shard/$NSHARDS | $(date)"
echo "${#videos[@]} of ${#all_videos[@]} videos -> $OUT_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if (( ${#videos[@]} == 0 )); then
    echo "nothing to do"
    exit 0
fi

next_job=
if [[ -n ${SLURM_ARRAY_JOB_ID:-} && $EXTRA_ARGS != *--max-windows* ]] && (( CHAIN_LEFT > 0 )); then
    next_job=$(sbatch --parsable --array="$task" --dependency="afterany:${SLURM_ARRAY_JOB_ID}_$task" \
                      --job-name="$SLURM_JOB_NAME" \
                      --export="ALL,NSHARDS=$NSHARDS,DATA_ROOT=$DATA_ROOT,OUT_ROOT=$OUT_ROOT,BATCH_SIZE=$BATCH_SIZE,CHAIN_LEFT=$((CHAIN_LEFT - 1))" \
                      "$REPO/sbatch_script/caption_comind.sbatch")
    echo "queued successor job ${next_job}_$task (CHAIN_LEFT=$((CHAIN_LEFT - 1)))"
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
srun "$ENV/bin/python" caption_videos_qwen3vl.py "${videos[@]}" \
    --out-dir "$OUT_ROOT" --input-root "$DATA_ROOT" \
    --view "${VIEW_TYPE[$name]}" --batch-size "$BATCH_SIZE" $EXTRA_ARGS

# Reaching this line means every video in the shard was processed (a timeout kills the script above).
if [[ -n $next_job ]]; then
    scancel "$next_job" && echo "shard finished; cancelled successor job $next_job"
fi
echo "done $(date)"
