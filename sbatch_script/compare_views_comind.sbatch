#!/bin/bash
# Compare helper vs leader captions of every CoMinD v1 recording with Qwen3 (compare_views_qwen3.py), on DeltaAI.
#
# Input:  <CAPTION_ROOT>/recordings/<id>/mp4s/{helper,leader}_trimmed_sync_qwen3vl_30s-captions.json
# Output: <CAPTION_ROOT>/recordings/<id>/mp4s/helper_vs_leader_<model tag>_30s-events.json
#
# One array task = one shard of the recordings (those with both helper and leader captions), on one GH200.
# Task i handles recordings i, i + NSHARDS, ... (sorted). The array size must be NSHARDS:
#   sbatch --array=0-7 --export=ALL,NSHARDS=8,MODEL=32b-awq sbatch_script/compare_views_comind.sbatch
#   sbatch --array=0 --export=ALL,EXTRA_ARGS="--max-windows 2" sbatch_script/compare_views_comind.sbatch  # test
# Usually submitted through submit_compare_views_comind.sh, which pre-downloads the model first (compute
# nodes run with HF_HUB_OFFLINE=1). The script resumes per window, so a rerun only compares what is missing.
#
# Each array task chains itself: at start it queues a successor for the same task index (afterany, so it
# starts once this task ends or times out) and cancels it again if its shard finishes. CHAIN_LEFT caps how
# many successors one task may queue (default 10). Runs with --max-windows do not chain.
#
#SBATCH --job-name=compare-views
#SBATCH --account=bivz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out

set -euo pipefail

NSHARDS=${NSHARDS:-8}
MODEL=${MODEL:-8b}              # 8b | 32b | 32b-awq (32b-nf4 needs bitsandbytes, not in the qwen3vl env)
CAPTION_ROOT=${CAPTION_ROOT:-/projects/bivz/hnguyen24/multiviewagent/outputs/comind_v1}
BATCH_SIZE=${BATCH_SIZE:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}      # anything else, e.g. "--think", "--save-raw", "--max-windows 2"
CHAIN_LEFT=${CHAIN_LEFT:-10}    # successor jobs this task may still queue

REPO=/projects/bivz/hnguyen24/multiviewagent
PYTHON=/projects/bivz/hnguyen24/envs/qwen3vl/bin/python
HF_HOME_DIR=/work/hdd/bivz/hnguyen24/hf
cd "$REPO"

export HF_HOME=$HF_HOME_DIR
export HF_HUB_OFFLINE=1         # model is pre-downloaded; never hit the Hub from compute nodes

task=${SLURM_ARRAY_TASK_ID:-0}
if (( task >= NSHARDS )); then
    echo "task $task out of range for NSHARDS=$NSHARDS (array must be 0-$(( NSHARDS - 1 )))" >&2
    exit 1
fi

all_dirs=()
while read -r f; do
    d=$(dirname "$f")
    [[ -f $d/leader_trimmed_sync_qwen3vl_30s-captions.json ]] && all_dirs+=("$d")
done < <(find "$CAPTION_ROOT/recordings" -path "*/mp4s/helper_trimmed_sync_qwen3vl_30s-captions.json" | sort)
dirs=()
for (( i = task; i < ${#all_dirs[@]}; i += NSHARDS )); do
    dirs+=("${all_dirs[$i]}")
done

echo "job ${SLURM_JOB_ID:-local} task $task on $(hostname) | model $MODEL | shard $task/$NSHARDS | $(date)"
echo "${#dirs[@]} of ${#all_dirs[@]} recordings in $CAPTION_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if (( ${#dirs[@]} == 0 )); then
    echo "nothing to do"
    exit 0
fi

next_job=
if [[ -n ${SLURM_ARRAY_JOB_ID:-} && $EXTRA_ARGS != *--max-windows* ]] && (( CHAIN_LEFT > 0 )); then
    next_job=$(sbatch --parsable --array="$task" --dependency="afterany:${SLURM_ARRAY_JOB_ID}_$task" \
                      --job-name="$SLURM_JOB_NAME" \
                      --export="ALL,NSHARDS=$NSHARDS,MODEL=$MODEL,CAPTION_ROOT=$CAPTION_ROOT,BATCH_SIZE=$BATCH_SIZE,EXTRA_ARGS=$EXTRA_ARGS,CHAIN_LEFT=$((CHAIN_LEFT - 1))" \
                      "$REPO/sbatch_script/compare_views_comind.sbatch")
    echo "queued successor job ${next_job}_$task (CHAIN_LEFT=$((CHAIN_LEFT - 1)))"
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
srun "$PYTHON" compare_views_qwen3.py "${dirs[@]}" \
    --model "$MODEL" --hf-home "$HF_HOME_DIR" --batch-size "$BATCH_SIZE" $EXTRA_ARGS

# Reaching this line means every recording in the shard was processed (a timeout kills the script above).
if [[ -n $next_job ]]; then
    scancel "$next_job" && echo "shard finished; cancelled successor job $next_job"
fi
echo "done $(date)"
