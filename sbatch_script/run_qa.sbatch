#!/bin/bash
# Run Qwen3-VL on all LEMMA 2x1/2x2 questions (Q1-Q8, 1341 questions) on one GH200.
#
# One view  -> qwen3vl_qa.py,  two views -> qwen3vl_qa_multiview.py. Pick with env vars:
#   sbatch sbatch_script/run_qa.sbatch                                         # fpv1, 8b
#   sbatch --export=ALL,VIEWS=master sbatch_script/run_qa.sbatch
#   sbatch --export=ALL,VIEWS="fpv1 master",MODEL=32b sbatch_script/run_qa.sbatch
# Submit from the multiviewagent/ directory. Results append to outputs/qa_all_<views>_<model>.jsonl,
# so resubmitting the same job resumes where a timed-out or failed run stopped.
#
# Jobs are short (30 min) so they get scheduled sooner. Until every question has an answer, each job
# queues its successor (afterany, so it starts once this one ends or times out) and cancels it again
# if it finishes everything. CHAIN_LEFT caps how many successors are queued (default 20). Runs with
# --limit in EXTRA_ARGS do not chain.
#
#SBATCH --job-name=qwen3vl-qa
#SBATCH --account=bivz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail

VIEWS=${VIEWS:-fpv1}            # "fpv1", "fpv2", "master", or two of them, e.g. "fpv1 master"
MODEL=${MODEL:-32b-awq}             # 8b, 32b, 32b-nf4, 32b-awq (see qwen3vl_video.py)
QUESTIONS=${QUESTIONS:-/projects/bivz/hnguyen24/LEMMA/subsets/2x1_2x2_qa.json}
EXTRA_ARGS=${EXTRA_ARGS:-}      # anything else, e.g. "--limit 5" or "--max-frames 160"
CHAIN_LEFT=${CHAIN_LEFT:-20}    # successor jobs this job may still queue

REPO=/projects/bivz/hnguyen24/multiviewagent
PYTHON=/projects/bivz/hnguyen24/envs/qwen3vl/bin/python
cd "$REPO"

read -ra view_list <<< "$VIEWS"
view_tag=$(IFS=+; echo "${view_list[*]}")
output=outputs/qa_all_${view_tag}_${MODEL//\//_}.jsonl

case ${#view_list[@]} in
    1) cmd=("$PYTHON" qwen3vl_qa.py --view "${view_list[0]}") ;;
    2) cmd=("$PYTHON" qwen3vl_qa_multiview.py --views "${view_list[@]}") ;;
    *) echo "VIEWS must name one or two views, got: '$VIEWS'" >&2; exit 1 ;;
esac

remaining() {  # questions in $QUESTIONS without a line in $output
    python3 - "$QUESTIONS" "$output" <<'PY'
import json, os, sys
questions, output = sys.argv[1:]
done = set()
if os.path.exists(output):
    with open(output) as f:
        done = {json.loads(line)["id"] for line in f if line.strip()}
print(len(set(json.load(open(questions))) - done))
PY
}

# Slurm already restricts CUDA_VISIBLE_DEVICES to the allocated GPU; pass it through unchanged.
gpu=${CUDA_VISIBLE_DEVICES:-0}
gpu=${gpu%%,*}

echo "job $SLURM_JOB_ID on $(hostname) | views: $VIEWS | model: $MODEL | GPU $gpu | $(date)"
echo "questions: $QUESTIONS"
echo "output: $output"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

left=$(remaining)
echo "$left questions left"
if [[ $left == 0 ]]; then
    echo "nothing to do"
    exit 0
fi

next_job=
if [[ $EXTRA_ARGS != *--limit* ]] && (( CHAIN_LEFT > 0 )); then
    next_job=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" --job-name="$SLURM_JOB_NAME" \
                      --export="ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1))" "$REPO/sbatch_script/run_qa.sbatch")
    echo "queued successor job $next_job (CHAIN_LEFT=$((CHAIN_LEFT - 1)))"
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
srun "${cmd[@]}" --model "$MODEL" --gpu "$gpu" --questions "$QUESTIONS" --output "$output" $EXTRA_ARGS

left=$(remaining)
if [[ $left == 0 && -n $next_job ]]; then
    scancel "$next_job" && echo "all questions answered; cancelled successor job $next_job"
fi
echo "done $(date) | $left questions left"
