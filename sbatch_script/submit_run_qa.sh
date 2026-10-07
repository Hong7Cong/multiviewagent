#!/bin/bash
# Submit one run_qa.sbatch job per view setting: 3 single views + 3 view pairs, all with MODEL.
#   bash sbatch_script/submit_all.sh            # 32b
#   MODEL=32b-nf4 bash sbatch_script/submit_all.sh
# Add DRY_RUN=1 to only print the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/slurm

MODEL=${MODEL:-32b-awq}
# for views in fpv1 fpv2 master "fpv1 fpv2" "fpv1 master" "fpv2 master"; do
for views in fpv1 fpv2 master "fpv1 fpv2" "fpv1 master" "fpv2 master"; do
    cmd=(sbatch --job-name="qa-${views// /+}-$MODEL" --export="ALL,VIEWS=$views,MODEL=$MODEL"
         sbatch_script/run_qa.sbatch)
    echo "${cmd[*]}"
    [[ ${DRY_RUN:-0} == 1 ]] || "${cmd[@]}"
done
