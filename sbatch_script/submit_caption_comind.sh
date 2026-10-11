#!/bin/bash
# Caption the whole CoMinD v1 dataset: submit caption_comind.sbatch as a 3 * NSHARDS task array
# (helper, leader, gopro_front views). Each task re-queues itself until its shard is done.
#   bash sbatch_script/submit_caption_comind.sh                 # NSHARDS=8 -> 24 tasks
#   NSHARDS=4 CHAIN_LEFT=20 bash sbatch_script/submit_caption_comind.sh
# Add DRY_RUN=1 to only print the sbatch command. Monitor with: squeue -u $USER -n caption-comind
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/slurm

NSHARDS=${NSHARDS:-8}
CHAIN_LEFT=${CHAIN_LEFT:-10}
OUT_ROOT=${OUT_ROOT:-/anvil/scratch/x-hnguyen24/multiviewagent/outputs/comind_v1}

cmd=(sbatch --array="0-$(( 3 * NSHARDS - 1 ))" --job-name=caption-comind
     --export="ALL,NSHARDS=$NSHARDS,CHAIN_LEFT=$CHAIN_LEFT,OUT_ROOT=$OUT_ROOT"
     sbatch_script/caption_comind.sbatch)
echo "${cmd[*]}"
[[ ${DRY_RUN:-0} == 1 ]] || "${cmd[@]}"
