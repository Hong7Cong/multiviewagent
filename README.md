# multiviewagent

Run Qwen3-VL on a local video and save the model's answer to a text file.

## Setup

```bash
pip install "transformers>=4.57.0" accelerate "qwen-vl-utils[decord]"
pip install bitsandbytes   # for --model 32b-nf4
# gptqmodel needs pypcre; on hosts with an old system PCRE2 (e.g. RHEL 8) build it first:
# PYPCRE_BUILD_FROM_SOURCE=1 CC=gcc CFLAGS=-fPIC pip install --no-cache-dir pypcre
pip install gptqmodel       # for --model 32b-awq (transformers 5 loads AWQ via gptqmodel, not autoawq)
```

## Usage

```bash
python qwen3vl_video.py \
    --video path/to/video.mp4 \
    --system "You are an expert in egocentric video understanding." \
    --prompt "Describe interactions between the camera wearer and other people." \
    --output outputs/answer.txt \
    --model 8b --gpu 0
```

| `--model`  | Checkpoint                                         | Peak GPU memory (32 frames) |
|------------|----------------------------------------------------|-----------------------------|
| `8b`       | `Qwen/Qwen3-VL-8B-Instruct`, bf16                  | ~18 GB |
| `32b-nf4`  | `Qwen/Qwen3-VL-32B-Instruct` + bitsandbytes NF4    | ~25 GB |
| `32b-awq`  | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ`              | ~22 GB |

Prompts can be read from files with `--prompt-file` / `--system-file`.
Frame sampling: `--fps`, `--max-frames`, `--max-pixels`. Model cache: `--hf-home`.
See `python qwen3vl_video.py -h` for all options.

## LEMMA question answering

`qwen3vl_qa.py` answers the multiple-choice questions in
`LEMMA/subsets/q8_all_2x1_2x2.json` from one camera view and reports accuracy:

```bash
python qwen3vl_qa.py --view fpv1      # or fpv2 / master
```

Results go to `outputs/qa_<view>_<model>.jsonl`, one line per question. Rerunning the
same command skips questions already in that file. Delete the file to start over.
Use `--limit N` for a quick test and `--model` / `--output` like above.

### Two views

`qwen3vl_qa_multiview.py` feeds two synchronized views to one model in a single prompt:

```bash
python qwen3vl_qa_multiview.py --views fpv1 master --model 32b
```

Each video is preceded by a label (`View 1 (first-person, fpv1):`, `View 2 (third-person, master):`).
`--max-frames` applies per view and defaults to 320, half the single-view budget.
Results go to `outputs/qa_<v1>+<v2>_<model>.jsonl`.

`qwen3vl_video.ipynb` is the interactive notebook version (8B only).

## CoMinD helper vs leader comparison

`compare_views_qwen3.py` reads the per-window helper and leader captions written by
`caption_videos_qwen3vl.py` and asks a text-only Qwen3 model which actions were seen only by the
helper, only by the leader, or by both. Output goes next to the captions as
`recordings/<id>/mp4s/helper_vs_leader_<model>_30s-events.json`. Reruns skip windows already done.

| `--model`  | Checkpoint                              | GPU memory |
|------------|-----------------------------------------|------------|
| `8b`       | `Qwen/Qwen3-8B`, bf16                   | ~16 GB |
| `32b`      | `Qwen/Qwen3-32B`, bf16                  | ~66 GB (GH200) |
| `32b-nf4`  | `Qwen/Qwen3-32B` + bitsandbytes NF4     | ~20 GB |
| `32b-awq`  | `Qwen/Qwen3-32B-AWQ`                    | ~20 GB |

To run every recording on DeltaAI, use the submit script. It downloads the model on the login node,
then submits `sbatch_script/compare_views_comind.sbatch` as an array of `NSHARDS` GPU tasks (default 8),
each of which re-queues itself until its recordings are done:

```bash
bash sbatch_script/submit_compare_views_comind.sh                  # Qwen3-8B
MODEL=32b-awq bash sbatch_script/submit_compare_views_comind.sh    # Qwen3-32B-AWQ
MODEL=32b NSHARDS=16 EXTRA_ARGS="--think" bash sbatch_script/submit_compare_views_comind.sh
squeue -u $USER -n compare-views
```

Other settings: `CAPTION_ROOT` (default `outputs/comind_v1`), `BATCH_SIZE`, `CHAIN_LEFT`, and
`DRY_RUN=1` to print the `sbatch` command without submitting. The `qwen3vl` env has no
bitsandbytes, so `32b-nf4` does not run there.
