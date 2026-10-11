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
