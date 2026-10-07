r"""Answer LEMMA multiple-choice questions with Qwen3-VL from one camera view and report accuracy.

Example:
    cd /projects/bivz/hnguyen24/multiviewagent
    python qwen3vl_qa.py --view fpv1 --gpu 0
    python qwen3vl_qa.py --view master --model 32b-awq --output outputs/qa_master_32b.jsonl

Question ids look like "26l_21_5_1_Q8_002"; the video is <lemma-root>/26l-21-5-1/<view>.mp4.
One JSON line per question is appended to --output, so an interrupted run resumes where it stopped
(delete the file to start over). Accuracy is printed at the end.
"""
import argparse
import json
import os
import re
import sys
import time

from qwen3vl_video import DEFAULT_HF_HOME, MODELS, load_model

LETTERS = "ABCD"
DEFAULT_QUESTIONS = "/projects/bivz/hnguyen24/LEMMA/subsets/q8_all_2x1_2x2.json"
DEFAULT_ROOT = "/projects/bivz/hnguyen24/LEMMA"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", required=True, choices=["fpv1", "fpv2", "master"], help="camera view to answer from")
    p.add_argument("--questions", default=DEFAULT_QUESTIONS, help="question json: {id: {question, options, ...}}")
    p.add_argument("--lemma-root", default=DEFAULT_ROOT, help="directory holding <take>/<view>.mp4")
    p.add_argument("--output", default=None, help="results .jsonl (default: outputs/qa_<view>_<model>.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
    p.add_argument("--system", default=None, help="optional system prompt")

    p.add_argument("--gpu", type=int, default=0, help="physical GPU index (nvidia-smi)")
    p.add_argument("--model", default="8b", help="shortcut (%s) or Hugging Face repo id" % ", ".join(MODELS))
    p.add_argument("--quant", choices=["none", "nf4", "awq"], default=None,
                   help="4-bit quantization; defaults to the shortcut's setting, else none")
    p.add_argument("--hf-home", default=DEFAULT_HF_HOME, help="Hugging Face cache root")

    p.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video")
    p.add_argument("--max-frames", type=int, default=640, help="hard cap on frames fed to the model")
    p.add_argument("--max-pixels", type=int, default=360 * 420, help="per-frame pixel budget after resizing")
    p.add_argument("--max-new-tokens", type=int, default=16)
    return p.parse_args()


def take_of(qid):
    """'26l_21_5_1_Q8_002' -> '26l-21-5-1'"""
    return "-".join(qid.split("_")[:4])


def format_question(q):
    options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(q["options"]))
    return (f"{q['question']}\n{options}\n"
            "Answer with the option's letter from the given choices directly.")


def parse_letter(response):
    m = re.search(r"\b([A-D])\b", response.strip())
    return m.group(1) if m else None


def load_done(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def main():
    args = parse_args()

    # Must be set before torch / transformers / huggingface_hub are imported.
    os.environ["HF_HOME"] = args.hf_home
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from huggingface_hub import snapshot_download
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    with open(args.questions) as f:
        questions = json.load(f)
    # group questions of the same take together so each video is decoded once
    qids = sorted(questions, key=lambda k: (take_of(k), k))[:args.limit]

    output = args.output or f"outputs/qa_{args.view}_{args.model.replace('/', '_')}.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    done = load_done(output)
    todo = [k for k in qids if k not in done]
    print(f"{len(qids)} questions | {len(qids) - len(todo)} already in {output} | {len(todo)} to run",
          file=sys.stderr)

    if todo:
        repo_id, quant = MODELS.get(args.model, (args.model, "none"))
        quant = args.quant or quant
        print(f"model: {repo_id} | quant: {quant} | view: {args.view} | GPU {args.gpu}", file=sys.stderr)
        model_path = snapshot_download(repo_id, cache_dir=os.path.join(args.hf_home, "hub"))
        model = load_model(model_path, quant)
        processor = AutoProcessor.from_pretrained(model_path)

    cached_take, cached_video = None, None
    t_start = time.time()
    with open(output, "a") as out:
        for n, qid in enumerate(todo, 1):
            q = questions[qid]
            take = take_of(qid)
            video_path = os.path.join(args.lemma_root, take, f"{args.view}.mp4")
            video_item = {"type": "video", "video": video_path, "fps": args.fps,
                          "max_frames": args.max_frames, "max_pixels": args.max_pixels}

            if take != cached_take:
                # Qwen3-VL uses 16px patches; video metadata gives the model real frame timestamps.
                _, videos, video_kwargs = process_vision_info(
                    [{"role": "user", "content": [video_item]}],
                    image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
                videos, video_metadata = map(list, zip(*videos))
                cached_take, cached_video = take, (videos, video_metadata, video_kwargs)
            videos, video_metadata, video_kwargs = cached_video

            messages = []
            if args.system:
                messages.append({"role": "system", "content": [{"type": "text", "text": args.system}]})
            messages.append({"role": "user", "content": [video_item, {"type": "text", "text": format_question(q)}]})
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], videos=videos, video_metadata=video_metadata,
                               do_resize=False,  # frames were already resized by qwen_vl_utils
                               return_tensors="pt", padding=True, **video_kwargs).to(model.device)

            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            response = processor.batch_decode(generated[:, inputs["input_ids"].shape[-1]:],
                                              skip_special_tokens=True)[0].strip()

            pred = parse_letter(response)
            gt = LETTERS[q["correct_answer"]]
            record = {"id": qid, "take": take, "view": args.view, "pred": pred, "gt": gt,
                      "correct": pred == gt, "response": response,
                      "frames": videos[0].shape[0], "input_tokens": inputs["input_ids"].shape[-1]}
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[qid] = record
            print(f"[{n}/{len(todo)}] {qid} pred={pred} gt={gt} {'✓' if record['correct'] else '✗'} "
                  f"| {record['frames']} frames, {record['input_tokens']} tokens", file=sys.stderr)

    results = [done[k] for k in qids]
    correct = sum(r["correct"] for r in results)
    unparsed = sum(r["pred"] is None for r in results)
    print(f"\nview {args.view} | accuracy {correct}/{len(results)} = {correct / max(len(results), 1):.1%} "
          f"| unparsed {unparsed} | {time.time() - t_start:.0f}s | results in {output}")


if __name__ == "__main__":
    main()
