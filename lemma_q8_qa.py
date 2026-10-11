r"""Run Qwen3-VL multiple-choice QA on every LEMMA Question-8 (activity order) item.

Loads the model once, answers each question from the take's video(s), and appends one
JSON line per question to --output. Re-running skips questions already in --output.

Example:
    python lemma_q8_qa.py --views master --model 8b --gpu 0 \
        --output outputs/q8_master_8b.jsonl

--views takes one or more of master, fpv1, fpv2; several views are passed to the model
as separate videos in the same prompt. Model / frame-sampling options match qwen3vl_video.py.
"""
import argparse
import json
import os
import re
import sys
import time

from qwen3vl_video import MODELS, load_model, read_text

LETTERS = "ABCD"
DEFAULT_SYSTEM = "You are an expert in multi-person video understanding."
VIEW_NAMES = {
    "master": "third-person (exocentric) view",
    "fpv1": "first-person (egocentric) view of person 1",
    "fpv2": "first-person (egocentric) view of person 2",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", default="/anvil/scratch/x-hnguyen24/LEMMA/subsets/q8_all_2x1_2x2.json",
                   help="question json: {qid: {question, options, correct_answer, ...}}")
    p.add_argument("--data-root", default="/anvil/scratch/x-hnguyen24/LEMMA",
                   help="directory holding one folder per take (e.g. 26l-21-5-1/master.mp4)")
    p.add_argument("--views", nargs="+", default=["master"], choices=list(VIEW_NAMES),
                   help="which video(s) of the take to feed the model")
    p.add_argument("--output", required=True, help="results .jsonl (appended to; enables resume)")
    p.add_argument("--limit", type=int, default=None, help="only run the first N questions")

    system = p.add_mutually_exclusive_group()
    system.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt ('' to omit)")
    system.add_argument("--system-file", help="read the system prompt from this text file")

    p.add_argument("--gpu", type=int, default=0, help="physical GPU index (nvidia-smi)")
    p.add_argument("--model", default="8b", help="shortcut (%s) or Hugging Face repo id" % ", ".join(MODELS))
    p.add_argument("--quant", choices=["none", "nf4", "awq"], default=None,
                   help="4-bit quantization; defaults to the shortcut's setting, else none")
    p.add_argument("--hf-home", default="/anvil/scratch/x-hnguyen24/foundation_model_checkpoints",
                   help="Hugging Face cache root")

    # Frame sampling is per video: with several views, lower these to stay within memory.
    p.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video")
    p.add_argument("--max-frames", type=int, default=640, help="hard cap on frames per video")
    p.add_argument("--max-pixels", type=int, default=360 * 420, help="per-frame pixel budget after resizing")
    p.add_argument("--max-new-tokens", type=int, default=16)
    return p.parse_args()


def take_dir(qid):
    # "26l_21_5_1_Q8_002" -> "26l-21-5-1"
    return "-".join(qid.split("_")[:4])


def build_prompt(item, views):
    if len(views) == 1:
        intro = f"The video is the {VIEW_NAMES[views[0]]} of the same scene."
    else:
        intro = "The videos show the same scene from different cameras: " + "; ".join(
            f"video {i + 1} is the {VIEW_NAMES[v]}" for i, v in enumerate(views)) + "."
    options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(item["options"]))
    return (f"{intro}\n\nQuestion: {item['question']}\n\nOptions:\n{options}\n\n"
            "Answer with the option's letter from the given choices directly.")


def parse_choice(answer, n_options):
    """Return the 0-based option index in the model's reply, or None."""
    valid = LETTERS[:n_options]
    m = re.search(rf"\b([{valid}])\b", answer.strip())
    return valid.index(m.group(1)) if m else None


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["qid"]] = r
    return done


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
    qids = sorted(questions)[:args.limit]
    system = read_text(args.system_file) if args.system_file else args.system

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = load_done(args.output)
    todo = [q for q in qids if q not in done]
    print(f"{len(qids)} questions | {len(qids) - len(todo)} already done | views: {args.views}",
          file=sys.stderr)

    repo_id, quant = MODELS.get(args.model, (args.model, "none"))
    quant = args.quant or quant
    print(f"model: {repo_id} | quant: {quant} | GPU {args.gpu}", file=sys.stderr)
    model_path = snapshot_download(repo_id, cache_dir=os.path.join(args.hf_home, "hub"))
    model = load_model(model_path, quant)
    processor = AutoProcessor.from_pretrained(model_path)

    with open(args.output, "a") as out:
        for n, qid in enumerate(todo, 1):
            item = questions[qid]
            videos_in = [os.path.join(args.data_root, take_dir(qid), f"{v}.mp4") for v in args.views]
            missing = [v for v in videos_in if not os.path.exists(v)]
            if missing:
                print(f"[{n}/{len(todo)}] {qid}: missing {missing}, skipped", file=sys.stderr)
                continue

            messages = []
            if system:
                messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
            content = [{"type": "video", "video": v, "fps": args.fps, "max_frames": args.max_frames,
                        "max_pixels": args.max_pixels} for v in videos_in]
            content.append({"type": "text", "text": build_prompt(item, args.views)})
            messages.append({"role": "user", "content": content})

            t0 = time.time()
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos, video_kwargs = process_vision_info(
                    messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
                video_metadata = None
                if videos:
                    videos, video_metadata = map(list, zip(*videos))
                inputs = processor(text=[text], images=images, videos=videos, video_metadata=video_metadata,
                                   do_resize=False, return_tensors="pt", padding=True,
                                   **video_kwargs).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], generated)]
                answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0].strip()
            except torch.cuda.OutOfMemoryError:
                print(f"[{n}/{len(todo)}] {qid}: CUDA OOM, skipped (lower --max-frames/--max-pixels)",
                      file=sys.stderr)
                torch.cuda.empty_cache()
                continue

            pred = parse_choice(answer, len(item["options"]))
            record = {
                "qid": qid, "take": take_dir(qid), "views": args.views,
                "question": item["question"], "options": item["options"],
                "correct_answer": item["correct_answer"], "pred": pred,
                "correct": pred == item["correct_answer"], "raw_answer": answer,
                "frames": [v.shape[0] for v in videos], "input_tokens": inputs["input_ids"].shape[-1],
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[qid] = record
            print(f"[{n}/{len(todo)}] {qid}: pred={LETTERS[pred] if pred is not None else '?'} "
                  f"gt={LETTERS[item['correct_answer']]} | {time.time() - t0:.1f}s", file=sys.stderr)
            del inputs, generated
            torch.cuda.empty_cache()

    results = [done[q] for q in qids if q in done]
    n_correct = sum(r["correct"] for r in results)
    n_unparsed = sum(r["pred"] is None for r in results)
    acc = n_correct / len(results) if results else 0.0
    print(f"\naccuracy: {n_correct}/{len(results)} = {acc:.2%} | unparsed: {n_unparsed} | "
          f"results in {args.output}")


if __name__ == "__main__":
    main()
