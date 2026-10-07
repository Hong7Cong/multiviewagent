r"""Answer LEMMA multiple-choice questions with Qwen3-VL from two camera views at once and report accuracy.

Example:
    cd /projects/bivz/hnguyen24/multiviewagent
    python qwen3vl_qa_multiview.py --views fpv1 master --model 32b
    python qwen3vl_qa_multiview.py --views fpv1 fpv2 --model 32b-nf4 --limit 5

Both videos go into a single user turn, each preceded by a label, then the question:
    View 1 (first-person, fpv1):<video>View 2 (third-person, master):<video>The two videos are
    synchronized recordings of the same event from different cameras.
    <question + options>
--max-frames is per view (default 320, half the single-view budget), so the total visual tokens
stay close to qwen3vl_qa.py. Output and resume behave like qwen3vl_qa.py.
"""
import argparse
import json
import os
import sys
import time

from qwen3vl_qa import DEFAULT_QUESTIONS, DEFAULT_ROOT, LETTERS, format_question, load_done, parse_letter, take_of
from qwen3vl_video import DEFAULT_HF_HOME, MODELS, load_model

VIEWS = ["fpv1", "fpv2", "master"]
VIEW_KIND = {"fpv1": "first-person", "fpv2": "first-person", "master": "third-person"}
MULTIVIEW_NOTE = "The two videos are synchronized recordings of the same event from different cameras."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", nargs=2, required=True, choices=VIEWS, metavar="VIEW",
                   help="two camera views, in prompt order (choices: %s)" % ", ".join(VIEWS))
    p.add_argument("--questions", default=DEFAULT_QUESTIONS, help="question json: {id: {question, options, ...}}")
    p.add_argument("--lemma-root", default=DEFAULT_ROOT, help="directory holding <take>/<view>.mp4")
    p.add_argument("--output", default=None, help="results .jsonl (default: outputs/qa_<v1>+<v2>_<model>.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
    p.add_argument("--system", default=None, help="optional system prompt")

    p.add_argument("--gpu", type=int, default=0, help="physical GPU index (nvidia-smi)")
    p.add_argument("--model", default="8b", help="shortcut (%s) or Hugging Face repo id" % ", ".join(MODELS))
    p.add_argument("--quant", choices=["none", "nf4", "awq"], default=None,
                   help="4-bit quantization; defaults to the shortcut's setting, else none")
    p.add_argument("--hf-home", default=DEFAULT_HF_HOME, help="Hugging Face cache root")

    p.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video")
    p.add_argument("--max-frames", type=int, default=320, help="hard cap on frames per view")
    p.add_argument("--max-pixels", type=int, default=360 * 420, help="per-frame pixel budget after resizing")
    p.add_argument("--max-new-tokens", type=int, default=16)
    args = p.parse_args()
    if args.views[0] == args.views[1]:
        p.error("--views must be two different views")
    return args


def build_content(video_items, views, question_text):
    content = []
    for i, (item, view) in enumerate(zip(video_items, views), 1):
        content.append({"type": "text", "text": f"View {i} ({VIEW_KIND[view]}, {view}):"})
        content.append(item)
    content.append({"type": "text", "text": f"{MULTIVIEW_NOTE}\n{question_text}"})
    return content


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
    # group questions of the same take together so each pair of videos is decoded once
    qids = sorted(questions, key=lambda k: (take_of(k), k))[:args.limit]

    view_tag = "+".join(args.views)
    output = args.output or f"outputs/qa_{view_tag}_{args.model.replace('/', '_')}.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    done = load_done(output)
    todo = [k for k in qids if k not in done]
    print(f"{len(qids)} questions | {len(qids) - len(todo)} already in {output} | {len(todo)} to run",
          file=sys.stderr)

    if todo:
        repo_id, quant = MODELS.get(args.model, (args.model, "none"))
        quant = args.quant or quant
        print(f"model: {repo_id} | quant: {quant} | views: {view_tag} | GPU {args.gpu}", file=sys.stderr)
        model_path = snapshot_download(repo_id, cache_dir=os.path.join(args.hf_home, "hub"))
        model = load_model(model_path, quant)
        processor = AutoProcessor.from_pretrained(model_path)

    cached_take, cached_video = None, None
    t_start = time.time()
    with open(output, "a") as out:
        for n, qid in enumerate(todo, 1):
            q = questions[qid]
            take = take_of(qid)
            video_items = [{"type": "video", "video": os.path.join(args.lemma_root, take, f"{view}.mp4"),
                            "fps": args.fps, "max_frames": args.max_frames, "max_pixels": args.max_pixels}
                           for view in args.views]

            if take != cached_take:
                # Qwen3-VL uses 16px patches; video metadata gives the model real frame timestamps.
                # Videos come back in the order they appear in the message, i.e. args.views order.
                _, videos, video_kwargs = process_vision_info(
                    [{"role": "user", "content": video_items}],
                    image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
                videos, video_metadata = map(list, zip(*videos))
                cached_take, cached_video = take, (videos, video_metadata, video_kwargs)
            videos, video_metadata, video_kwargs = cached_video

            messages = []
            if args.system:
                messages.append({"role": "system", "content": [{"type": "text", "text": args.system}]})
            messages.append({"role": "user", "content": build_content(video_items, args.views, format_question(q))})
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
            frames = [v.shape[0] for v in videos]
            record = {"id": qid, "take": take, "views": args.views, "pred": pred, "gt": gt,
                      "correct": pred == gt, "response": response,
                      "frames": frames, "input_tokens": inputs["input_ids"].shape[-1]}
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[qid] = record
            print(f"[{n}/{len(todo)}] {qid} pred={pred} gt={gt} {'✓' if record['correct'] else '✗'} "
                  f"| {'+'.join(map(str, frames))} frames, {record['input_tokens']} tokens", file=sys.stderr)

    results = [done[k] for k in qids]
    correct = sum(r["correct"] for r in results)
    unparsed = sum(r["pred"] is None for r in results)
    print(f"\nviews {view_tag} | accuracy {correct}/{len(results)} = {correct / max(len(results), 1):.1%} "
          f"| unparsed {unparsed} | {time.time() - t_start:.0f}s | results in {output}")


if __name__ == "__main__":
    main()
