"""
Compare the helper and leader egocentric captions of each 30 s window with a text-only Qwen3 LLM (8B or 32B) and
list the actions/events seen only from the helper view, only from the leader view, and from both.

Inputs are the per-window caption JSONs written by caption_videos_qwen3vl.py:
    <caption-root>/recordings/<id>/mp4s/helper_trimmed_sync_qwen3vl_30s-captions.json
    <caption-root>/recordings/<id>/mp4s/leader_trimmed_sync_qwen3vl_30s-captions.json
Windows are matched by index; each window's caption and (if saved) the captioner's thinking trace from both
views go into one prompt. Output, next to the inputs, is one JSON per recording:
    <id>/mp4s/helper_vs_leader_<model>_30s-events.json
a list with one entry per window:
    {"window", "start_sec", "end_sec", "start_t", "end_t",
     "helper_event_only": [str], "leader_event_only": [str], "same_event_from_both_view": [str], ...}
The file is rewritten after every batch, so an interrupted run resumes where it left off.

Example:
    python compare_views_qwen3.py --caption-root outputs/comind_v1
    python compare_views_qwen3.py outputs/comind_v1/recordings/<id>/mp4s --max-windows 2   # quick test
    python compare_views_qwen3.py --caption-root outputs/comind_v1 --think   # let Qwen3 reason first
    python compare_views_qwen3.py --caption-root outputs/comind_v1 --model 32b-awq

Model shortcuts (--model):
    8b         Qwen/Qwen3-8B, bf16                  ~16 GB
    32b        Qwen/Qwen3-32B, bf16                 ~66 GB  (needs an 80 GB+ GPU, e.g. GH200)
    32b-nf4    Qwen/Qwen3-32B, bitsandbytes NF4     ~20 GB  (downloads ~66 GB bf16 weights)
    32b-awq    Qwen/Qwen3-32B-AWQ, AWQ 4-bit        ~20 GB  (downloads ~20 GB)
--quant {none,nf4,awq} is set by the shortcut; pass it explicitly for other repo ids.
"""

import argparse
import json
import os
from pathlib import Path
import re
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_HF_HOME = "/anvil/scratch/x-hnguyen24/foundation_model_checkpoints"
MODELS = {  # shortcut -> (repo id, quantization); any other repo id works too
    "8b": ("Qwen/Qwen3-8B", "none"),
    "32b": ("Qwen/Qwen3-32B", "none"),  # ~66 GB bf16: fits on a GH200 (96 GB), not on a 48 GB GPU
    "32b-nf4": ("Qwen/Qwen3-32B", "nf4"),
    "32b-awq": ("Qwen/Qwen3-32B-AWQ", "awq"),
}
HELPER_FILE = "helper_trimmed_sync_qwen3vl_30s-captions.json"
LEADER_FILE = "leader_trimmed_sync_qwen3vl_30s-captions.json"
KEYS = ("helper_event_only", "leader_event_only", "same_event_from_both_view")

SYSTEM_PROMPT = "You are an expert in multi-view egocentric video understanding. You answer only with valid JSON."

COMPARE_PROMPT = """Two people, a HELPER and a LEADER, work together in the same place. Each wears a first-person camera. Both cameras recorded the same {duration:.0f}-second segment ({start_t} to {end_t}), and each recording was described by a captioning model. Below are both descriptions, each with the captioning model's reasoning notes.

In the HELPER view, "the camera wearer" is the HELPER and any other person is most likely the LEADER.
In the LEADER view, "the camera wearer" is the LEADER and any other person is most likely the HELPER.

=== HELPER VIEW ===
{helper}

=== LEADER VIEW ===
{leader}

Compare the two views and list the actions and events (who does what to which object) in three groups:
* "helper_event_only": visible only in the HELPER view, not described in the LEADER view.
* "leader_event_only": visible only in the LEADER view, not described in the HELPER view.
* "same_event_from_both_view": the same action or event described in both views, even if worded differently.
Match events by meaning, not wording: the same person doing the same thing with the same object is one event. Each event goes in exactly one group. Write each event as one short sentence that names the person as "the helper" or "the leader" (never "the camera wearer"); if a person cannot be identified, say "a person". Only list events that the descriptions support; do not invent events. Ignore static scene details (furniture, wall decorations) unless someone interacts with them. A group may be an empty list.

Answer with only a JSON object of this form:
{{"helper_event_only": ["..."], "leader_event_only": ["..."], "same_event_from_both_view": ["..."]}}"""


def format_view(entry: dict, use_thinking: bool) -> str:
    text = f"Caption: {entry['caption']}"
    if use_thinking and entry.get("thinking"):
        text += f"\nReasoning notes: {entry['thinking']}"
    return text


def build_prompt(helper: dict, leader: dict, use_thinking: bool) -> str:
    return COMPARE_PROMPT.format(
        duration=helper["end_sec"] - helper["start_sec"],
        start_t=helper["start_t"],
        end_t=helper["end_t"],
        helper=format_view(helper, use_thinking),
        leader=format_view(leader, use_thinking),
    )


def parse_events(text: str) -> dict | None:
    """Pull the three event lists out of the model's answer; None if no valid JSON object is found."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return {k: [str(e).strip() for e in obj.get(k, []) if str(e).strip()] if isinstance(obj.get(k), list) else []
            for k in KEYS}


@torch.inference_mode()
def generate_batch(model, tokenizer, args, prompts: list[str]) -> list[tuple[str, str]]:
    """Returns (thinking, answer) per prompt; thinking is empty unless --think."""
    texts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True, enable_thinking=args.think) for p in prompts]
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.greedy:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    elif not args.think:  # Qwen3's recommended non-thinking sampling (generation_config holds the thinking one)
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    out = model.generate(**inputs, **gen_kwargs)

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    results = []
    for row in out[:, inputs["input_ids"].shape[1]:].tolist():
        row = [t for t in row if t != tokenizer.pad_token_id]
        idx = len(row) - row[::-1].index(think_end_id) if think_end_id in row else 0
        thinking = tokenizer.decode(row[:idx], skip_special_tokens=True).replace("<think>", "").replace("</think>", "")
        results.append((thinking.strip(), tokenizer.decode(row[idx:], skip_special_tokens=True).strip()))
    return results


def load_model(model_path, quant):
    kwargs = dict(dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"],  # keep the output head in bf16
        )
    elif quant == "awq":
        # autoawq 0.2.9 imports a name that transformers >= 4.57 renamed.
        import transformers.activations as act
        if not hasattr(act, "PytorchGELUTanh"):
            act.PytorchGELUTanh = act.GELUTanh
        kwargs["dtype"] = torch.float16  # AWQ GEMM kernels run in fp16
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        config.quantization_config["modules_to_not_convert"] = ["lm_head"]
        # The auto-picked Marlin kernel JIT-compiles a C++ extension (needs ninja + nvcc on the node);
        # the Triton GEMM kernel needs no build step.
        config.quantization_config["backend"] = "gemm_triton"
        kwargs["config"] = config
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()


def load_windows(path: Path) -> dict[int, dict]:
    return {c["window"]: c for c in json.loads(path.read_text()) if not c["truncated"] and c["caption"]}


def compare_recording(model, tokenizer, args, mp4s_dir: Path) -> None:
    out_file = mp4s_dir / f"helper_vs_leader_{args.model_tag}_30s-events.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else []
    done = {r["window"] for r in results if r["parse_ok"]}

    helper, leader = load_windows(mp4s_dir / HELPER_FILE), load_windows(mp4s_dir / LEADER_FILE)
    windows = sorted(set(helper) & set(leader))
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    todo = [w for w in windows if w not in done]
    missing = sorted(set(helper) ^ set(leader))
    print(f"{mp4s_dir} | {len(windows)} windows in both views | {len(todo)} to compare -> {out_file}"
          + (f" | windows missing from one view: {missing}" if missing else ""))

    for b in range(0, len(todo), args.batch_size):
        batch = todo[b:b + args.batch_size]
        t0 = time.time()
        answers = generate_batch(model, tokenizer, args,
                                 [build_prompt(helper[w], leader[w], args.use_thinking) for w in batch])
        for w, (thinking, answer) in zip(batch, answers):
            events = parse_events(answer)
            h = helper[w]
            results = [r for r in results if r["window"] != w]
            results.append({
                "window": w,
                "start_sec": h["start_sec"],
                "end_sec": h["end_sec"],
                "start_t": h["start_t"],
                "end_t": h["end_t"],
                **(events or {k: [] for k in KEYS}),
                "parse_ok": events is not None,
                "raw_output": None if events is not None and not args.save_raw else answer,
                "model_thinking": thinking or None,
            })
            counts = "/".join(str(len(events[k])) for k in KEYS) if events else "PARSE FAILED"
            print(f"  window {w} [{h['start_t']}-{h['end_t']}] helper/leader/both: {counts}")
        print(f"  batch of {len(batch)} | {time.time() - t0:.1f}s")
        results.sort(key=lambda r: r["window"])
        out_file.write_text(json.dumps(results, indent=4))


def find_recordings(args) -> list[Path]:
    dirs = list(args.recordings)
    if args.caption_root is not None:
        dirs += [p.parent for p in args.caption_root.glob(f"recordings/*/mp4s/{HELPER_FILE}")]
    return sorted({d for d in dirs if (d / HELPER_FILE).exists() and (d / LEADER_FILE).exists()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare helper vs leader captions per window with Qwen3.")
    parser.add_argument("recordings", nargs="*", type=Path,
                        help="mp4s/ folders that hold the helper and leader caption JSONs.")
    parser.add_argument("--caption-root", type=Path, default=None,
                        help="Also process every <caption-root>/recordings/*/mp4s folder with both views.")
    parser.add_argument("--model", default="8b", help="shortcut (%s) or Hugging Face repo id" % ", ".join(MODELS))
    parser.add_argument("--quant", choices=["none", "nf4", "awq"], default=None,
                        help="4-bit quantization; defaults to the shortcut's setting, else none")
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME, help="Hugging Face cache root")
    parser.add_argument("--no-thinking", dest="use_thinking", action="store_false",
                        help="Compare captions only; leave out the captioner's reasoning traces.")
    parser.add_argument("--think", action="store_true",
                        help="Turn on Qwen3's thinking mode (reasons before answering; saved as model_thinking).")
    parser.add_argument("--batch-size", type=int, default=8, help="Windows per generate() call (default: 8).")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Generation budget (default: 1024, or 8192 with --think).")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy decoding instead of the model's default sampling config.")
    parser.add_argument("--save-raw", action="store_true",
                        help="Keep the model's raw answer for every window (always kept when parsing fails).")
    parser.add_argument("--max-windows", type=int, default=None, help="Only compare the first N windows (testing).")
    args = parser.parse_args()
    if not args.recordings and args.caption_root is None:
        parser.error("give mp4s/ folders or --caption-root")
    if args.max_new_tokens is None:
        args.max_new_tokens = 8192 if args.think else 1024
    args.repo_id, quant = MODELS.get(args.model, (args.model, "none"))
    args.quant = args.quant or quant
    # Output tag: the shortcut (or repo name), plus the quantization when it overrides the shortcut's default.
    args.model_tag = (f"qwen3-{args.model}" if args.model in MODELS else args.model.split("/")[-1]) \
        + (f"-{args.quant}" if args.quant != quant else "") + ("-think" if args.think else "")
    return args


def main():
    args = parse_args()
    recordings = find_recordings(args)
    print(f"{len(recordings)} recordings with both helper and leader captions")
    if not recordings:
        return

    model_path = snapshot_download(args.repo_id, cache_dir=os.path.join(args.hf_home, "hub"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = load_model(model_path, args.quant)
    print(f"model: {args.repo_id} | quant: {args.quant}")

    for mp4s_dir in recordings:
        compare_recording(model, tokenizer, args, mp4s_dir)


if __name__ == "__main__":
    main()
