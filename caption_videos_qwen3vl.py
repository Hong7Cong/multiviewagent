"""
Caption long videos in fixed-length windows (default 30 s at 1 FPS) with Qwen3-VL-30B-A3B-Thinking.

Each window's 1 FPS frames are passed to the model as one video clip whose frame timestamps are the
absolute seconds in the source video. Output is one JSON per video, written after every batch so
interrupted runs resume where they left off.

Example:
    python caption_videos_qwen3vl.py /path/to/video.mp4 --out-dir captions/qwen3vl --view ego
    python caption_videos_qwen3vl.py /path/to/videos/*.mp4 --out-dir captions/qwen3vl --batch-size 4
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Thinking"

VIEW_DESCRIPTIONS = {
    "ego": (
        "The frames come from a first-person (egocentric) camera worn by a person. Refer to that person as "
        "'the camera wearer'. Describe what the camera wearer is doing with their hands, what they look at, "
        "and who they interact with."
    ),
    "exo": (
        "The frames come from a fixed third-person (exocentric) camera observing a scene. Describe the people "
        "in view, where they are in the scene, and how they move and interact."
    ),
    "generic": "The frames come from a video.",
}

CAPTION_PROMPT = """You are given {num_frames} frames sampled at {fps:g} frame(s) per second from a {duration:.0f}-second segment of a longer video (from {start_t} to {end_t}). {view_description}
{extra_context}
Write a detailed, factual caption of this segment as a single paragraph. Cover:
* Location and setting (room or place, notable furniture and layout).
* People: how many, distinguishing appearance (clothing, hair), and what each one does.
* Actions and events, in temporal order.
* Objects that are handled, used, or prominent, with their attributes (color, size, position).
* Any visible text (signs, screens, labels).
Use a neutral, descriptive tone. Only describe what is visible; do not speculate about intentions or invent names. Do not mention frame numbers or timestamps."""


def seconds_to_hhmmss(sec: float) -> str:
    """Format seconds as HH:MM:SS."""
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def get_windows(duration: float, window_sec: float, min_window_sec: float) -> list[tuple[float, float]]:
    """Split [0, duration) into consecutive windows; drop a trailing window shorter than min_window_sec."""
    windows = []
    num_windows = math.ceil(duration / window_sec)
    for w in range(num_windows):
        start, end = w * window_sec, min((w + 1) * window_sec, duration)
        if end - start >= min_window_sec or w == 0:
            windows.append((start, end))
    return windows


def sample_window_frames(decoder: VideoDecoder, start: float, end: float, fps: float, duration: float):
    """Decode frames at `fps` within [start, end). Returns uint8 tensor (T, C, H, W) and sample times."""
    step = 1.0 / fps
    times = [start + step / 2 + k * step for k in range(max(1, round((end - start) * fps)))]
    times = [min(t, duration - 1e-3) for t in times]
    frames = decoder.get_frames_played_at(seconds=times).data
    if frames.shape[0] < 2:  # Qwen3-VL merges 2 frames per temporal patch
        frames = frames.repeat(2, 1, 1, 1)
        times = times * 2
    return frames, times


def build_prompt(args, num_frames: int, start: float, end: float) -> str:
    extra = f"Additional context: {args.context}\n" if args.context else ""
    return CAPTION_PROMPT.format(
        num_frames=num_frames,
        fps=args.fps,
        duration=end - start,
        start_t=seconds_to_hhmmss(start),
        end_t=seconds_to_hhmmss(end),
        view_description=VIEW_DESCRIPTIONS[args.view],
        extra_context=extra,
    )


def split_thinking(token_ids: list[int], think_end_id: int, tokenizer) -> tuple[str, str, bool]:
    """Split generated ids into (thinking, caption, truncated) at the last </think> token."""
    if think_end_id in token_ids:
        idx = len(token_ids) - token_ids[::-1].index(think_end_id)
        thinking = tokenizer.decode(token_ids[:idx - 1], skip_special_tokens=True).strip()
        caption = tokenizer.decode(token_ids[idx:], skip_special_tokens=True).strip()
        return thinking, caption, False
    # Ran out of tokens while still thinking
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip(), "", True


@torch.inference_mode()
def caption_batch(model, processor, args, frames_list, times_list, windows):
    """Caption a batch of windows. Returns list of (thinking, caption, truncated, num_new_tokens)."""
    texts, metadata = [], []
    for frames, times, (start, end) in zip(frames_list, times_list, windows):
        messages = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": build_prompt(args, frames.shape[0], start, end)},
        ]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        # fps=1/step with integer indices makes the model's <t seconds> markers absolute video time
        metadata.append(VideoMetadata(
            total_num_frames=frames.shape[0],
            fps=args.fps,
            width=frames.shape[-1],
            height=frames.shape[-2],
            frames_indices=[t * args.fps for t in times],
        ))

    n_frames = frames_list[0].shape[0]
    inputs = processor(
        text=texts,
        videos=frames_list,
        video_metadata=metadata,
        do_sample_frames=False,
        cap_pixels_per_frame=False,
        size={"shortest_edge": 32 * 32 * 4, "longest_edge": args.max_pixels_per_frame * n_frames},
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.greedy:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]

    think_end_id = processor.tokenizer.convert_tokens_to_ids("</think>")
    pad_id = processor.tokenizer.pad_token_id
    results = []
    for row in new_tokens.tolist():
        row = [t for t in row if t != pad_id]
        thinking, caption, truncated = split_thinking(row, think_end_id, processor.tokenizer)
        results.append((thinking, caption, truncated, len(row)))
    return results, inputs["input_ids"].shape[1]


def caption_video(model, processor, args, video_path: Path) -> None:
    out_dir = Path(args.out_dir)
    if args.input_root is not None:  # mirror the input folder structure under out_dir
        out_dir = out_dir / video_path.resolve().parent.relative_to(args.input_root.resolve())
        out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{video_path.stem}_qwen3vl_{int(args.window_sec)}s-captions.json"
    captions = json.loads(out_file.read_text()) if out_file.exists() else []
    done = {c["window"] for c in captions if not c["truncated"]}

    decoder = VideoDecoder(str(video_path), device="cpu")
    duration = decoder.metadata.duration_seconds
    windows = get_windows(duration, args.window_sec, args.min_window_sec)
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    todo = [w for w in range(len(windows)) if w not in done]
    print(f"{video_path} | {duration:.1f}s | {len(windows)} windows | {len(todo)} to caption -> {out_file}")

    for b in range(0, len(todo), args.batch_size):
        batch_ids = todo[b:b + args.batch_size]
        # Group by frame count so a short trailing window gets its own batch
        groups = {}
        for w in batch_ids:
            start, end = windows[w]
            frames, times = sample_window_frames(decoder, start, end, args.fps, duration)
            groups.setdefault(frames.shape[0], []).append((w, frames, times))

        for group in groups.values():
            t0 = time.time()
            results, prompt_len = caption_batch(
                model, processor, args,
                [g[1] for g in group], [g[2] for g in group], [windows[g[0]] for g in group],
            )
            elapsed = time.time() - t0
            for (w, frames, times), (thinking, caption, truncated, n_new) in zip(group, results):
                start, end = windows[w]
                captions = [c for c in captions if c["window"] != w]
                captions.append({
                    "video": str(video_path),
                    "window": w,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "start_t": seconds_to_hhmmss(start),
                    "end_t": seconds_to_hhmmss(end),
                    "num_frames": int(frames.shape[0]),
                    "caption": caption,
                    "thinking": thinking if args.save_thinking else None,
                    "truncated": truncated,
                    "num_generated_tokens": n_new,
                })
                flag = " [TRUNCATED: no </think> within max_new_tokens]" if truncated else ""
                print(f"  window {w} [{seconds_to_hhmmss(start)}-{seconds_to_hhmmss(end)}] "
                      f"{n_new} tokens{flag}")
            print(f"  batch of {len(group)} | prompt {prompt_len} tokens | {elapsed:.1f}s")
            captions.sort(key=lambda c: c["window"])
            out_file.write_text(json.dumps(captions, indent=4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption long videos in fixed windows with Qwen3-VL Thinking.")
    parser.add_argument("videos", nargs="+", type=Path, help="Video files to caption.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for per-video caption JSONs.")
    parser.add_argument("--input-root", type=Path, default=None,
                        help="If set, write each JSON under out-dir at the video's path relative to this root.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--view", choices=list(VIEW_DESCRIPTIONS), default="generic",
                        help="Camera viewpoint, used to tailor the prompt (default: generic).")
    parser.add_argument("--context", default=None, help="Optional extra context added to every prompt.")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Window length in seconds (default: 30).")
    parser.add_argument("--min-window-sec", type=float, default=2.0,
                        help="Drop a trailing window shorter than this (default: 2).")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames sampled per second (default: 1).")
    parser.add_argument("--max-pixels-per-frame", type=int, default=640 * 360,
                        help="Per-frame pixel budget before patching (default: 640*360).")
    parser.add_argument("--batch-size", type=int, default=4, help="Windows per generate() call (default: 4).")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Generation budget including thinking (default: 4096).")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy decoding instead of the model's default sampling config.")
    parser.add_argument("--no-save-thinking", dest="save_thinking", action="store_false",
                        help="Do not store the model's reasoning trace in the output JSON.")
    parser.add_argument("--max-windows", type=int, default=None, help="Only caption the first N windows (testing).")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    ).eval()

    for video_path in args.videos:
        caption_video(model, processor, args, video_path)


if __name__ == "__main__":
    main()
