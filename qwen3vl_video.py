r"""Run Qwen3-VL (8B, or 32B in 4-bit) on a local video and save the answer to a .txt file.

Example:
    python qwen3vl_video.py \
        --video lemma/fpv1.mp4 \
        --system "You are an expert in egocentric video understanding." \
        --prompt "Describe interactions between the camera wearer and other people." \
        --output outputs/fpv1.txt --gpu 1

Prompts can also be read from files with --system-file / --prompt-file.

Models (--model takes a shortcut below or any Hugging Face repo id):
    8b         Qwen/Qwen3-VL-8B-Instruct, bf16                 ~17 GB
    32b-nf4    Qwen/Qwen3-VL-32B-Instruct, bitsandbytes NF4    ~20 GB  (downloads ~66 GB bf16 weights)
    32b-awq    QuantTrio/Qwen3-VL-32B-Instruct-AWQ, AWQ 4-bit  ~21 GB  (downloads ~21 GB)
--quant {none,nf4,awq} is set by the shortcut; pass it explicitly for other repo ids.
"""
import argparse
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="path to the input video")
    p.add_argument("--output", required=True, help="path of the .txt file to write the answer to")

    user = p.add_mutually_exclusive_group()
    user.add_argument("--prompt", default="Describe what happens in this video, step by step.",
                      help="user prompt")
    user.add_argument("--prompt-file", help="read the user prompt from this text file")
    system = p.add_mutually_exclusive_group()
    system.add_argument("--system", default=None, help="system prompt (omitted if not given)")
    system.add_argument("--system-file", help="read the system prompt from this text file")

    p.add_argument("--gpu", type=int, default=0, help="physical GPU index (nvidia-smi)")
    p.add_argument("--model", default="8b", help="shortcut (%s) or Hugging Face repo id" % ", ".join(MODELS))
    p.add_argument("--quant", choices=["none", "nf4", "awq"], default=None,
                   help="4-bit quantization; defaults to the shortcut's setting, else none")
    p.add_argument("--hf-home", default="/anvil/scratch/x-hnguyen24/foundation_model_checkpoints",
                   help="Hugging Face cache root")

    # Frame sampling. Lower these if you hit OOM; raise for finer temporal detail.
    p.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video")
    p.add_argument("--max-frames", type=int, default=640, help="hard cap on frames fed to the model")
    p.add_argument("--max-pixels", type=int, default=360 * 420, help="per-frame pixel budget after resizing")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    return p.parse_args()


MODELS = {  # shortcut -> (repo id, quantization)
    "8b": ("Qwen/Qwen3-VL-8B-Instruct", "none"),
    "32b": ("Qwen/Qwen3-VL-32B-Instruct", "none"),  # ~66 GB bf16: does not fit on one 48 GB GPU
    "32b-nf4": ("Qwen/Qwen3-VL-32B-Instruct", "nf4"),
    "32b-awq": ("QuantTrio/Qwen3-VL-32B-Instruct-AWQ", "awq"),
}


def load_model(model_path, quant):
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    kwargs = dict(dtype=torch.bfloat16, device_map={"": 0})  # whole model on the single visible GPU
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual", "lm_head"],  # keep the vision encoder and output head in bf16
        )
    elif quant == "awq":
        # autoawq 0.2.9 imports a name that transformers >= 4.57 renamed.
        import transformers.activations as act
        if not hasattr(act, "PytorchGELUTanh"):
            act.PytorchGELUTanh = act.GELUTanh
        kwargs["dtype"] = torch.float16  # AWQ GEMM kernels run in fp16
    return Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs).eval()


def read_text(path):
    with open(path) as f:
        return f.read().strip()


def main():
    args = parse_args()

    # Must be set before torch / transformers / huggingface_hub are imported.
    os.environ["HF_HOME"] = args.hf_home
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from huggingface_hub import snapshot_download
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")
    prompt = read_text(args.prompt_file) if args.prompt_file else args.prompt
    system = read_text(args.system_file) if args.system_file else args.system

    repo_id, quant = MODELS.get(args.model, (args.model, "none"))
    quant = args.quant or quant
    print(f"model: {repo_id} | quant: {quant} | GPU {args.gpu}", file=sys.stderr)

    # Load from a local snapshot path so nothing falls back to an inherited cache dir.
    model_path = snapshot_download(repo_id, cache_dir=os.path.join(args.hf_home, "hub"))
    model = load_model(model_path, quant)
    processor = AutoProcessor.from_pretrained(model_path)

    messages = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": args.video, "fps": args.fps,
             "max_frames": args.max_frames, "max_pixels": args.max_pixels},
            {"type": "text", "text": prompt},
        ],
    })

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Qwen3-VL uses 16px patches; video metadata gives the model real frame timestamps.
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    video_metadata = None
    if videos:
        videos, video_metadata = map(list, zip(*videos))
    inputs = processor(text=[text], images=images, videos=videos, video_metadata=video_metadata,
                       do_resize=False,  # frames were already resized by qwen_vl_utils
                       return_tensors="pt", padding=True, **video_kwargs).to(model.device)
    print(f"frames sampled: {videos[0].shape[0] if videos else 0} | "
          f"input tokens: {inputs['input_ids'].shape[-1]}", file=sys.stderr)

    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    # strip the prompt tokens off the front of the generation
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(answer.strip() + "\n")

    print(answer)
    print(f"\nsaved to {args.output} | generation {time.time() - t0:.1f}s | "
          f"peak GPU memory {torch.cuda.max_memory_allocated(0) / 1e9:.1f} GB", file=sys.stderr)


if __name__ == "__main__":
    main()
