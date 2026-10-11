# LEMMA QA prompts

Prompts behind the results in [LEMMA.md](LEMMA.md). Single-view runs use `qwen3vl_qa.py` and two-view runs use `qwen3vl_qa_multiview.py`. Both are launched by `sbatch_script/run_qa.sbatch` with questions from `LEMMA/subsets/2x1_2x2_qa.json`.

- **No system prompt**: `--system` is never passed, so the conversation has only one user turn.
- **Same chat template for all models**: `Qwen/Qwen3-VL-8B-Instruct` (8b) and `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` (32b-awq) produce the same text from `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.
- **Decoding**: greedy (`do_sample=False`), `max_new_tokens=16`. The first standalone `A`-`D` in the response is taken as the prediction.
- **Video sampling**: 1 fps, frames resized to at most 360x420 pixels. Up to 640 frames for one view, and up to 320 frames per view when there are two.

## Question text

`format_question` (in `qwen3vl_qa.py`) gives the same block for every condition. Options are lettered in the order they appear in the question file:

```
{question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer with the option's letter from the given choices directly.
```

## Single view (fpv1 / fpv2 / master)

```python
messages = [{"role": "user", "content": [
    {"type": "video", "video": "<take>/<view>.mp4", "fps": 1.0, "max_frames": 640, "max_pixels": 151200},
    {"type": "text", "text": format_question(q)},
]}]
```

Rendered by the chat template (shown for `26k_21_6_1_Q1_001`; `<|video_pad|>` expands to the video's visual tokens):

```
<|im_start|>user
<|vision_start|><|video_pad|><|vision_end|>Where was the meat located in the scene?
A. In the refrigerator, behind the lettuce and other groceries
B. On the black stovetop next to the metal kettle
C. On the dining table with the striped tablecloth
D. Inside a white upper kitchen cabinet
Answer with the option's letter from the given choices directly.<|im_end|>
<|im_start|>assistant

```

## Two views (fpv1+fpv2 / fpv1+master / fpv2+master)

Each video comes after a text label, in the order given by `--views`. fpv1 and fpv2 are labelled `first-person` and master is labelled `third-person`. After both videos comes a note that they are synchronized, then the question:

```python
messages = [{"role": "user", "content": [
    {"type": "text", "text": "View 1 ({kind1}, {view1}):"},
    {"type": "video", "video": "<take>/<view1>.mp4", "fps": 1.0, "max_frames": 320, "max_pixels": 151200},
    {"type": "text", "text": "View 2 ({kind2}, {view2}):"},
    {"type": "video", "video": "<take>/<view2>.mp4", "fps": 1.0, "max_frames": 320, "max_pixels": 151200},
    {"type": "text", "text": "The two videos are synchronized recordings of the same event from different cameras.\n"
                             + format_question(q)},
]}]
```

Rendered by the chat template (fpv1+master, same question):

```
<|im_start|>user
View 1 (first-person, fpv1):<|vision_start|><|video_pad|><|vision_end|>View 2 (third-person, master):<|vision_start|><|video_pad|><|vision_end|>The two videos are synchronized recordings of the same event from different cameras.
Where was the meat located in the scene?
A. In the refrigerator, behind the lettuce and other groceries
B. On the black stovetop next to the metal kettle
C. On the dining table with the striped tablecloth
D. Inside a white upper kitchen cabinet
Answer with the option's letter from the given choices directly.<|im_end|>
<|im_start|>assistant

```
