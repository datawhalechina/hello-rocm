# Gemma 4 E4B-it LoRA on ModelScope `AI-ModelScope/emotion` (single GPU)

Runnable notebook: [`src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb`](https://github.com/datawhalechina/hello-rocm/blob/main/src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb)

This workflow fine-tunes **`google/gemma-4-E4B-it`** with LoRA on the **`AI-ModelScope/emotion`** dataset (a ModelScope mirror of [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)). Everything is loaded via **ModelScope**; **no Hugging Face Hub login** is required.

Background reading: [How to Fine-Tune Gemma 4: A Complete Practical Guide Using a Human Emotion Dataset](https://www.datacamp.com/tutorial/fine-tune-gemma-4)

Compared with a typical Hugging Face–centric notebook, this version:

1. Does not use `HF_TOKEN` or `huggingface_hub.login`.
2. Downloads the base model from ModelScope ([`google/gemma-4-E4B-it`](https://www.modelscope.cn/models/google/gemma-4-E4B-it)) and loads it from a local directory.
3. Downloads the dataset repo with `dataset_snapshot_download`, then loads splits from local Parquet via `datasets` to avoid `MsDataset` / `datasets` version mismatches.
4. Runs on **a single GPU** (no multi-GPU `accelerate launch`).
5. Keeps LoRA SFT, pre/post evaluation, and CSV exports; training logs use `report_to="none"` (no external experiment trackers).
6. Defaults to **BF16 LoRA** without `bitsandbytes` 4-bit, for more predictable behavior on AMD ROCm.

> On ROCm, PyTorch still exposes devices through `torch.cuda`; that is the unified API and does not mean you are on NVIDIA CUDA.

## 1. Install dependencies

```python
%%capture
!pip install -U modelscope transformers accelerate datasets trl peft scikit-learn pandas tqdm
```

## 2. Imports and global settings

Default is **single-GPU BF16** LoRA. If BF16 is unsupported, switch to FP16:

```python
MODEL_DTYPE = torch.float16
BF16 = False
FP16 = True
```

If VRAM is tight, lower `TRAIN_LIMIT`, `EVAL_LIMIT`, `per_device_train_batch_size`, or `max_length`.

```python
import os
import re
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from datasets import DatasetDict, ClassLabel, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from modelscope import snapshot_download
from modelscope.hub.snapshot_download import dataset_snapshot_download

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore")

MODELSCOPE_MODEL_ID = "google/gemma-4-E4B-it"
MODELSCOPE_DATASET_ID = "AI-ModelScope/emotion"
OUTPUT_DIR = "./gemma4-it-emotion-lora-ms-single-gpu"

TRAIN_LIMIT = 4000
VALIDATION_LIMIT = 400
TEST_LIMIT = 400
EVAL_LIMIT = 400

SEED = 42
MODEL_DTYPE = torch.bfloat16
BF16 = True
FP16 = False

SYSTEM_PROMPT = """You are an emotion classification assistant.
Read the user's text and answer with exactly one label.
Only choose from: sadness, joy, love, anger, fear, surprise.
Return only the label and nothing else."""

LABEL_PATTERN = re.compile(r"\b(sadness|joy|love|anger|fear|surprise)\b", re.IGNORECASE)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./datasets", exist_ok=True)
```

## 3. Reproducibility

```python
def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

setup_seed(SEED)
```

## 4. Download the Gemma model from ModelScope

```python
model_dir = snapshot_download(MODELSCOPE_MODEL_ID, cache_dir="./models")
LOCAL_MODEL_DIR = model_dir
```

## 5. Download and load `AI-ModelScope/emotion`

Labels match the original dataset: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`. After loading Parquet, cast `label` to `ClassLabel` so downstream code stays compatible with the HF-style API.

```python
import glob

EMOTION_LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

dataset_dir = dataset_snapshot_download(
    MODELSCOPE_DATASET_ID,
    cache_dir="./datasets",
)

def _parquet_files_for(split_name: str):
    pattern = os.path.join(dataset_dir, "data", f"{split_name}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched pattern: {pattern}")
    return files

raw_dataset = load_dataset(
    "parquet",
    data_files={
        "train": _parquet_files_for("train"),
        "validation": _parquet_files_for("validation"),
        "test": _parquet_files_for("test"),
    },
)

for split_name in list(raw_dataset.keys()):
    if not isinstance(raw_dataset[split_name].features.get("label"), ClassLabel):
        raw_dataset[split_name] = raw_dataset[split_name].cast_column(
            "label", ClassLabel(names=EMOTION_LABEL_NAMES)
        )

def maybe_limit(split, limit):
    split = split.shuffle(seed=SEED)
    if limit is None:
        return split
    return split.select(range(min(limit, len(split))))

dataset = DatasetDict({
    "train": maybe_limit(raw_dataset["train"], TRAIN_LIMIT),
    "validation": maybe_limit(raw_dataset["validation"], VALIDATION_LIMIT),
    "test": maybe_limit(raw_dataset["test"], TEST_LIMIT),
})

label_names = dataset["train"].features["label"].names
VALID_LABELS = set(label_names)
ALL_EVAL_LABELS = label_names + ["INVALID"]
```

## 6. Build TRL prompt–completion examples

```python
def to_prompt_completion(example):
    text = example["text"]
    label = label_names[example["label"]]
    user_content = f"Classify the emotion of this text:\n\n{text}"
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": label},
        ],
    }

sft_dataset = dataset.map(
    to_prompt_completion,
    remove_columns=dataset["train"].column_names,
)
```

## 7. Tokenizer and base model

Load from **`LOCAL_MODEL_DIR`**. The instruction-tuned Gemma build normally includes `chat_template`; if it is missing, the notebook can pull `chat_template.jinja` from the same ModelScope repo.

```python
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,
    use_fast=True,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    torch_dtype=MODEL_DTYPE,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
base_model.generation_config.bos_token_id = tokenizer.bos_token_id
base_model.generation_config.eos_token_id = tokenizer.eos_token_id
```

## 8. Inference helpers

`extract_label` maps raw generations to a label or `INVALID` when parsing fails.

```python
def extract_label(raw_text: str) -> str:
    raw_text = raw_text.strip().lower()
    match = LABEL_PATTERN.search(raw_text)
    if match:
        return match.group(1)
    tokens = raw_text.split()
    if not tokens:
        return "INVALID"
    return tokens[0].strip(".,!?:;\"'()[]{}")

def generate_label(model, tokenizer, user_text: str, system_prompt: str = SYSTEM_PROMPT, max_new_tokens: int = 4) -> str:
    user_content = f"Classify the emotion of this text:\n\n{user_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    device = next(model.parameters()).device
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw_pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return extract_label(raw_pred)
```

## 9. Evaluation

```python
def evaluate_model(model, tokenizer, split="test", limit=EVAL_LIMIT):
    y_true, y_pred, rows = [], [], []
    raw_source = dataset[split]
    if limit is not None:
        raw_source = raw_source.select(range(min(limit, len(raw_source))))
    model.eval()
    for ex in tqdm(raw_source, desc=f"Evaluating {split}", leave=False):
        true_label = label_names[ex["label"]]
        raw_pred_label = generate_label(model, tokenizer, ex["text"], SYSTEM_PROMPT)
        pred_label = raw_pred_label if raw_pred_label in VALID_LABELS else "INVALID"
        y_true.append(true_label)
        y_pred.append(pred_label)
        rows.append({
            "text": ex["text"],
            "true_label": true_label,
            "pred_label": pred_label,
            "raw_pred_label": raw_pred_label,
            "correct": true_label == pred_label,
        })
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=label_names, average="macro", zero_division=0),
        "invalid_predictions": sum(1 for p in y_pred if p == "INVALID"),
        "evaluated_examples": len(y_true),
    }
    report = classification_report(
        y_true, y_pred, labels=label_names, output_dict=True, zero_division=0,
    )
    return metrics, report, pd.DataFrame(rows)
```

Run **before** training to establish a baseline.

## 10. LoRA configuration

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
```

## 11. Training arguments (single GPU)

Effective batch size is `per_device_train_batch_size * gradient_accumulation_steps` (default **16**). Uses `adamw_torch` instead of 8-bit optimizers for broader ROCm compatibility.

```python
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=50,
    num_train_epochs=1,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    bf16=BF16,
    fp16=FP16,
    tf32=False,
    max_length=256,
    packing=False,
    completion_only_loss=True,
    remove_unused_columns=False,
    dataloader_num_workers=2,
    optim="adamw_torch",
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)
```

## 12. Train with `SFTTrainer`

Confirm trainable LoRA parameter count is non-zero before a long run.

```python
if isinstance(base_model, PeftModel):
    base_model = base_model.unload()
    base_model.config.use_cache = False

trainer = SFTTrainer(
    model=base_model,
    train_dataset=sft_dataset["train"],
    eval_dataset=sft_dataset["validation"],
    peft_config=lora_config,
    args=training_args,
    processing_class=tokenizer,
)

train_result = trainer.train()
trainer.model.eval()
trainer.model.config.use_cache = True
```

## 13. Save adapter and tokenizer

```python
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

## 14. Post-training evaluation and exports

The notebook evaluates the in-memory fine-tuned model, compares metrics, and writes CSVs under `OUTPUT_DIR` (metrics, predictions, confusion matrices, etc.).

## 15. Optional: reload adapter for inference

After freeing VRAM (e.g. restart kernel), load the base model from `LOCAL_MODEL_DIR` and apply `PeftModel.from_pretrained(..., OUTPUT_DIR)`.

## 16. Troubleshooting (short)

- **ModelScope model ID or license**: ensure you accepted Gemma terms on ModelScope if download fails.
- **Parquet path errors**: verify the dataset repo layout under `dataset_dir` matches `data/{split}-*.parquet`.
- **OOM**: reduce limits, batch size, or `max_length`; consider FP16 instead of BF16.
- **`MsDataset` / `verification_mode` errors**: this flow avoids `MsDataset`; if you switch APIs, align `modelscope` and `datasets` versions.
