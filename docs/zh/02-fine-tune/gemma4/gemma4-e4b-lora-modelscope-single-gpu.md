# 在 ModelScope `AI-ModelScope/emotion` 上单卡微调 Gemma 4 E4B-it（LoRA）

可运行笔记本：[「在 AI-ModelScope/emotion 上单卡微调 Gemma 4 E4B-it（魔搭 ModelScope 版本）」](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb)

本流程使用 **魔搭 ModelScope** 上的 **`AI-ModelScope/emotion`**（[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) 的官方镜像）数据集，对 **`google/gemma-4-E4B-it`** 进行 **LoRA 指令微调**。模型与数据均从 ModelScope 拉取，**不需要 `HF_TOKEN` 或 Hugging Face 登录**。

背景阅读：[如何微调 Gemma 4：基于人类情绪数据集的完整实战指南](https://www.datacamp.com/tutorial/fine-tune-gemma-4)

与典型的 Hugging Face Hub 流程相比，本版：

1. **不使用** `HF_TOKEN`，不依赖 `huggingface_hub.login`。
2. **模型**从魔搭下载（[`google/gemma-4-E4B-it`](https://www.modelscope.cn/models/google/gemma-4-E4B-it)），再用本地路径加载。
3. **数据集**通过 `dataset_snapshot_download` 整仓拉取后，用 `datasets` 从本地 Parquet 读取，绕开 `MsDataset` 与 `datasets` 版本不兼容问题。
4. **单卡训练**，不使用多卡 `accelerate launch`。
5. 保留 LoRA、微调前后评估与 CSV 导出；训练侧 `report_to="none"`，**不依赖外部实验跟踪服务**。
6. AMD / ROCm 环境下默认采用 **BF16 LoRA**，**不启用** `bitsandbytes` 4bit，稳定性通常更好。

> 在 ROCm 上，PyTorch 仍通过 `torch.cuda` 访问 GPU，这是统一接口，**不代表**在使用 NVIDIA CUDA。

## 1. 安装依赖

```python
%%capture
!pip install -U modelscope transformers accelerate datasets trl peft scikit-learn pandas tqdm
```

## 2. 导入与全局配置

默认 **单卡 BF16**。若硬件不支持 BF16，可改为 FP16：

```python
MODEL_DTYPE = torch.float16
BF16 = False
FP16 = True
```

显存紧张时，优先降低 `TRAIN_LIMIT`、`EVAL_LIMIT`、`per_device_train_batch_size` 或 `max_length`。

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

## 3. 固定随机种子

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

## 4. 从魔搭下载 Gemma 模型

```python
model_dir = snapshot_download(MODELSCOPE_MODEL_ID, cache_dir="./models")
LOCAL_MODEL_DIR = model_dir
```

## 5. 下载并加载 `AI-ModelScope/emotion`

字段与标签与原始数据集一致：`sadness`、`joy`、`love`、`anger`、`fear`、`surprise`。从 Parquet 读入后需将 `label` **显式** `cast` 为 `ClassLabel`，以便后续与 HF 版接口一致。

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

## 6. 构造 TRL 的 prompt–completion 数据

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

## 7. 加载 tokenizer 与基座模型

从 **`LOCAL_MODEL_DIR`** 加载。instruction-tuned 权重一般自带 `chat_template`；若缺失，笔记本会从同一魔搭仓库补充 `chat_template.jinja`。

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

## 8. 推理辅助函数

`extract_label` 将生成结果解析为标签；无法解析时记为 `INVALID`。

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

## 9. 评估

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

训练前运行评估作为基线。

## 10. LoRA 配置

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

## 11. 训练参数（单卡）

等效 batch = `per_device_train_batch_size * gradient_accumulation_steps`（默认 **16**）。使用 `adamw_torch`，避免 ROCm 下 8bit 优化器兼容问题。

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

## 12. 使用 `SFTTrainer` 训练

开始长时间训练前，务必确认 **可训练 LoRA 参数量** 大于 0。

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

## 13. 保存 LoRA 与 tokenizer

```python
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

## 14. 微调后评估与导出

笔记本在内存中直接评估微调后模型，对比指标，并将多个 CSV 写入 `OUTPUT_DIR`（指标、预测样例、混淆矩阵等）。

## 15. 可选：重新加载 adapter 推理

释放显存后（例如重启 kernel），从 `LOCAL_MODEL_DIR` 加载基座模型，再用 `PeftModel.from_pretrained(..., OUTPUT_DIR)` 挂载 LoRA。

## 16. 常见问题（节选）

- **魔搭模型或许可**：若下载失败，确认已在网页端接受 Gemma 许可。
- **Parquet 路径**：确认数据集目录下为 `data/{split}-*.parquet` 布局。
- **显存不足**：减小数据上限、batch、`max_length`，或改用 FP16。
- **`MsDataset` / `verification_mode`**：本流程不走 `MsDataset`；若改回高层 API，需对齐 `modelscope` 与 `datasets` 版本。
