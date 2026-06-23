# Qwen3.5-4B LoRA and SwanLab Records

> Companion Notebook: [Qwen3.5-4B-LoRA.ipynb](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/models/qwen3.5/Qwen3.5-4B-LoRA.ipynb)

This tutorial fine-tunes `Qwen3.5-4B` with `transformers + peft` LoRA and records training with **SwanLab**. The Notebook has been adjusted for the hello-rocm repository layout: datasets and code paths point to files inside this project.

## Qwen3.5-4B Overview

`Qwen3.5-4B` uses a hybrid architecture that combines Gated Delta Network linear attention with Full Attention layers. It also supports thinking mode and long context. Because the architecture is relatively new, use `transformers>=4.57`.

For text-only fine-tuning, `AutoModelForCausalLM` loads the text language model (`Qwen3_5ForCausalLM`) without the vision tower.

## Environment Setup

The Notebook provides two alternative model download paths. Choose one and keep `model_id` consistent with the actual local model directory.

### Hugging Face

```bash
pip install "transformers>=4.57" accelerate datasets peft swanlab huggingface_hub

# Optional: acceleration for linear attention; the model can still run with PyTorch fallback
pip install flash-linear-attention
```

### ModelScope

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install "transformers>=4.57" accelerate datasets peft swanlab modelscope

# Optional: acceleration for linear attention; the model can still run with PyTorch fallback
pip install flash-linear-attention
```

> In the tested Notebook run, the Hugging Face download cell was interrupted due to network waiting, while the ModelScope download completed. Use Hugging Face when the network is stable; otherwise use ModelScope. The training flow is otherwise the same.

## Model Download

### Hugging Face

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="Qwen/Qwen3.5-4B",
    local_dir="./model/Qwen3.5-4B",
)
print(f"Model downloaded to: {model_dir}")
```

Then use:

```python
model_id = "./model/Qwen3.5-4B"
```

### ModelScope

```python
from modelscope import snapshot_download

model_dir = snapshot_download("Qwen/Qwen3.5-4B", cache_dir="./model")
print(f"Model downloaded to: {model_dir}")
```

The tested Notebook uses:

```python
model_id = "./model/Qwen/Qwen3.5-4B"
```

Do not commit runtime artifacts such as `model/`, `output/`, or `swanlog/`.

## Dataset

The tutorial uses Alpaca-style supervised fine-tuning data:

```json
{
  "instruction": "Answer the following question. Output only the answer.",
  "input": "1+1 equals what?",
  "output": "2"
}
```

Available datasets in this repository:

- Tested subset: [`src/fine-tune/datasets/huanhuan-100.json`](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/datasets/huanhuan-100.json), 100 samples for quick validation.
- Full dataset: [`src/fine-tune/datasets/huanhuan.json`](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/datasets/huanhuan.json), for longer experiments.

The Notebook reads:

```python
dataset_path = "../../datasets/huanhuan-100.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ds = Dataset.from_list(data)
```

From `src/fine-tune/models/qwen3.5/`, this resolves to `src/fine-tune/datasets/huanhuan-100.json`.

## Chat Template

Qwen3.5 enables thinking mode by default. For role-play fine-tuning, the Notebook disables it:

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)
```

Even with thinking disabled, the template may keep an empty `<think></think>` placeholder. This is normal for the Qwen3.5 chat template.

## Data Processing

The processing function tokenizes the prompt prefix (`system + user`) and the full conversation separately. Only the assistant response contributes to loss; prompt tokens are masked with `-100`.

```python
def process_func(example):
    MAX_LENGTH = 1024
    SYS = "现在你要扮演皇帝身边的女人--甄嬛"

    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": example["instruction"] + example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        messages[:2], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        enable_thinking=False, return_dict=False,
    )
    response_ids = full_ids[len(prompt_ids):]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

## Model Loading

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.enable_input_require_grads()
```

If linear-attention acceleration libraries are unavailable, Transformers falls back to a PyTorch implementation. This is slower but still runnable.

## LoRA Configuration

The Notebook applies LoRA to full-attention projection layers and MLP layers:

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

The tested run reports about 10.6M trainable parameters, or roughly 0.25% of the full model.

## Training with SwanLab

```python
args = TrainingArguments(
    output_dir="./output/Qwen3_5_4B_LoRA",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen3.5-Lora",
    experiment_name="Qwen3.5-4B-LoRA",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()
```

With `huanhuan-100.json`, this is a quick workflow validation. Using the full dataset increases training time.

## Inference with LoRA Weights

```python
model_id = "./model/Qwen/Qwen3.5-4B"
lora_path = "./output/Qwen3_5_4B_LoRA/checkpoint-21"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, model_id=lora_path)
model.eval()
```

If the Notebook is executed inside a container, make sure the `model/`, `output/`, and `swanlog/` directories are mounted or copied out before the container is removed.
