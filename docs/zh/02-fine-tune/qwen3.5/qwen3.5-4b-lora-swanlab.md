# Qwen3.5-4B LoRA 及 SwanLab 可视化记录

> 本教程配套 Notebook：[Qwen3.5-4B-LoRA.ipynb](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/models/qwen3.5/Qwen3.5-4B-LoRA.ipynb)

本教程基于 `Qwen3.5-4B`，使用 `transformers + peft` 完成 LoRA 微调，并使用 **SwanLab** 记录训练过程。Notebook 已按 hello-rocm 项目目录结构调整，数据集与代码均来自本仓库。

## Qwen3.5-4B 简介

`Qwen3.5-4B` 是通义千问团队推出的新一代基础模型，具备以下特点：

- **混合架构**：将 Gated Delta Network（线性注意力）与传统 Full Attention 交错堆叠，每 4 层中前 3 层为线性注意力，第 4 层为全注意力。
- **统一多模态底座**：模型内置视觉编码器；本教程只使用文本能力，因此使用 `AutoModelForCausalLM` 加载文本语言模型。
- **Thinking 能力**：默认开启思考模式；角色扮演类微调通常通过 `enable_thinking=False` 关闭显式思考输出。
- **长上下文**：支持最长 262144（256K）上下文。

由于 Qwen3.5 架构较新，建议使用 `transformers>=4.57`。

## 环境配置

Notebook 提供 Hugging Face 与 ModelScope 两种模型下载方式，二选一即可。除下载工具外，其余训练依赖基本一致。

### Hugging Face 方式

```bash
pip install "transformers>=4.57" accelerate datasets peft swanlab huggingface_hub

# 可选：为线性注意力安装加速算子；不安装也可运行，会回退到 PyTorch 实现
pip install flash-linear-attention
```

### ModelScope 方式

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install "transformers>=4.57" accelerate datasets peft swanlab modelscope

# 可选：为线性注意力安装加速算子；不安装也可运行，会回退到 PyTorch 实现
pip install flash-linear-attention
```

> 说明：Notebook 实测时 Hugging Face 下载单元曾因网络等待被中断，ModelScope 下载单元完成下载。若网络可稳定访问 Hugging Face，可优先使用 Hugging Face；若访问不稳定，可使用 ModelScope。无论使用哪种方式，后续 `model_id` 都需要与实际下载目录保持一致。

## 模型下载

### Hugging Face 下载

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="Qwen/Qwen3.5-4B",
    local_dir="./model/Qwen3.5-4B",
)

print(f"模型下载完成，保存路径为：{model_dir}")
```

如果使用该方式，后续模型路径可设置为：

```python
model_id = "./model/Qwen3.5-4B"
```

### ModelScope 下载

```python
from modelscope import snapshot_download

model_dir = snapshot_download("Qwen/Qwen3.5-4B", cache_dir="./model")
print(f"模型下载完成，保存路径为：{model_dir}")
```

Notebook 实测使用该方式，后续模型路径使用：

```python
model_id = "./model/Qwen/Qwen3.5-4B"
```

> 注意：模型文件较大，不建议提交 `model/`、`output/`、`swanlog/` 等运行产物到仓库。

## 数据集构建

对大语言模型进行 supervised fine-tuning（SFT，有监督微调）的数据格式如下：

```json
{
  "instruction": "回答以下用户问题，仅输出答案。",
  "input": "1+1等于几?",
  "output": "2"
}
```

本教程使用甄嬛对话示例数据集：

- Notebook 实测数据集：[`src/fine-tune/datasets/huanhuan-100.json`](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/datasets/huanhuan-100.json)，共 100 条样本，适合快速跑通流程。
- 完整数据集：[`src/fine-tune/datasets/huanhuan.json`](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/datasets/huanhuan.json)，适合完整训练实验。

Notebook 中的数据读取路径为：

```python
dataset_path = "../../datasets/huanhuan-100.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ds = Dataset.from_list(data)
ds
```

该路径从 `src/fine-tune/models/qwen3.5/` 出发，指向 `src/fine-tune/datasets/huanhuan-100.json`。

## 认识 Qwen3.5 的 Chat Template

Qwen3.5 默认开启思考模式（`enable_thinking=True`）。对于角色扮演任务，通常关闭思考模式，让模型直接输出目标风格的回答。

```python
messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": "你父亲是谁？"},
    {"role": "assistant", "content": "家父是大理寺少卿甄远道。"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)
print(text)
```

关闭 thinking 后，模板仍可能保留空的 `<think></think>` 占位；这是 Qwen3.5 Chat Template 的正常行为。

## 数据准备

处理函数对每条样本分别 tokenize「前缀（system + user）」和「完整对话」，再通过 token 级别切片得到 assistant response。`labels` 中只有回答部分参与 loss 计算，前缀部分使用 `-100` 屏蔽。

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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

## 加载模型和 tokenizer

Qwen3.5-4B 是多模态模型。只使用文本能力时，`AutoModelForCausalLM` 会加载文本语言模型 `Qwen3_5ForCausalLM`，不会加载视觉塔，显存占用更低。

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.enable_input_require_grads()
model.dtype
```

## LoRA 配置

Qwen3.5-4B 的 32 层中，每 4 层有 3 层是线性注意力（`linear_attn`），1 层是全注意力（`self_attn`）。

- 全注意力层：`q_proj / k_proj / v_proj / o_proj`
- 线性注意力层：`in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj`
- 每层 MLP：`gate_proj / up_proj / down_proj`

Notebook 使用的 `target_modules` 覆盖所有全注意力层与每一层 MLP：

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

实测可训练参数约为：

```text
trainable params: 10,616,832 || all params: 4,216,368,128 || trainable%: 0.2518
```

## 配置训练参数

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
```

`huanhuan-100.json` 共 100 条样本，上述配置会快速完成一次流程演示；如果改用完整数据集，训练时间会显著增加。

## 使用 SwanLab 记录训练

[SwanLab](https://github.com/swanhubx/swanlab) 是开源训练记录工具，可记录 loss、超参数、实验配置与可视化曲线。

```python
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

训练过程中会生成 `./output/Qwen3_5_4B_LoRA/` 与 SwanLab 日志。若在容器内运行，请确认输出目录已挂载到需要保留的位置。

## 加载 LoRA 权重推理

得到 checkpoint 后，加载基础模型并挂载 LoRA 权重进行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_id = "./model/Qwen/Qwen3.5-4B"
lora_path = "./output/Qwen3_5_4B_LoRA/checkpoint-21"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, model_id=lora_path)
model.eval()

messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": "你是谁？"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

gen_kwargs = {"max_new_tokens": 128, "do_sample": True, "top_p": 0.8, "temperature": 0.7}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
outputs = outputs[:, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

输出示例：

```text
我是甄嬛，家父是大理寺少卿甄远道。
```

## 常见问题

### Hugging Face 下载中断怎么办？

网络不稳定时可以切换到 ModelScope 下载。切换下载源后，需要同步修改 `model_id`，确保后续 tokenizer、模型加载与 LoRA 推理使用同一个实际模型目录。

### 为什么 `flash-linear-attention` 没装也能运行？

Qwen3.5 的线性注意力有加速路径；未安装相关算子时，Transformers 会回退到 PyTorch 实现。回退路径可运行，但训练速度会更慢。

### 为什么本地看不到 `model/` 或 `output/`？

Notebook 可能在容器中运行，输出目录位于容器文件系统。需要保留模型、checkpoint 或 SwanLab 日志时，应提前挂载输出目录或在训练结束后复制出来。
