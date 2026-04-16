# 第六章：大模型训练流程实践 AMD ROCm 版本

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

</div>

## 简介

&emsp;&emsp;本章从实战角度，深入讲解如何使用主流框架（Transformers、DeepSpeed、PEFT）在 AMD GPU 上进行大规模高效的大模型训练。相比第五章的从零实现，本章重点在于使用经过验证的工业级框架，实现可扩展的多卡分布式训练。

&emsp;&emsp;通过本章的学习，你将掌握使用 Transformers Trainer API 进行预训练和微调、理解 DeepSpeed 分布式优化、学习参数高效微调技术（LoRA、QLoRA），以及实现模型对齐（RLHF、DPO）等高级技能。

## 学习目标

- ✅ 掌握 Hugging Face Transformers 框架的完整使用
- ✅ 实现多卡分布式训练（DDP、DeepSpeed）
- ✅ 优化显存使用和训练速度
- ✅ 学习参数高效微调（LoRA、QLoRA）
- ✅ 实现模型对齐和偏好学习（RLHF、DPO）
- ✅ 在 AMD GPU 上进行工业级训练

## 章节内容

### 6.1 - 框架介绍与基础

这一部分介绍 Transformers 框架和相关工具的基本概念和使用方法。

#### 6.1.1 Transformers 框架概述

**Hugging Face Transformers** 是当今最流行的 NLP 框架，提供：
- 数百个预训练模型
- 统一的模型加载 API
- 高级 Trainer 类，内置分布式训练支持
- 与 DeepSpeed、PEFT 等框架的无缝集成

**关键优势**：
- 📦 **广泛的模型支持**：BERT、GPT、LLaMA、Qwen、Mistral 等
- 🚀 **开箱即用的性能优化**：混合精度、梯度累积、梯度检查点
- 📊 **完整的训练管道**：数据加载、训练、评估、推理
- 🔗 **生态集成**：Datasets、Evaluate、Accelerate 等
- 💾 **模型管理**：本地保存、Hugging Face Hub 上传

**核心类**：
- `AutoModel / AutoModelForCausalLM`：模型加载
- `AutoConfig`：配置管理
- `AutoTokenizer`：分词器
- `Trainer`：统一训练接口
- `TrainingArguments`：训练超参数

📖 [详细教程](./6.1-框架与基础.md)

---

#### 6.1.2 Trainer API 讲解

**Trainer** 是 Transformers 中最重要的类，它封装了训练的所有细节。

**基本用法**：
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    bf16=True,  # 混合精度，ROCm 推荐使用 BF16
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
```

**关键参数**：

| 参数 | 说明 | ROCm 建议 |
|------|------|---------|
| `per_device_train_batch_size` | 每卡批大小 | 从 1 开始，逐步增加 |
| `gradient_accumulation_steps` | 梯度累积步数 | 8-16 (模拟更大批大小) |
| `num_train_epochs` | 训练轮数 | 根据数据量调整 |
| `learning_rate` | 学习率 | 2e-5 ~ 5e-5 (微调) |
| `bf16` | BF16 混合精度 | True (ROCm 推荐) |
| `save_strategy` | 模型保存策略 | "steps" 或 "epoch" |
| `eval_strategy` | 评估策略 | "steps" 或 "epoch" |
| `ddp_find_unused_parameters` | DDP 参数优化 | False |

**分布式训练支持**：
- 数据并行 (DDP)：通过 `num_train_epochs` 自动启用
- DeepSpeed 集成：通过 `deepspeed_config_file` 指定配置
- Accelerate 支持：自动检测硬件和配置

📖 [详细教程](./6.1-框架与基础.md) | 💻 [示例代码](./code/)

---

#### 6.1.3 配置管理与超参数

**TrainingArguments 主要参数分类**：

1. **基础训练参数**
   ```python
   output_dir: str  # 输出目录
   num_train_epochs: int  # 训练轮数
   max_steps: int  # 最大步数（可覆盖 num_train_epochs）
   ```

2. **批大小和梯度参数**
   ```python
   per_device_train_batch_size: int
   per_device_eval_batch_size: int
   gradient_accumulation_steps: int
   max_grad_norm: float  # 梯度裁剪
   ```

3. **优化器参数**
   ```python
   learning_rate: float
   weight_decay: float
   warmup_steps: int
   adam_beta1: float
   adam_beta2: float
   adam_epsilon: float
   ```

4. **性能优化参数**
   ```python
   bf16: bool  # BF16 混合精度
   fp16: bool  # FP16 混合精度
   gradient_checkpointing: bool  # 梯度检查点
   dataloader_num_workers: int  # 数据加载并行数
   ```

5. **保存和评估**
   ```python
   save_strategy: str  # "steps", "epoch", "no"
   save_steps: int
   save_total_limit: int  # 最多保留的检查点数
   eval_strategy: str
   eval_steps: int
   ```

6. **日志和监控**
   ```python
   logging_strategy: str
   logging_steps: int
   report_to: List[str]  # ["tensorboard", "wandb"]
   ```

---

### 6.2 - 预训练实践

这一部分详细讲解如何使用 Transformers 进行大规模模型预训练。

#### 6.2.1 模型下载与初始化

**文件**: `code/download_model.py`

预训练有两种方式：

1. **从零初始化** (Pretraining from scratch)
   ```python
   from transformers import AutoConfig, AutoModelForCausalLM
   
   # 使用现有模型架构初始化
   config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B")
   model = AutoModelForCausalLM.from_config(config)
   ```

2. **基于预训练模型** (Continued pretraining)
   ```python
   # 加载已有权重，在新数据上继续训练
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
   ```

**模型下载脚本**：
```bash
# 使用 Hugging Face 官方工具
huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B \
    --local-dir ./models/Qwen2.5-1.5B

# 或使用 git clone (需要 git-lfs)
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B ./models/Qwen2.5-1.5B
```

**常用模型列表** (按推荐顺序)：

| 模型 | 参数量 | 推荐硬件 | 用途 |
|------|--------|--------|------|
| Qwen2.5-1.5B | 1.5B | 8GB+ | 快速原型测试 |
| Qwen2.5-7B | 7B | 16GB+ | 通用大模型 |
| Llama-2-7B | 7B | 16GB+ | 开源基准模型 |
| Qwen2.5-14B | 14B | 32GB+ | 高性能微调 |
| Llama-2-13B | 13B | 32GB+ | 对标 Qwen |

💻 [下载脚本](./code/download_model.py)

---

#### 6.2.2 数据集准备

**文件**: `code/download_dataset.py`, `code/process_dataset.ipynb`

**数据加载流程**：
```python
from datasets import load_dataset

# 方式 1: 从 Hugging Face 加载
dataset = load_dataset("wikitext", "wikitext-103-v1")

# 方式 2: 加载本地数据
dataset = load_dataset("text", data_files="path/to/data.txt")

# 方式 3: 使用自定义数据集
dataset = load_dataset("json", data_files="path/to/data.jsonl")
```

**数据处理管道**：
```python
def preprocess_function(examples):
    # 分词
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_len,
        return_overflowing_tokens=True,
    )
    
    # 处理 overflowing tokens
    input_batch = []
    for i in range(0, len(result["input_ids"]), max_seq_len):
        input_batch.append(result["input_ids"][i:i+max_seq_len])
    
    return {"input_ids": input_batch, "labels": input_batch}

# 应用处理函数
dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"],
    num_proc=4,
)
```

**常用数据集**：

| 数据集 | 大小 | 用途 |
|-------|------|------|
| Wikitext-103 | 100MB | 通用预训练 |
| Common Crawl | TB 级 | 大规模预训练 |
| GitHub | TB 级 | 代码训练 |
| LLAMA2-HC3 | GB 级 | 指令微调 |
| Belle | GB 级 | 中文微调 |

💻 [数据处理脚本](./code/download_dataset.py)

---

#### 6.2.3 分布式预训练

**文件**: `code/pretrain.py`, `code/pretrain.sh`

**单卡训练**：
```bash
python code/pretrain.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --output_dir ./outputs/pretrain \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3
```

**多卡 DDP 训练**：
```bash
# 自动检测 GPU 数量
torchrun --nproc_per_node auto code/pretrain.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --output_dir ./outputs/pretrain_ddp \
    --per_device_train_batch_size 2
```

**使用 DeepSpeed 优化**：
```bash
# 使用 DeepSpeed ZeRO-2 优化显存
deepspeed --num_gpus=4 code/pretrain.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --deepspeed code/ds_config_zero2.json \
    --bf16
```

**DeepSpeed 配置文件** (`ds_config_zero2.json`)：
```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "bf16": {
        "enabled": true
    },
    "gradient_clipping": 1.0
}
```

**监控训练**：
```bash
# 实时监控 GPU 状态
watch -n 1 rocm-smi

# 查看 VRAM 使用
rocm-smi | grep "GPU Memory"

# 使用 rocm-bandwidth-test 测试通信带宽
rocm-bandwidth-test
```

📖 [详细教程](./6.2-预训练实践.md) | 💻 [预训练脚本](./code/pretrain.py) | 📜 [执行脚本](./code/pretrain.sh)

---

### 6.3 - 微调实践

这一部分讲解如何使用主流框架高效进行模型微调。

#### 6.3.1 监督微调 (SFT)

**文件**: `code/finetune.py`, `code/finetune.sh`

**全量微调** (Fine-tune all parameters)：
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs/sft",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

**数据格式** (对话数据)：
```json
{
    "instruction": "What is Python?",
    "input": "",
    "output": "Python is a high-level programming language..."
}
```

---

#### 6.3.2 参数高效微调 (PEFT)

**文件**: `code/finetune.py`

**LoRA 微调**：
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # LoRA 秩
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 输出可训练参数数量
model.print_trainable_parameters()
```

**LoRA 参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `r` | LoRA 秩 | 8, 16, 32 |
| `lora_alpha` | 缩放因子 | r * 2 |
| `target_modules` | 目标模块 | ["q_proj", "v_proj"] |
| `lora_dropout` | LoRA Dropout | 0.05 ~ 0.1 |
| `bias` | 偏置训练 | "none", "lora_only", "all" |

**显存对比** (微调 7B 模型)：

| 方法 | 显存占用 | 可训练参数 |
|------|--------|---------|
| 全量微调 | ~28GB | 7B |
| LoRA (r=8) | ~12GB | 2.1M |
| LoRA (r=16) | ~14GB | 4.2M |
| QLoRA | ~6GB | 2.1M |

**QLoRA 微调** (极低显存)：
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

💻 [微调脚本](./code/finetune.py) | 📜 [执行脚本](./code/finetune.sh)

---

#### 6.3.3 性能优化

**显存优化**：
- 梯度检查点：降低显存 50%，速度降低 20%
- 梯度累积：模拟更大批大小而不增加显存
- 混合精度 (BF16)：显存减半，精度几乎无损
- 参数共享：某些情况下减少模型大小

**训练速度优化**：
- 增加 `num_workers` 加快数据加载
- 使用 FlashAttention 加速注意力计算
- 启用 cuDNN autotuner
- 使用混合精度训练

**DeepSpeed ZeRO 优化**：

| 阶段 | 优化内容 | 显存减少 |
|------|--------|--------|
| ZeRO-1 | 优化器状态分片 | 4× |
| ZeRO-2 | + 梯度分片 | 8× |
| ZeRO-3 | + 参数分片 | 8×+ |

---

### 6.4 - 偏好对齐

这一部分讲解模型对齐和偏好学习相关技术。

#### 6.4.1 强化学习微调 (RLHF)

**基本流程**：
1. 监督微调 (SFT)：在指令数据上微调模型
2. 奖励模型训练：训练奖励模型预测人类偏好
3. 强化学习：使用奖励信号优化策略

**RLHF 框架**：
```python
# 1. 奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "path/to/reward_model"
)

# 2. PPO 训练
from trl import PPOTrainer

ppo_trainer = PPOTrainer(
    model=model,
    reward_model=reward_model,
    args=training_args,
    train_dataset=train_dataset,
)

ppo_trainer.train()
```

---

#### 6.4.2 DPO 和 IPO

**DPO (Direct Preference Optimization)**：
- 不需要训练奖励模型
- 直接在偏好数据上优化
- 训练更简单高效

**数据格式**：
```json
{
    "prompt": "...",
    "chosen": "chosen response",
    "rejected": "rejected response"
}
```

**DPO 训练**：
```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    beta=0.1,  # DPO 温度参数
)

dpo_trainer.train()
```

📖 [详细教程](./6.4-偏好对齐.md)

---

## 环境要求

### 硬件
- AMD GPU：MI300X (192GB) 推荐，RX 6900 XT (16GB) 以上可用
- 系统内存：64GB+
- 存储空间：200GB+（用于模型、数据集、检查点）
- 网络：稳定的 HuggingFace Hub 连接

### 软件
- 操作系统：Linux (Ubuntu 22.04 LTS 或 24.04 LTS)
- ROCm：7.2.0 或更高版本
- Python：3.10+
- PyTorch：2.3+ (ROCm 版本)
- 主要依赖：
  - Transformers 4.36+
  - DeepSpeed 0.13+
  - PEFT (参数高效微调)
  - TRL (强化学习微调)
  - Datasets 2.14+
  - Accelerate 0.24+

### 一键安装

```bash
# 进入项目目录
cd chapter6/code

# 安装所有依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 快速开始

### 1. 环境准备

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定使用的 GPU

# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖
bash code/install_rocm_deps.sh
```

### 2. 下载模型和数据

```bash
# 下载预训练模型
python code/download_model.py --model_name Qwen/Qwen2.5-1.5B

# 下载训练数据
python code/download_dataset.py --dataset_name wikitext
```

### 3. 运行预训练

```bash
# 单卡预训练
python code/pretrain.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --output_dir ./outputs/pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2

# 多卡预训练 (4 GPU)
torchrun --nproc_per_node 4 code/pretrain.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --output_dir ./outputs/pretrain_ddp
```

### 4. 运行微调

```bash
# 全量微调
python code/finetune.py \
    --model_name_or_path ./outputs/pretrain/final \
    --output_dir ./outputs/sft \
    --learning_rate 5e-5

# LoRA 微调
python code/finetune.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --output_dir ./outputs/lora \
    --use_lora \
    --lora_rank 8
```

### 5. 模型推理

```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/sft/final")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

inputs = tokenizer("What is AI?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

---

## 性能基准

### 硬件配置

- **4 × MI300X** (768GB HBM)
- **8 × RX 7900 XTX** (128GB VRAM)

### 预训练性能

| 模型 | 显存/卡 | 吞吐量 | 时间/epoch |
|------|--------|--------|-----------|
| Qwen-7B (DDP) | 25GB | 600 tok/s | ~8 小时 |
| Llama-13B (ZeRO-2) | 32GB | 450 tok/s | ~12 小时 |
| Qwen-32B (ZeRO-3) | 35GB | 250 tok/s | ~24 小时 |

### 微调性能

| 方法 | 显存/卡 | 吞吐量 | 时间/epoch |
|------|--------|--------|-----------|
| SFT 全量 (7B) | 28GB | 250 tok/s | ~4 小时 |
| LoRA (7B) | 12GB | 400 tok/s | ~2 小时 |
| QLoRA (7B) | 6GB | 200 tok/s | ~3 小时 |

> 实际性能取决于硬件型号、数据集大小、超参数设置等因素

---

## 常见问题

<details>
<summary>Q: 如何处理 OOM 错误？</summary>

**A:** 按以下顺序尝试：

1. **减小批大小**：`per_device_train_batch_size = 1`
2. **启用梯度累积**：`gradient_accumulation_steps = 8`
3. **启用梯度检查点**：`gradient_checkpointing = True`
4. **使用 DeepSpeed ZeRO**：配置文件中设置 `"stage": 3`
5. **使用 LoRA**：参数量减少 99%
6. **启用 FlashAttention**：编译更快的注意力核心

</details>

<details>
<summary>Q: 多卡训练时，不同 GPU 负载不均衡？</summary>

**A:** 检查：

1. 确保所有 GPU 都被识别：`rocm-smi`
2. 检查通信限制：`rocm-bandwidth-test`
3. 增加批大小以降低通信开销
4. 检查是否有 GPU 被其他进程占用

</details>

<details>
<summary>Q: 如何选择合适的学习率？</summary>

**A:**
- **预训练**：1e-4 ~ 2e-4
- **全量微调**：5e-5 ~ 2e-4
- **LoRA 微调**：1e-4 ~ 5e-4
- **规则**：学习率 × 批大小应该保持稳定

</details>

<details>
<summary>Q: 如何在训练中恢复检查点？</summary>

**A:**
```bash
# 自动检测最新检查点
python code/pretrain.py \
    --output_dir ./outputs \
    --resume_from_checkpoint ./outputs/checkpoint-1000

# 或让 Trainer 自动恢复
# Trainer 会自动检测并从最新检查点恢复
```

</details>

---

## 下一步

完成本章后，你已经掌握了大模型训练的完整技能。建议：

1. **在自己的数据集上进行训练实验**
2. **优化超参数以获得最佳性能**
3. **尝试使用 RLHF 或 DPO 进行对齐**
4. **部署模型到生产环境**

📖 [返回主目录](../README.md) | 📖 [返回第五章](../chapter5/README.md)

---

<div align="center">

**掌握现代 LLM 训练技能，成为大模型工程师！** 🚀

</div>
