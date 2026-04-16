# 第五章：动手搭建大模型 AMD ROCm 版本

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

</div>

## 简介

&emsp;&emsp;本章从零开始，深入讲解大语言模型（LLM）的核心原理和完整实现。我们将以 LLaMA2 为例，从最基础的组件开始，逐步构建一个功能完整的大模型。

&emsp;&emsp;通过本章的学习，你将理解 Transformer 架构的每个细节，掌握模型的完整实现，并学会如何进行预训练和微调。这些知识是理解和优化大模型的基础。

## 学习目标

- ✅ 理解 Transformer 基础架构和自注意力机制
- ✅ 实现 LLaMA2 模型的所有核心组件
- ✅ 掌握数据加载、预处理的完整流程
- ✅ 理解和实现模型的预训练过程
- ✅ 学习参数高效的微调方法（LoRA）
- ✅ 在 AMD GPU 上进行高效训练

## 章节内容

### 5.1 - 动手实现 LLaMA2 大模型

这一部分详细讲解 LLaMA2 模型的完整结构实现。

#### 5.1.1 模型超参数配置
**文件**: `code/model_config.py`
- ModelConfig 类设计
- 超参数含义解释
- ROCm 优化参数设置

**关键概念**：
- `dim`: 模型的隐藏层维度（通常 256~4096）
- `n_layers`: Transformer 堆叠的层数
- `n_heads`: 多头注意力的头数
- `vocab_size`: 词汇表大小
- `max_seq_len`: 最大输入序列长度

**代码框架**：
```python
from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "llama"
    
    def __init__(self, dim=4096, n_layers=32, n_heads=32, ...):
        # 初始化所有超参数
        pass
```

📖 [详细教程](./5.1-模型结构设计.md) | 💻 [完整代码](./code/model_config.py)

---

#### 5.1.2 核心组件实现

**RMSNorm（根均方范数归一化）**
- 数学原理讲解
- 与 LayerNorm 的对比
- PyTorch 实现
- ROCm 下的优化技巧

**代码示例**：
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # RMSNorm 的核心计算
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

**多头自注意力 (Multi-Head Attention)**
- 自注意力机制详解
- 多头并行计算
- KV Cache 优化
- FlashAttention 支持

**前馈网络 (Feed Forward Network)**
- MLP 结构设计
- 激活函数选择（SiLU vs ReLU）
- 维度扩展比例

**位置编码 (Rotary Positional Embedding)**
- 相对位置信息编码
- 与绝对位置编码的区别
- 旋转矩阵的数学原理

📖 [详细教程](./5.1-模型结构设计.md) | 💻 [完整实现](./code/k_model.py)

---

#### 5.1.3 完整模型架构

将所有组件组合成完整的 LLaMA2 模型。

**模型流程**：
```
输入 Token
    ↓
Token Embedding (词嵌入)
    ↓
[重复 n_layers 次]:
  ├─ RMSNorm (前置归一化)
  ├─ Attention (自注意力)
  ├─ 残差连接
  ├─ RMSNorm
  ├─ FFN (前馈网络)
  └─ 残差连接
    ↓
最终 RMSNorm
    ↓
输出投影 (Output Projection)
    ↓
Logits (预测概率)
```

**关键特性**：
- 前置归一化 (Pre-LN) 设计
- 残差连接确保梯度流动
- KV Cache 支持推理优化

💻 [完整模型代码](./code/k_model.py)

---

### 5.2 - 预训练与微调

这一部分讲解如何使用实现好的模型进行预训练和微调。

#### 5.2.1 数据准备

**文件**: `code/dataset.py`

**数据加载流程**：
1. 原始文本数据 → 分词 (Tokenization)
2. Token 序列 → 样本构造 (样本长度 = max_seq_len)
3. 样本 → 批次组织 (DataLoader)
4. 批次 → GPU 加载

**关键操作**：
```python
# 1. 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# 2. 数据处理函数
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, 
                       max_length=max_seq_len)
    return tokens

# 3. 创建 DataLoader
train_dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True
)
```

**数据集来源**：
- 🤗 Hugging Face Datasets (推荐)
- 本地文本文件
- 自定义数据格式

📖 [详细教程](./5.2-预训练与微调.md) | 💻 [数据加载代码](./code/dataset.py)

---

#### 5.2.2 预训练过程

**文件**: `code/pretrain.py`

**预训练目标**：
- 因果语言模型 (Causal Language Modeling, CLM)
- 自回归预测：根据前 N 个 token，预测第 N+1 个 token
- 损失函数：交叉熵损失

**训练流程**：
```python
model = LLaMA2(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 前向传播
        logits = model(batch['input_ids'])
        
        # 计算损失
        loss = cross_entropy_loss(logits, batch['labels'])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 监控训练进度
        log_loss(loss)
```

**ROCm 优化技巧**：
- 启用混合精度训练 (BF16)
- 梯度累积增加有效批大小
- 梯度检查点降低显存占用
- 分布式数据并行 (DDP)

**监控指标**：
- 训练损失 (Training Loss)
- 验证损失 (Validation Loss)
- 困惑度 (Perplexity)
- GPU 利用率和吞吐量

📖 [详细教程](./5.2-预训练与微调.md) | 💻 [预训练脚本](./code/pretrain.py)

---

#### 5.2.3 微调过程

**文件**: `code/finetune.py`

**微调方式**：

1. **全量微调 (Full Fine-tuning)**
   - 更新所有参数
   - 显存占用大，效果最好

2. **LoRA 微调 (Low-Rank Adaptation)**
   - 只更新低秩矩阵
   - 参数量大幅减少 (< 1%)
   - 推荐用于资源受限场景

**LoRA 原理**：
```
标准微调: W_new = W + ΔW
LoRA 微调: ΔW = A · B^T  (低秩分解)
```

其中 A 和 B 是小矩阵，学习参数大幅减少。

**微调脚本框架**：
```python
# 加载预训练模型
model = LLaMA2.from_pretrained("path/to/pretrained")

# 冻结大部分参数，只训练指定层
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True

# 标准微调训练过程
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)
# ... 训练循环
```

📖 [详细教程](./5.2-预训练与微调.md) | 💻 [微调脚本](./code/finetune.py)

---

## 环境要求

### 硬件
- AMD GPU: RX 6700 XT (12GB) 以上，推荐 MI300X
- 系统内存: 32GB+
- 存储空间: 50GB+

### 软件
- ROCm: 7.2.0+
- Python: 3.10+
- PyTorch: 2.3+ (ROCm 版本)
- Transformers: 4.36+

### 依赖安装

```bash
# 进入项目目录
cd chapter5/code

# 安装依赖
pip install -r requirements.txt

# 对于 AMD GPU，确保安装正确的 PyTorch 版本
pip install torch torchvision torchaudio --index-url https://repo.amd.com/rocm/whl/rocm-7.2
```

---

## 快速开始

### 1. 验证模型实现

```bash
cd code

# 测试模型是否能正常运行
python model_config.py
python k_model.py

# 输出模型参数量和结构信息
```

### 2. 准备训练数据

```bash
# 使用示例数据集进行快速测试
python dataset.py --mode test

# 或下载完整数据集
bash download_dataset.sh
```

### 3. 运行预训练

```bash
# 单卡训练
python pretrain.py \
    --model_name "llama2-7b" \
    --batch_size 4 \
    --num_epochs 3 \
    --output_dir "./checkpoints"

# 多卡训练 (DDP)
torchrun --nproc_per_node 4 pretrain.py \
    --model_name "llama2-7b" \
    --batch_size 2 \
    --num_epochs 3
```

### 4. 运行微调

```bash
# LoRA 微调
python finetune.py \
    --model_name "llama2-7b" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir "./lora_checkpoints"
```

### 5. 监控训练

在另一个终端运行：

```bash
# 监控 GPU 状态
watch rocm-smi

# 查看训练日志
tail -f training.log

# 使用 TensorBoard 可视化 (可选)
tensorboard --logdir ./runs
```

---

## 常见问题

<details>
<summary>Q: 运行时显示 "CUDA out of memory" 怎么办？</summary>

**A:** 这个错误在 ROCm 中对应 "HIP out of memory"。尝试以下解决方案：

1. **减小批大小**：
   ```bash
   python pretrain.py --batch_size 1 --gradient_accumulation_steps 8
   ```

2. **启用梯度检查点**：
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **使用 LoRA 微调**而不是全量微调

4. **减小模型大小**或使用量化版本

</details>

<details>
<summary>Q: ROCm 无法检测到 GPU 怎么办？</summary>

**A:** 按以下步骤排查：

```bash
# 1. 检查 GPU 是否被识别
rocminfo | grep "GPU"

# 2. 检查环境变量
echo $LD_LIBRARY_PATH

# 3. 重新初始化 HIP
export HIP_VISIBLE_DEVICES=0

# 4. 如果还是不行，尝试指定 GPU 架构
export HSA_OVERRIDE_GFX_VERSION=gfx90a
```

</details>

<details>
<summary>Q: 预训练速度很慢怎么办？</summary>

**A:** 检查和优化：

1. **检查 GPU 利用率**：
   ```bash
   rocm-smi --json | grep "use"
   ```

2. **启用混合精度训练**：
   ```bash
   python pretrain.py --bf16  # 使用 BF16
   ```

3. **增加批大小**（如果显存允许）

4. **启用分布式训练**使用多 GPU

5. **检查数据加载**是否是瓶颈，可增加 `num_workers`

</details>

<details>
<summary>Q: 如何导出训练好的模型供推理使用？</summary>

**A:** 使用提供的导出脚本：

```bash
python code/export_model.py \
    --model_path "./checkpoints/final" \
    --output_format "huggingface"
```

导出后可以使用标准的 Hugging Face API 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/exported")
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")
```

</details>

---

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `model_config.py` | 模型超参数配置 |
| `k_model.py` | LLaMA2 完整模型实现 |
| `dataset.py` | 数据加载和预处理 |
| `pretrain.py` | 预训练脚本 |
| `finetune.py` | 微调脚本 |
| `download_dataset.sh` | 数据集下载脚本 |
| `export_model.py` | 模型导出脚本 |
| `requirements.txt` | Python 依赖包列表 |

---

## 学习路径建议

**时间**: 约 20-30 小时

1. **第 1-2 小时**：理解 Transformer 基础（阅读文档 5.1）
2. **第 3-8 小时**：学习并实现模型组件（对应 5.1.2-5.1.3）
3. **第 9-12 小时**：数据准备和预训练流程（对应 5.2.1-5.2.2）
4. **第 13-16 小时**：运行小规模预训练，理解训练过程
5. **第 17-20 小时**：学习微调方法（对应 5.2.3）
6. **第 21-30 小时**：在更大数据集上训练，优化性能

---

## 下一步

完成本章后，你将准备好进入 [第六章：大模型训练流程实践](../chapter6/README.md)，学习如何使用主流框架和分布式技术进行工业级训练。

📖 [返回主目录](../README.md) | 📖 [进入第六章](../chapter6/README.md)

---

<div align="center">

**理解大模型原理的最好方式，就是从零开始实现它！** 💡

</div>
