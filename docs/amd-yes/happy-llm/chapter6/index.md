# 第六章：大模型训练流程实践（AMD ROCm 版）

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

</div>

本章基于 Transformers + DeepSpeed 框架，在 AMD ROCm 上实现完整的大模型训练流程，覆盖预训练与 SFT 微调。有关框架介绍与训练原理的详细说明，请参考[第六章 大模型训练流程实践.md](./第六章%20大模型训练流程实践.md)。

对应英文教程：[chapter6-llm-training-workflow-practice.md](./chapter6-llm-training-workflow-practice.md)

6.4 小节英文版本：[chapter6-4-wip-preference-alignment.md](./chapter6-4-wip-preference-alignment.md)

---

## 文件结构

```
chapter6/
├── 第六章 大模型训练流程实践.md    # 章节详细教程
├── 6.4[WIP] 偏好对齐.md           # 6.4 小节
├── chapter6-llm-training-workflow-practice.md  # 英文教程
├── chapter6-4-wip-preference-alignment.md      # 6.4 英文版
├── README.md                      # 本文件
├── README_EN.md                   # 英文 README
└── code/
    ├── 00_download_model.py        # 步骤 0：下载基座模型（Qwen2.5-1.5B）
    ├── 01_download_dataset.py      # 步骤 1a：下载数据集
    ├── 01_process_dataset.ipynb    # 步骤 1b：数据处理（Notebook）
    ├── 02_pretrain.py              # 步骤 2：预训练脚本
    ├── 02_pretrain.sh              # 步骤 2：预训练启动脚本（DeepSpeed）
    ├── 02_pretrain.ipynb           # 步骤 2：预训练（Notebook）
    ├── 03_finetune.py              # 步骤 3：SFT 微调脚本
    ├── 03_finetune.sh              # 步骤 3：SFT 微调启动脚本（DeepSpeed）
    ├── ds_config_zero2.json        # DeepSpeed ZeRO-2 配置
    ├── whole.ipynb                 # 全流程 Notebook
    └── requirements.txt
```

---

## 执行流程

### 步骤 0：安装依赖并下载基座模型

```bash
# 安装 ROCm 以及 ROCm 版本的 torch、torchvision、torchaudio
# 本测试使用 4*AMD Radeon™ AI PRO R9700（gfx1201 架构）。
# 如果你的 GPU 架构不同，请自行安装匹配架构的软件包。
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ "rocm[libraries,devel]"
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ torch torchvision torchaudio

# 安装依赖
pip install -r code/requirements.txt

# 下载 Qwen2.5-1.5B 作为基座模型（通过 hf-mirror 加速）
cd code
python 00_download_model.py
```

> 本项目在 4*AMD Radeon™ AI PRO R9700 上完成验证。其他 Instinct/Radeon PRO/Radeon/Ryzen 系列设备兼容性可参考 https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html

模型将下载到 `autodl-tmp/qwen-1.5b/`。如需下载其他模型，请修改脚本中 `huggingface-cli download` 的目标仓库及 `--local-dir` 参数。

---

### 步骤 1：下载并处理数据集

```bash
# 下载数据集
python 01_download_dataset.py
```

| 数据集 | 用途 | 下载路径 |
|--------|------|----------|
| seq-monkey（ModelScope） | 预训练语料 | `autodl-tmp/dataset/pretrain_data/` |
| BelleGroup/train_3.5M_CN | SFT 指令数据 | `autodl-tmp/dataset/sft_data/BelleGroup/` |

> 可使用 `01_process_dataset.ipynb` 进行交互式数据处理与分析。

---

### 步骤 2：预训练（DeepSpeed ZeRO-2）

**推荐方式：直接运行 Shell 启动脚本**

```bash
bash 02_pretrain.sh
```

**`02_pretrain.sh` 中的关键参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 可见 GPU；在 ROCm 下等价于 `HIP_VISIBLE_DEVICES` | `0,1` |
| `--config_name` | 模型结构配置目录（用于从头训练） | `autodl-tmp/qwen-1.5b` |
| `--tokenizer_name` | Tokenizer 目录 | `autodl-tmp/qwen-1.5b` |
| `--train_files` | 预训练数据文件路径 | `autodl-tmp/dataset/pretrain_data/...jsonl` |
| `--output_dir` | 训练输出目录（checkpoints） | `autodl-tmp/output/pretrain` |
| `--per_device_train_batch_size` | 每卡 batch size | `16` |
| `--gradient_accumulation_steps` | 梯度累积步数 | `4` |
| `--num_train_epochs` | 训练轮数 | `1` |
| `--learning_rate` | 学习率 | `1e-4` |
| `--block_size` | 最大文本长度（token 数） | `2048` |
| `--bf16` | 启用 bfloat16 精度（ROCm 推荐） | 启用 |
| `--deepspeed` | DeepSpeed 配置文件 | `./ds_config_zero2.json` |

> **ROCm 说明**：`--bf16` 在 AMD MI 与 RDNA 架构上均可原生支持。DeepSpeed ZeRO-2 可显著降低单卡显存占用，并支持更大的 batch size。

---

### 步骤 3：SFT 微调（DeepSpeed ZeRO-2）

**推荐方式：直接运行 Shell 启动脚本**

```bash
bash 03_finetune.sh
```

**`03_finetune.sh` 中的关键参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 可见 GPU | `0,1` |
| `--model_name_or_path` | 预训练模型或 checkpoint 路径 | `autodl-tmp/qwen-1.5b` |
| `--train_files` | SFT 数据文件路径（json 格式） | `autodl-tmp/dataset/sft_data/BelleGroup/train_3.5M_CN.json` |
| `--output_dir` | 微调输出目录 | `autodl-tmp/output/sft` |
| `--per_device_train_batch_size` | 每卡 batch size | `16` |
| `--gradient_accumulation_steps` | 梯度累积步数 | `4` |
| `--num_train_epochs` | 训练轮数 | `3` |
| `--learning_rate` | 学习率 | `1e-4` |
| `--block_size` | 最大文本长度 | `2048` |
| `--bf16` | 启用 bfloat16 精度 | 启用 |
| `--deepspeed` | DeepSpeed 配置文件 | `./ds_config_zero2.json` |

**从 checkpoint 恢复训练（两个脚本都支持）：**

```bash
# 取消脚本最后一行注释，并替换为你的 checkpoint 路径
--resume_from_checkpoint autodl-tmp/output/pretrain/checkpoint-XXXXX
```

---

## DeepSpeed ZeRO-2 配置

`ds_config_zero2.json` 已为 AMD ROCm 预置关键配置，核心选项如下：

| 选项 | 说明 |
|------|------|
| `zero_optimization.stage: 2` | 优化器状态 + 梯度分片，平衡多卡显存占用 |
| `bf16.enabled: true` | ROCm bfloat16 加速 |
| `gradient_clipping: 1.0` | 梯度裁剪阈值 |

如需进一步降低显存占用，可将 `stage` 调整为 `3`（全参数分片）。

---

## 训练监控

两个训练脚本均集成了 SwanLab 可视化：

```bash
# .sh 文件中已包含 --report_to swanlab
# 首次使用前先登录
swanlab login
```
