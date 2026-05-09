# 第五章：动手搭建大模型（AMD ROCm 版）

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

</div>

本章将从零实现一个 8000 万参数、LLaMA2 架构的语言模型，并在 AMD ROCm 平台上完成预训练与 SFT 微调。详细原理与代码解读可参考[第五章 动手搭建大模型.md](./第五章%20动手搭建大模型.md)。

对应英文教程：[chapter5-hands-on-llm-building.md](./chapter5-hands-on-llm-building.md)

---

## 文件结构

```
chapter5/
├── 第五章 动手搭建大模型.md      # 章节详细教程
├── chapter5-hands-on-llm-building.md  # 英文教程
├── README.md                    # 本文件
├── README_EN.md                 # 英文 README
└── code/
    ├── 00_download_dataset.sh          # 步骤 0：下载数据集（Linux）
    ├── 00_windows_download_dataset.sh  # 步骤 0：下载数据集（Windows）
    ├── 01_deal_dataset.py              # 步骤 1：预处理数据集
    ├── 02_train_tokenizer.py           # 步骤 2：训练 BPE Tokenizer
    ├── 03_ddp_pretrain.py              # 步骤 3：DDP 多卡预训练
    ├── 04_ddp_sft_full.py              # 步骤 4：DDP 多卡 SFT 微调
    ├── 05_model_sample.py              # 步骤 5：推理测试
    ├── 06_export_model.py              # 步骤 6：导出为 HuggingFace 格式
    ├── k_model.py                      # 模型定义（库文件）
    ├── dataset.py                      # 数据集类（库文件）
    ├── tokenizer_k/                    # 预训练 Tokenizer
    └── requirements.txt
```

---

## 执行流程

### 步骤 0：安装依赖并下载数据集

```bash
# 安装 ROCm 以及 ROCm 版本的 torch、torchvision、torchaudio
# 本测试使用 4*AMD Radeon™ AI PRO R9700（gfx1201 架构）。
# 如果你的 GPU 架构不同，请自行安装匹配架构的软件包。
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ "rocm[libraries,devel]"
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ torch torchvision torchaudio

# 安装依赖
pip install -r code/requirements.txt

# 下载数据集（需预先安装 modelscope）
cd code
bash 00_download_dataset.sh
```

> 本项目在 4*AMD Radeon™ AI PRO R9700 上完成验证。其他 Instinct/Radeon PRO/Radeon/Ryzen 系列设备兼容性可参考 https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html

脚本会将数据下载到 `./datasets/` 目录，包含：
- SeqMonkey 预训练语料（约 10B Token）
- BelleGroup 350 万条中文 SFT 样本

> Windows 用户请使用 `00_windows_download_dataset.sh`（支持 PowerShell 与 CMD）

---

### 步骤 1：预处理数据集

```bash
python 01_deal_dataset.py
```

**输入路径（在脚本顶部修改）：**

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `pretrain_data` | 原始预训练 jsonl 文件路径 | `./datasets/mobvoi_seq_monkey_general_open_corpus.jsonl` |
| `sft_data` | 原始 SFT json 文件路径 | `./datasets/BelleGroup/train_3.5M_CN.json` |
| `output_pretrain_data` | 处理后的预训练数据输出路径 | `./tmp/seq_monkey_datawhale.jsonl` |
| `output_sft_data` | 处理后的 SFT 数据输出路径 | `./tmp/BelleGroup_sft.jsonl` |

---

### 步骤 2：训练 BPE Tokenizer

```bash
python 02_train_tokenizer.py
```

**关键参数（在 `main()` 函数中修改）：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data_path` | 用于训练 Tokenizer 的 jsonl 数据路径 | `./tmp/seq_monkey_datawhale.jsonl` |
| `save_dir` | Tokenizer 保存目录 | `./tokenizer_k/` |
| `vocab_size` | 词表大小 | `6144` |

> 项目已提供预训练好的 Tokenizer（`tokenizer_k/`），可跳过本步骤直接开始训练。

---

### 步骤 3：多卡预训练（DDP）

```bash
# 使用 ROCm 多卡进行 DDP 训练
python 03_ddp_pretrain.py --gpus 0,1,2,3 --data_path ./tmp/seq_monkey_datawhale.jsonl
```

**必填参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpus` | 使用的 GPU ID，逗号分隔 | `0,1,2,3` |
| `--data_path` | 预训练数据路径（步骤 1 输出） | `./datasets/mobvoi_seq_monkey_general_open_corpus.jsonl` |

**常用可选参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--out_dir` | 模型权重输出目录 | `base_model_215M` |
| `--epochs` | 训练轮数 | `1` |
| `--batch_size` | 每卡 batch size（显存不足时可调小） | `64` |
| `--learning_rate` | 初始学习率 | `2e-4` |
| `--dtype` | 训练精度，ROCm 推荐 `bfloat16` | `bfloat16` |
| `--accumulation_steps` | 梯度累积步数 | `8` |
| `--use_swanlab` | 是否启用 SwanLab 训练可视化 | 关闭 |

> **ROCm 提示**：在 ROCm 平台下，`cuda` 设备会自动映射到 AMD GPU，无需额外修改设备参数。若显存不足，可将 `--batch_size` 降到 4~8，将显存需求降至约 7GB。

---

### 步骤 4：多卡 SFT 微调（DDP）

```bash
python 04_ddp_sft_full.py --gpus 0,1,2,3 --data_path ./tmp/BelleGroup_sft.jsonl --out_dir sft_model_215M
```

**必填参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpus` | 使用的 GPU ID，逗号分隔 | `0,1,2,3` |
| `--data_path` | SFT 数据路径（步骤 1 输出） | `./BelleGroup_sft.jsonl` |
| `--out_dir` | 微调模型权重输出目录 | `sft_model_215M` |

**常用可选参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | `1` |
| `--batch_size` | 每卡 batch size | `64` |
| `--learning_rate` | 初始学习率 | `2e-4` |
| `--dtype` | 训练精度 | `bfloat16` |
| `--accumulation_steps` | 梯度累积步数 | `8` |
| `--use_swanlab` | 是否启用 SwanLab 训练可视化 | 关闭 |

---

### 步骤 5：推理测试

```bash
python 05_model_sample.py
```

默认加载 `./base_model_215M/pretrain_1024_18_6144.pth`（预训练）与 `./sft_model_215M/`（SFT）。你可以修改脚本顶部 `TextGenerator` 初始化参数切换模型：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `checkpoint` | 模型 `.pth` 权重路径 | `./base_model_215M/pretrain_1024_18_6144.pth` |
| `tokenizer_model_path` | Tokenizer 路径 | `./tokenizer_k/` |
| `device` | 推理设备 | 自动检测 |
| `dtype` | 推理精度 | `bfloat16` |

---

### 步骤 6：导出为 HuggingFace 格式（可选）

```bash
python 06_export_model.py
```

修改脚本底部 `export_model()` 调用中的参数：

| 参数 | 说明 |
|------|------|
| `tokenizer_path` | Tokenizer 目录路径 |
| `model_ckpt_path` | 训练后 `.pth` 权重路径 |
| `save_directory` | HuggingFace 格式模型保存目录 |

导出后可通过 `AutoModelForCausalLM.from_pretrained()` 直接加载。

---

## 训练参考数据

测试环境：4 x AMD Radeon™ AI PRO R9700（ROCm 7.12）

| 阶段 | 数据规模 | 硬件 | 耗时参考 |
|------|--------|------|----------|
| 预训练 | 完整 SeqMonkey 数据集 | 8 x GPU | ~46 小时 |
| SFT | BelleGroup 350 万样本 | 8 x GPU | ~24 小时 |
| 预训练（batch=4，单卡） | - | 1 x GPU（7GB 显存） | ~533 小时 |

---

## 预训练模型下载

如果硬件不足以完成完整训练，可以直接下载作者已训练好的模型：

- 🤖 [ModelScope 模型下载](https://www.modelscope.cn/collections/Happy-LLM-e98b91b10b684a)
- 🎮 [ModelScope 创空间演示](https://www.modelscope.cn/studios/kmno4zx/happy_llm_215M_sft)
