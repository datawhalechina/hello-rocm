# 🚀 Happy-LLM - 从零训练大模型 AMD ROCm 版本

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

</div>

**Happy-LLM** 是一套完整的大语言模型（LLM）学习与实践教程，从零开始讲解大模型的核心原理和实现方法，包含模型结构设计、预训练、微调等全流程。本版本已适配 AMD ROCm 7.2.0+ 平台，支持 Linux 环境的多卡分布式训练。

> Happy-LLM 原始项目地址：[*Link*](https://github.com/datawhalechina/happy-llm.git)

***本教程将从零开始，带领你手把手实现一个完整的大语言模型，深入理解 LLM 的原理与实践！***

## 简介

&emsp;&emsp;本模块基于 Happy-LLM 教程，针对 AMD GPU 和 ROCm 平台进行了优化和适配。它包含两个核心章节：

- **第五章：动手搭建大模型** - 从零开始实现 LLaMA2 模型的完整结构，包括 RMSNorm、Attention、FFN 等核心组件
- **第六章：大模型训练流程实践** - 使用主流框架（Transformers、DeepSpeed）实现预训练和微调的完整流程

&emsp;&emsp;通过本教程，你将深入理解大模型的工作原理，掌握在 AMD GPU 上高效训练 LLM 的技能。

## 核心内容

### 📚 第五章：动手搭建大模型

深入讲解大语言模型的核心原理和实现细节。

#### 学习目标
- ✅ 理解 Transformer 架构和注意力机制
- ✅ 从零实现 LLaMA2 模型的完整结构
- ✅ 掌握模型预训练的全流程
- ✅ 实现参数高效的微调方法（LoRA）

#### 主要内容
- **5.1 - 动手实现 LLaMA2 大模型**
  - 超参数定义与 ModelConfig 类设计
  - RMSNorm 归一化层实现
  - Attention 多头注意力机制
  - 前馈网络（FFN）实现
  - 完整 LLaMA2 模型架构

- **5.2 - 预训练与微调**
  - 数据加载与预处理
  - 模型预训练流程
  - 监督微调（SFT）实现
  - 模型保存与导出

📖 [进入第五章](./chapter5/README.md)

---

### 🔧 第六章：大模型训练流程实践

使用主流框架实现工业级的大模型训练。

#### 学习目标
- ✅ 掌握 Transformers 框架的使用方法
- ✅ 实现多卡分布式训练（DDP、DeepSpeed）
- ✅ 学习参数高效微调技术（LoRA、QLoRA）
- ✅ 优化训练性能与显存使用

#### 主要内容
- **6.1 - 框架介绍与基础**
  - Transformers 框架概述
  - Trainer API 讲解
  - 配置文件与超参数设置

- **6.2 - 预训练实践**
  - 模型下载与初始化
  - 数据集加载与处理
  - 分布式预训练脚本
  - 检查点保存与恢复

- **6.3 - 微调实践**
  - 监督微调（SFT）
  - 参数高效微调（PEFT）
  - DeepSpeed 零阶段优化
  - 模型量化与推理优化

- **6.4 - 偏好对齐**
  - 强化学习微调（RLHF）
  - DPO、IPO 等对齐方法

📖 [进入第六章](./chapter6/README.md)

---

## 环境要求

### 硬件要求

- **GPU**：AMD RDNA 2/3 系列（如 RX 6700 XT、RX 7900 XTX）或 MI 系列（如 MI300X）
- **显存**：建议 16GB+ 以上（8GB 可运行小模型）
- **系统内存**：32GB+ 推荐
- **存储空间**：100GB+（用于模型和数据集）

### 软件要求

- **操作系统**：Linux (Ubuntu 22.04 LTS / 24.04 LTS 推荐)
- **ROCm 版本**：7.2.0 或更高版本
- **Python 版本**：3.10+
- **CUDA/HIP 编译工具链**：rocm-developer-tools

### 依赖库

核心依赖包括：
- PyTorch 2.3+ (ROCm 版本)
- Transformers 4.36+
- DeepSpeed 0.13+
- PEFT (参数高效微调)
- Datasets
- Accelerate

详细安装指南请查看 [环境准备文档](./docs/environment-setup.md)

---

## 快速开始

### 第一步：环境准备

```bash
# 1. 克隆或复制本项目
cd 04-happy-llm

# 2. 创建 Python 虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 3. 安装 ROCm 相关依赖
bash code/install_rocm_deps.sh

# 4. 安装项目依赖
pip install -r chapter5/code/requirements.txt
pip install -r chapter6/code/requirements.txt
```

### 第二步：选择学习路径

**初学者推荐路线：**
1. 先完成第五章第一节，理解 LLaMA2 的完整结构
2. 按顺序完成第五章，掌握从零实现大模型
3. 进入第六章，学习使用主流框架训练模型

**进阶开发者路线：**
1. 跳过第五章基础部分，从第五章第二节开始
2. 重点学习第六章的分布式训练和优化技术
3. 完成性能基准测试和优化案例

### 第三步：运行示例

```bash
# 运行第五章模型实现示例
cd chapter5/code
python k_model.py

# 运行第六章预训练脚本
cd ../../chapter6/code
bash pretrain.sh
```

---

## 关键技术点

### 模型架构
| 组件 | 说明 |
|------|------|
| **Token Embedding** | 词嵌入层 |
| **RMSNorm** | 根均方范数归一化 |
| **Attention** | 多头自注意力机制，支持 FlashAttention2 |
| **FFN** | 前馈网络 (MLP) |
| **Rotary Embedding** | 旋转位置编码 |

### 训练优化
| 技术 | 用途 |
|------|------|
| **DDP** | 数据并行分布式训练 |
| **DeepSpeed ZeRO** | 内存优化与参数分片 |
| **FlashAttention** | 降低注意力计算复杂度 |
| **PEFT/LoRA** | 参数高效微调 |
| **梯度累积** | 增大有效批大小 |
| **混合精度训练** | 降低显存占用 |

---

## 性能参考

### 第五章（纯 PyTorch 实现）

在 AMD MI300X (192GB HBM) 上的性能参考：

| 模型 | 显存占用 | 训练速度 | 备注 |
|------|--------|--------|------|
| LLaMA2-7B | ~14GB | ~150 tokens/s | 单卡，fp32 |
| LLaMA2-7B | ~7GB | ~200 tokens/s | 单卡，bf16 混合精度 |
| LLaMA2-13B | ~26GB | ~100 tokens/s | 单卡，bf16 |

> 注：实际性能取决于具体硬件配置、模型实现和超参数设置

### 第六章（Transformers + DeepSpeed）

在 2×MI300X (384GB HBM) 上的性能参考：

| 模型 | 显存/卡 | 训练速度 | 并行方式 |
|------|--------|--------|--------|
| Qwen-7B | ~20GB | ~500 tokens/s | DDP + ZeRO-2 |
| Llama-13B | ~35GB | ~350 tokens/s | DDP + ZeRO-2 |
| Qwen-32B | ~40GB | ~200 tokens/s | DDP + ZeRO-3 |

> 实际运行结果需在你的服务器上进行性能基准测试

---

## 常见问题

<details>
<summary>Q: ROCm 找不到 GPU 怎么办？</summary>

**A:** 按以下步骤排查：

1. 验证 GPU 和驱动：
   ```bash
   rocminfo
   rocm-smi
   ```

2. 检查环境变量：
   ```bash
   echo $LD_LIBRARY_PATH
   export HSA_OVERRIDE_GFX_VERSION=gfx90a  # 对某些 GPU 可能需要
   ```

3. 重新安装 PyTorch ROCm 版本：
   ```bash
   pip install torch torchvision torchaudio --index-url https://repo.amd.com/rocm/whl/11.8
   ```

4. 如果问题仍存在，请查看 [ROCm 官方故障排除指南](https://rocm.docs.amd.com/en/docs-7.2.0/deploy/linux/index.html)

</details>

<details>
<summary>Q: 显存不足（OOM）如何解决？</summary>

**A:** 尝试以下方法（优先级从高到低）：

1. **减小批大小** - 在配置文件中设置 `per_device_train_batch_size = 1`
2. **启用梯度累积** - 设置 `gradient_accumulation_steps = 8`
3. **使用混合精度** - 设置 `fp16 = true` 或 `bf16 = true`
4. **启用 DeepSpeed ZeRO-3** - 完整参数分片
5. **使用 LoRA 微调** - 而不是全量微调
6. **启用 FlashAttention** - 降低注意力显存占用

</details>

<details>
<summary>Q: 多卡训练速度不理想怎么办？</summary>

**A:** 检查和优化：

1. 验证多卡识别：
   ```bash
   rocm-smi  # 查看所有 GPU
   ```

2. 检查通信带宽：
   ```bash
   rocm-bandwidth-test
   ```

3. 在训练脚本中启用 distributed debug：
   ```bash
   export NCCL_DEBUG=INFO
   # 或 HIP 的等价环境变量
   ```

4. 尝试增加 `num_train_epochs` 和 `logging_steps` 以平衡通信开销

</details>

<details>
<summary>Q: 如何在服务器上实际运行这些项目？</summary>

**A:** 推荐流程：

1. **环境准备**（第一次运行）
   - 运行 `code/install_rocm_deps.sh` 安装所有依赖
   - 运行 `code/setup_environment.sh` 配置环境变量

2. **验证环境**
   - 运行 `code/performance_benchmark.py` 进行性能测试
   - 查看输出的 GPU 利用率和吞吐量

3. **选择模型**
   - 根据硬件显存选择合适的模型大小
   - 参考"性能参考"表格调整超参数

4. **开始训练**
   - 从 chapter5 开始学习基础概念
   - 按 chapter6 的脚本进行实际训练

5. **监控和优化**
   - 使用 `rocm-smi` 实时监控 GPU 状态
   - 根据训练日志调整超参数

详细的运行指南见各章节文档。

</details>

<details>
<summary>Q: 第五章和第六章有什么区别？</summary>

**A:** 
- **第五章**：纯 PyTorch 实现，从零搭建 LLaMA2，适合理解模型原理
- **第六章**：使用 Transformers 框架，重点在于工业级训练和优化，适合实际应用

如果想快速上手训练，可以直接进入第六章。如果想深入理解模型原理，建议先完成第五章。

</details>

---

## 项目结构

```
04-happy-llm/
├── README.md                      # 项目总览（本文件）
├── chapter5/                      # 第五章：动手搭建大模型
│   ├── README.md                  # 章节指南
│   ├── 5.1-模型结构设计.md       # LLaMA2 核心组件
│   ├── 5.2-预训练与微调.md       # 完整训练流程
│   └── code/                      # 实现代码
│       ├── k_model.py             # LLaMA2 完整模型实现
│       ├── model_config.py        # 模型配置
│       ├── dataset.py             # 数据加载器
│       ├── pretrain.py            # 预训练脚本
│       ├── finetune.py            # 微调脚本
│       ├── requirements.txt        # 依赖包列表
│       └── tokenizer_k/           # 自定义分词器
├── chapter6/                      # 第六章：大模型训练流程实践
│   ├── README.md                  # 章节指南
│   ├── 6.1-框架与基础.md         # Transformers 框架讲解
│   ├── 6.2-预训练实践.md         # 分布式预训练
│   ├── 6.3-微调实践.md           # 参数高效微调
│   ├── 6.4-偏好对齐.md           # RLHF 和 DPO
│   └── code/                      # 实现代码
│       ├── download_model.py      # 模型下载脚本
│       ├── download_dataset.py    # 数据集下载脚本
│       ├── pretrain.py            # 预训练脚本
│       ├── pretrain.sh            # 预训练执行脚本
│       ├── finetune.py            # 微调脚本
│       ├── finetune.sh            # 微调执行脚本
│       ├── ds_config_zero2.json   # DeepSpeed ZeRO-2 配置
│       ├── requirements.txt        # 依赖包列表
│       └── notebooks/             # Jupyter 交互式教程
├── code/                          # 共用工具和脚本
│   ├── install_rocm_deps.sh       # 一键安装 ROCm 依赖（Linux）
│   ├── setup_environment.sh       # 环境配置脚本
│   ├── performance_benchmark.py   # 性能基准测试
│   └── README.md                  # 工具使用说明
├── images/                        # 文档图片（运行结果、截图等）
│   ├── chapter5_training.png      # 第五章训练曲线
│   ├── chapter6_multi_gpu.png     # 多卡训练监控
│   └── ...
└── docs/                          # 补充文档
    ├── environment-setup.md       # 详细的环境搭建指南
    ├── rocm-troubleshooting.md    # ROCm 故障排除
    └── performance-tuning.md      # 性能调优指南
```

---

## 参考资源

- 📚 [Happy-LLM 原始项目](https://github.com/datawhalechina/happy-llm)
- 📖 [ROCm 官方文档](https://rocm.docs.amd.com/)
- 🤗 [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- ⚡ [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- 🔧 [PyTorch 官方文档](https://pytorch.org/docs/)

---

## 贡献与反馈

欢迎提交 Issue 和 Pull Request！

- 📝 改进文档和教程
- 🐛 报告 Bug
- 💡 分享优化技巧
- 📊 补充性能测试结果

[提交 Issue](https://github.com/datawhalechina/hello-rocm/issues) | [提交 PR](https://github.com/datawhalechina/hello-rocm/pulls)

---

<div align="center">

**开始学习大模型，从零构建 LLM！** 🎓

Made with ❤️ by the hello-rocm community

</div>
