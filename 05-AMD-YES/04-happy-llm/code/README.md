# 工具和脚本说明

本目录包含 Happy-LLM 项目的通用工具和脚本，用于环境配置、性能测试等。

## 脚本列表

### 1. install_rocm_deps.sh

一键安装 AMD ROCm 所有依赖的 Shell 脚本。

**使用方式**：
```bash
bash install_rocm_deps.sh
```

**功能**：
- 检查 ROCm 版本
- 安装 PyTorch ROCm 版本
- 安装 Transformers、DeepSpeed 等主要依赖
- 验证安装成功

**适用系统**：
- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS
- 其他基于 Debian 的 Linux

---

### 2. setup_environment.sh

环境变量和配置脚本。

**使用方式**：
```bash
source setup_environment.sh
```

**功能**：
- 设置 Python 虚拟环境
- 配置 ROCm 相关环境变量
- 设置 HuggingFace 镜像源（加快下载）
- 配置梯度检查点等优化参数

**配置内容**：
```bash
# ROCm 库路径
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# HuggingFace 镜像源（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# GPU 相关设置
export HSA_OVERRIDE_GFX_VERSION=gfx90a  # 某些 GPU 需要
```

---

### 3. performance_benchmark.py

性能基准测试脚本。

**使用方式**：
```bash
# 基础测试
python performance_benchmark.py

# 测试指定模型
python performance_benchmark.py --model_name Qwen/Qwen2.5-1.5B

# 详细输出
python performance_benchmark.py --verbose
```

**测试内容**：
- GPU 识别和基本信息
- GPU 显存容量和使用情况
- 矩阵乘法性能 (FLOPS)
- 模型加载时间
- 单步推理时间
- 训练吞吐量 (tokens/sec)
- 显存使用效率

**输出示例**：
```
========== AMD GPU 性能基准测试 ==========

GPU 信息:
  GPU 0: AMD Radeon RX 7900 XTX
  显存: 24GB
  架构: RDNA3

性能测试:
  单卡 FP32 理论峰值: 60.8 TFLOPS
  单卡 BF16 理论峰值: 121.6 TFLOPS

模型测试 (Qwen-1.5B):
  模型加载时间: 2.3s
  推理延迟: 45ms/token
  训练吞吐量: 120 tokens/sec (批大小=4, BF16)

建议:
  - 使用 BF16 混合精度训练以提升性能
  - 推荐批大小: 4-8（平衡显存和速度）
  - 使用梯度累积模拟更大批大小
```

---

## 使用建议

### 第一次运行（环境准备）

```bash
# 1. 安装依赖
bash code/install_rocm_deps.sh

# 2. 配置环境
source code/setup_environment.sh

# 3. 运行性能测试
python code/performance_benchmark.py

# 4. 根据输出建议调整参数
```

### 日常使用

```bash
# 每次运行前激活环境
source code/setup_environment.sh

# 运行你的训练脚本
python chapter5/code/pretrain.py
# 或
python chapter6/code/pretrain.py
```

### 故障排除

如果遇到 GPU 识别问题：

```bash
# 检查 ROCm 安装
rocminfo

# 检查 PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 重新运行安装脚本
bash code/install_rocm_deps.sh --force

# 运行性能测试获取诊断信息
python code/performance_benchmark.py --verbose
```

---

## 脚本细节

### install_rocm_deps.sh 脚本框架

```bash
#!/bin/bash
# ROCm 依赖一键安装脚本

set -e  # 错误时退出

echo "开始安装 AMD ROCm 依赖..."

# 1. 系统依赖
echo "安装系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    git \
    git-lfs

# 2. 检查 ROCm 版本
echo "检查 ROCm 版本..."
rocminfo | grep "General"

# 3. 安装 PyTorch ROCm 版本
echo "安装 PyTorch (ROCm 版本)..."
pip install torch torchvision torchaudio --index-url https://repo.amd.com/rocm/whl/rocm-7.2

# 4. 安装 Transformers 和相关库
echo "安装 Transformers 和依赖..."
pip install transformers==4.36.* accelerate datasets evaluate trl

# 5. 安装 DeepSpeed
echo "安装 DeepSpeed..."
pip install deepspeed

# 6. 安装 PEFT
echo "安装 PEFT..."
pip install peft

# 7. 验证安装
echo "验证安装..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers 版本: {transformers.__version__}')"
python -c "import deepspeed; print('DeepSpeed 安装成功')"

echo "所有依赖安装完成！"
```

### setup_environment.sh 脚本框架

```bash
#!/bin/bash
# 环境配置脚本

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "虚拟环境已激活"
fi

# ROCm 环境变量
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export PATH=/opt/rocm/bin:$PATH

# HuggingFace 配置（加快下载）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface

# GPU 架构设置（如需要）
# export HSA_OVERRIDE_GFX_VERSION=gfx90a

# PyTorch 优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "环境配置完成"
echo "ROCm 路径: $LD_LIBRARY_PATH"
echo "HuggingFace 镜像: $HF_ENDPOINT"
```

### performance_benchmark.py 脚本框架

```python
#!/usr/bin/env python3
"""
AMD GPU 性能基准测试脚本

用途：
1. 检查 GPU 和 ROCm 安装情况
2. 测试 GPU 性能和显存
3. 测试模型加载和推理速度
4. 提供训练参数建议
"""

import torch
import argparse
import time
from transformers import AutoModel, AutoTokenizer

def get_gpu_info():
    """获取 GPU 信息"""
    print("=" * 50)
    print("GPU 信息")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA 设备")
        return
    
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

def benchmark_model(model_name, batch_size=1):
    """测试模型性能"""
    print(f"\n{'='*50}")
    print(f"测试模型: {model_name}")
    print(f"{'='*50}")
    
    # 加载模型和分词器
    start = time.time()
    model = AutoModel.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_time = time.time() - start
    print(f"模型加载时间: {load_time:.2f}s")
    
    # 推理测试
    inputs = tokenizer(
        ["What is artificial intelligence?"] * batch_size,
        return_tensors="pt",
        padding=True
    ).to("cuda")
    
    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        inference_time = time.time() - start
    
    print(f"推理时间: {inference_time:.3f}s (批大小={batch_size})")
    print(f"吞吐量: {batch_size / inference_time:.1f} samples/sec")

def main():
    parser = argparse.ArgumentParser(description="AMD GPU 性能基准测试")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # 运行测试
    get_gpu_info()
    benchmark_model(args.model_name, args.batch_size)
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

---

## 文件清单

完成后的目录结构应该是：

```
code/
├── README.md                     # 本文件
├── install_rocm_deps.sh         # 依赖安装脚本
├── setup_environment.sh         # 环境配置脚本  
├── performance_benchmark.py     # 性能测试脚本
└── ... 其他辅助工具
```

---

## 相关资源

- 📚 [ROCm 官方文档](https://rocm.docs.amd.com/)
- 🤗 [Hugging Face 文档](https://huggingface.co/docs)
- ⚡ [DeepSpeed 文档](https://www.deepspeed.ai/)
- 🔧 [PyTorch ROCm 指南](https://pytorch.org/blog/pytorch-rocm/)

---

<div align="center">

有任何问题，欢迎提交 Issue！

</div>
