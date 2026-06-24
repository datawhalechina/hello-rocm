## vLLM 部署 MiniCPM-V（Ubuntu 24.04 + ROCm 7+）

### 模型简介

[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) 是由面壁智能（ModelBest）与清华大学自然语言处理实验室（OpenBMB）联合开发的端侧多模态大模型系列。MiniCPM-V 4.6 仅 1.3B 参数（SigLIP2 视觉编码器 + Qwen3.5 语言主干），支持图像理解和文本对话。

- 模型仓库：[openbmb/MiniCPM-V-4_6](https://huggingface.co/openbmb/MiniCPM-V-4_6)

本节使用 **vLLM** 部署 MiniCPM-V 4.6，包括：

- 使用官方 ROCm vLLM Docker 镜像快速启动
- 从源码手动编译 ROCm 版 vLLM（适用于没有 Docker 的环境）

> 前置条件：已完成 ROCm 7+ 安装与验证（见 `env-prepare-ubuntu24-rocm7.md`）。
> 参考机器：**AMD Ryzen AI MAX+ 395（Radeon 8060S，gfx1151），ROCm 7.13**。

---

### 版本要求

MiniCPM-V 4.6 在 vLLM 中以架构 `MiniCPMV4_6ForConditionalGeneration` 被原生支持，需要满足：

- **vLLM ≥ 0.22.0**
- **transformers ≥ 5.7**

> vLLM 0.22.0+ 要求 `torch == 2.11.0`。如果已有较旧的 ROCm PyTorch（如 `torch 2.9.x+rocm`），请在**独立虚拟环境**中编译 vLLM，避免影响已有环境。Docker 方式不受此限制。

---

## 方式一：Docker 方式（推荐）

参考官方文档：https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

> 若使用 Docker，需要安装 `amdgpu-dkms`：
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

### 1. 启动 vLLM 容器

```bash
sudo docker pull rocm/vllm-dev:nightly

sudo docker run -it --rm \
  --network=host \
  --cpus="16" \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/models:/app/models \
  -e HF_HOME="/app/models" \
  rocm/vllm-dev:nightly
```

容器内 `/app/models` 挂载到宿主机的 `~/models`。

> 进入容器后确认版本：`python -c "import vllm; print(vllm.__version__)"` 应 ≥ 0.22.0。

### 2. 下载模型（HF 格式，不是 GGUF）

vLLM 需要完整的 Hugging Face 检查点（safetensors），不是 llama.cpp 的 GGUF：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openbmb/MiniCPM-V-4_6 \
  --local-dir ~/models/MiniCPM-V-4_6
```

### 3. 启动模型服务

```bash
vllm serve /app/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8
```

如需快速验证，可加 `--enforce-eager` 跳过 HIP graph 捕获（启动更快，推理略慢）。

> - gfx1151 上建议使用 `--dtype bfloat16`（fp16 可能出现 NaN）。
> - 单请求多图 / 视频时，添加 `--limit-mm-per-prompt '{"image": 4, "video": 1}'`。
> - 如需思考模式，使用 `MiniCPM-V-4_6-Thinking` 检查点；不需要则用普通 `MiniCPM-V-4_6`。

### 4. 测试接口

先获取 model id：

```bash
MODEL_ID=$(curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[0].id')
echo "Model ID: $MODEL_ID"
```

**文本补全：**

```bash
start=$(date +%s.%N)
response=$(curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_ID\", \"prompt\": \"用一句话解释大语言模型\", \"max_tokens\": 128}")
end=$(date +%s.%N)

tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
duration=$(echo "$end - $start" | bc)
echo "$response" | jq -r '.choices[0].text'
echo "tokens: $tokens | time: ${duration}s | tokens/s: $(echo "scale=2; $tokens / $duration" | bc)"
```

**多模态对话**（通过 base64 传图）：

```bash
IMG_B64=$(base64 -w0 /app/models/image.jpeg)
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
  \"model\": \"$MODEL_ID\",
  \"max_tokens\": 128,
  \"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"text\", \"text\": \"用一句话描述这张图片\"},
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMG_B64\"}}
  ]}]
}" | jq -r '.choices[0].message.content'
```

---

## 方式二：从源码编译 vLLM（无 Docker）

当没有 Docker 时使用。

### 1. 环境要求

- vLLM **≥ 0.22.0**
- ROCm **7.0.2+**，GPU 支持 gfx1151/1150
- `torch == 2.11.0`（ROCm 版），在独立 venv 中编译

### 2. 创建独立 Python venv

```bash
uv venv --python 3.12 --seed ~/vllm-venv
source ~/vllm-venv/bin/activate
```

### 3. 安装 ROCm PyTorch

```bash
uv pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0 \
  "torch==2.11.0.dev*" torchvision
```

> 若没有 `torch 2.11` wheel，使用最接近的版本并以 `--no-build-isolation` 编译。

### 4. 安装 Triton

ROCm PyTorch wheel 通常已自带 Triton。验证：

```bash
python -c "import triton; print(triton.__version__)"
```

如果 `import triton` 失败，需从源码编译：

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -e python
```

### 5.（可选）FlashAttention

vLLM 在 gfx1151 上无需自定义 FlashAttention 即可运行（会回退到受支持的 attention 后端）。如需编译：

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
pip install -e .
```

### 6. 编译 vLLM

```bash
# AMD SMI
cp -r /opt/rocm/share/amd_smi ./amdsmi_src && (cd ./amdsmi_src && uv pip install .)

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.22.0

uv pip install -r requirements/rocm.txt
uv pip install numba scipy "huggingface-hub[cli,hf_transfer]" setuptools_scm setuptools wheel ninja cmake

export VLLM_TARGET_DEVICE="rocm"
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCM_HOME="/opt/rocm"

MAX_JOBS=16 uv pip install -e . --no-build-isolation
```

编译耗时较长。完成后验证：

```bash
python -c "from vllm import ModelRegistry; print('MiniCPMV4_6ForConditionalGeneration' in ModelRegistry.get_supported_archs())"
```

### 7. 启动服务

```bash
vllm serve ~/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 8192 \
  --max-num-seqs 8
```

然后使用与方式一第 4 节相同的接口测试。

---

### 补充说明

- 没有 Docker 时 vLLM 需为 gfx1151 从源码编译，且依赖 `torch 2.11`。务必使用独立 venv，避免影响已有的推理/微调环境。
- 同一模型的 llama.cpp 部署（更轻量、有预构建二进制）见 `minicpmv/llamacpp-rocm7-deploy.md`。
