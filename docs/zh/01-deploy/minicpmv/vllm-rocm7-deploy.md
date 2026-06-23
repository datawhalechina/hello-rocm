## vLLM 部署 MiniCPM-V（Ubuntu 24.04 + ROCm 7+）

本节介绍如何在 AMD GPU（Ubuntu 24.04 + ROCm 7+）上，使用 **vLLM** 服务**多模态**模型
**MiniCPM-V 4.6**，包括：

- 使用官方 ROCm vLLM Docker 镜像快速启动
- 从源码手动编译 ROCm 版 vLLM（适用于没有 Docker 的环境）

整体结构与 `qwen3/vllm-rocm7-deploy.md` 示例一致。MiniCPM-V 的特殊之处在于：它是**视觉语言模型**，
因此服务时需加上 `--trust-remote-code`（多图 / 视频还需 `--limit-mm-per-prompt`），并通过
OpenAI `chat/completions` 接口传入图像。

示例模型为 **MiniCPM-V 4.6**（1.3B 视觉语言模型；SigLIP2 视觉编码器 + Qwen3.5 语言主干）。

> 前置条件：已完成 ROCm 7+ 安装与验证（见 `env-prepare-ubuntu24-rocm7.md`）。
> 参考机器：**AMD Ryzen AI MAX+ 395（Radeon 8060S，gfx1151），ROCm 7.13**。

---

### 模型 / 版本兼容性（请先阅读）

MiniCPM-V 4.6 在 vLLM 中以架构 **`MiniCPMV4_6ForConditionalGeneration`** 被**原生支持**。
两条硬性要求：

- **vLLM ≥ 0.22.0** —— 更早的版本不含 `minicpmv4_6` 模型模块。
- **transformers ≥ 5.7** —— 4.6 架构作为独立类合入 transformers（本仓库环境为 transformers 5.12.1）。

> ROCm 源码编译的重要版本提示：vLLM 0.22.0+ 在其构建元数据中固定 **`torch == 2.11.0`**。
> 如果你的 ROCm PyTorch 较旧（例如 `torch 2.9.x+rocm`），请在**独立虚拟环境**中、使用匹配的
> torch（ROCm 版 `torch 2.10/2.11` wheel）编译 vLLM，以免破坏已有的推理 / 微调环境。
> Docker 方式（见下）镜像内自带匹配的 torch + vLLM，可完全规避该问题。

---

## 方式一：Docker 方式（推荐）

参考官方 Quickstart 文档：

- https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

> 注意：若使用 Docker，需要安装 `amdgpu-dkms`：
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

### 1. 启动 vLLM 容器

```bash
sudo docker pull rocm/vllm-dev:nightly # 获取最新镜像

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

> 提示：确认镜像内 vLLM 版本足够新：在容器内执行
> `python -c "import vllm; print(vllm.__version__)"`，应 ≥ 0.22.0。若不满足，请拉取更新的 tag。

### 2. 下载模型（HF 格式，**不是** GGUF）

vLLM 服务的是完整的 Hugging Face 检查点（safetensors），而非 llama.cpp 的 GGUF：

```bash
# 在宿主机（或容器内）下载到 ~/models
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openbmb/MiniCPM-V-4_6 \
  --local-dir ~/models/MiniCPM-V-4_6
```

### 3. 容器内启动模型服务

参考官方 MiniCPM-V cookbook 的启动命令，适配如下：

```bash
# 容器内运行
vllm serve /app/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8

# 快速启动（--enforce-eager）：关闭 HIP graph 捕获，
# 启动更快、推理略慢，适合首次启动验证。
vllm serve /app/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 8192 \
  --max-num-seqs 8
```

> - gfx1151 上建议使用 `--dtype bfloat16`（该系列模型在 fp16 下可能溢出为 NaN；Strix Halo 上 bf16 稳定）。
> - 4.6 最高支持 256K 上下文；先用较小值（`--max-model-len 8192`），随显存富余再调大。
> - 单请求多图 / 视频时，添加如 `--limit-mm-per-prompt '{"image": 4, "video": 1}'`。
> - `MiniCPM-V-4_6-Thinking` 检查点默认注入 `<think>` 块；若要直接回答，请服务普通
>   `MiniCPM-V-4_6` 检查点，或传入 `--chat-template-kwargs '{"enable_thinking": false}'`。

### 4. 测试接口（文本 + 图像，并计算 tokens/s）

先自动检测 model id，再调用 OpenAI 兼容接口（vLLM 默认端口 8000）：

```bash
MODEL_ID=$(curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[0].id')
echo "Detected model ID: $MODEL_ID"
```

**文本补全 + tokens/s**（墙钟法，与 Qwen3 示例一致）：

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

**多模态对话**（通过 base64 data URL 在 `chat/completions` 传图）：

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

## 方式二：从源码手动编译 vLLM（无 Docker / 进阶用户）

当没有 Docker 时使用。流程与 Qwen3 源码编译一致；MiniCPM-V 的差异点已单独标出。

### 1. 环境与版本要求

- vLLM **≥ 0.22.0**（MiniCPM-V 4.6 所需）；本指南使用发布 tag，例如 `v0.22.0`。
- GPU 支持包含 Ryzen AI MAX / AI 300（**gfx1151**/1150）；ROCm **7.0.2+**。
- 与 vLLM 构建固定版本匹配的 ROCm PyTorch（`torch == 2.11.0`）。请在独立 venv 中编译。

### 2. 用 uv 建立独立 Python venv

在独立 venv 中编译可保护已有的 ROCm torch（如 llama.cpp / 微调环境）：

```bash
uv venv --python 3.12 --seed ~/vllm-venv
source ~/vllm-venv/bin/activate
```

### 3. 安装与 vLLM 固定版本匹配的 ROCm PyTorch

```bash
# 与 vLLM 构建固定版本匹配的 ROCm torch wheel（示例为 2.11 nightly；选择最接近的可用版本）
uv pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0 \
  "torch==2.11.0.dev*" torchvision
```

> 若你的 ROCm 没有 `torch 2.11` wheel，可使用最接近的版本（如 `2.10+rocm7.0`），并以
> `--no-build-isolation` 编译，使 vLLM 针对已安装的 torch 构建。

### 4. ROCm 版 Triton

ROCm PyTorch wheel 通常已自带匹配的 Triton（`import triton` 验证）。仅当 `import triton` 失败时
才需从源码编译 Triton —— 参见 Qwen3 指南的 Triton 章节。

### 5.（可选）ROCm 版 FlashAttention

vLLM 在 gfx1151 上无需自定义 FlashAttention 即可运行（会回退到受支持的 attention 后端）。
如仍要编译，参见 Qwen3 指南的 FlashAttention 章节。

### 6. 编译 vLLM（ROCm，gfx1151）

```bash
# AMD SMI（来自本地 ROCm 安装）
cp -r /opt/rocm/share/amd_smi ./amdsmi_src && (cd ./amdsmi_src && uv pip install .)

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.22.0          # 任意 >= 0.22.0 且包含 minicpmv4_6.py 的 tag

uv pip install -r requirements/rocm.txt
uv pip install numba scipy "huggingface-hub[cli,hf_transfer]" setuptools_scm setuptools wheel ninja cmake

export VLLM_TARGET_DEVICE="rocm"
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCM_HOME="/opt/rocm"

MAX_JOBS=16 uv pip install -e . --no-build-isolation
```

> 该步骤会为 gfx1151 编译 vLLM 的 HIP 算子（数百个源文件），耗时较长。
> 编译完成后验证架构已注册：
> ```bash
> python -c "from vllm import ModelRegistry; print('MiniCPMV4_6ForConditionalGeneration' in ModelRegistry.get_supported_archs())"
> ```

### 7. 服务与测试

```bash
vllm serve ~/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 8192 \
  --max-num-seqs 8
```

随后使用与方式一第 4 节相同的「文本 + 多模态」接口测试。

---

### 说明 / 状态

- **模型侧兼容性已确认**：vLLM 暴露了 `MiniCPMV4_6ForConditionalGeneration`
  （vLLM ≥ 0.22.0，transformers ≥ 5.7），且 gfx1151 是 ROCm 7.0.2+ 上受支持的 vLLM 目标。
- **在消费级 ROCm 上，编译是最重的环节**：没有 Docker 时，vLLM 需为 gfx1151 从源码编译，
  而含 4.6 支持的版本均固定 `torch 2.11`。请使用独立 venv，以免破坏可正常工作的
  `torch 2.9.x+rocm` 环境（例如本仓库中的 llama.cpp / 微调环境）。
- 同一模型的 llama.cpp 部署（更轻量、预构建二进制）见
  `minicpmv/llamacpp-rocm7-deploy.md`。
