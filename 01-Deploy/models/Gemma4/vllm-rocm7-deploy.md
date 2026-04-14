## vLLM 零基础大模型部署（Ubuntu 24.04 + ROCm 7+）

本节介绍在 Ubuntu 24.04 + ROCm 7+ 环境下，如何使用 **vLLM** 进行推理，包括：

- 使用官方 ROCm vLLM Docker 镜像快速启动
- 手动编译 ROCm 版 vLLM（含 Triton、FlashAttention 等依赖）

示例模型同样以 **Qwen3-8B Q4_K_M（GGUF 格式）** 为主。

> 前置条件：已完成 ROCm 7.1.0 安装与验证（见 `env-prepare-ubuntu24-rocm7.md`）。

---

## 一、方式一：Docker 方式（推荐）

参考官方 Quickstart 文档：

- https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

> 注意：如果使用 Docker，需要安装 `amdgpu-dkms`：  
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html  
> 相关步骤在前文脚本中已包含，否则需手动安装。

---

### 1. 启动 vLLM 容器

从官方 ROCm vLLM 镜像启动容器：

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

此时容器中 `/app/models` 挂载到宿主机的 `~/models`（存放 GGUF 模型等）。

---

### 2. 容器内启动模型服务

在容器内，启动 Qwen3-8B Q4_K_M GGUF 模型：

```bash
# 在容器内运行
vllm serve /app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf --dtype float16 --max-model-len 4096 --max-num-seqs 32 --tokenizer qwen/Qwen3-8B

# 快速启动（--enforce-eager）：禁用 CUDA graph，
# 启动速度更快，但推理略慢（通常慢 10–20%，对测试 GGUF 通常是值得的）。
vllm serve /app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf \
  --tokenizer qwen/Qwen3-8B \
  --dtype float16 \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-model-len 4096
```

---

### 3. 性能测试脚本（tokens/s）

下面是一个完整的 Bash 脚本示例，用来测量 **Qwen3-8B-Q4_K_M** 的实际推理速度：

```bash
# 1. 准备随机 Prompt
RAND_PROMPT="随机码$(date +%N): 请详细介绍量子计算的未来，要求内容丰富，不要重复。"

# 2. 记录精确开始时间
start=$(date +%s.%N)

# 3. 发起请求并存入变量
response=$(curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
\"model\": \"/app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf\",
\"prompt\": \"$RAND_PROMPT\",
\"max_tokens\": 512,
\"temperature\": 0.8
}")

# 4. 记录结束时间
end=$(date +%s.%N)

# 5. 解析内容
# 提取生成文本原始内容
content=$(echo "$response" | jq -r '.choices[0].text')
# 提取 Token 数
tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
# 计算耗时
duration=$(echo "$end - $start" | bc)

# 6. 打印输出
echo "==================== 原始内容 ===================="
echo "$content"
echo "=================================================="

if (( $(echo "$duration < 0.05" | bc -l) )); then
  echo "检测到异常极速响应 ($duration 秒)，可能命中了缓存。"
else
  tps=$(echo "scale=2; $tokens / $duration" | bc)
  echo "生成 Token 数: $tokens"
  echo "实际总耗时: $duration 秒"
  echo "真实推理速度: $tps tokens/s"
fi
echo "=================================================="
```

测试结果示例（Qwen3-8B Q4_K_M，ctx=4096）：

- **约 30.21 tokens/s**

截图示例：

<img src="./images/media/image11.png" style="width:5.75in;height:3.25in" />
<img src="./images/media/image12.png" style="width:5.75in;height:3.25in" />

---

## 二、方式二：手动构建 vLLM（适合进阶用户）

本节内容较长，适合对 ROCm / Triton / FlashAttention / vLLM 有深入要求的用户。

### 1. 环境与版本要求

参考官方文档：

- https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

关键版本信息（示例）：

- vLLM 0.13.0
- GPU 支持：MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 (gfx1100/1101),
  Radeon RX 9000 (gfx1200/1201), Ryzen AI MAX / AI 300 (gfx1151/1150)
- ROCm 6.3 或以上
  - MI350 需要 ROCm 7.0+
  - Ryzen AI MAX / AI 300 需要 ROCm 7.0.2+

---

### 2. 使用 uv 构建 Python 虚拟环境

参考：`https://www.runoob.com/python3/uv-tutorial.html`

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建并激活虚拟环境
uv venv --python 3.12 --seed
source .venv/bin/activate
```

---

### 3. 安装 ROCm7+ 支持的 PyTorch

```bash
# 安装 PyTorch
uv pip uninstall torch
uv pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

---

### 4. 为 ROCm 安装 Triton

仓库：`https://github.com/ROCm/triton.git`

```bash
uv pip install ninja cmake wheel pybind11
uv pip uninstall triton

git clone https://github.com/ROCm/triton.git
cd triton
# git checkout $TRITON_BRANCH
git checkout f9e5bf54

# 启动 16 核并行编译，并显示实时输出
# --no-build-isolation 确保使用你当前环境已有的 ninja, cmake
if [ ! -f setup.py ]; then cd python; fi
MAX_JOBS=16 uv pip install --no-build-isolation -e .
cd ..
```

> 注意：Triton 依赖体积较大，`Preparing packages...` 阶段可能下载 10+GB，需科学上网，下载过程约数小时，编译约数分钟。

---

### 5. 构建 FlashAttention（ROCm 版）

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout origin/main_perf
git submodule update --init

export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_USE_HIP_DSA=1
python setup.py bdist_wheel --dist-dir=dist
uv pip install dist/*.whl
```

构建完成后，可进入 `benchmarks` 目录跑测试：

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_USE_HIP_DSA=1

cd benchmarks/
python benchmark_flash_attention.py
```

---

### 6. 构建 vLLM（ROCm7.0+）

#### 6.1 安装 AMD SMI

```bash
# 更新 uv 自身（可选）
uv self update

# 安装 AMD SMI（指向本地路径）
# uv pip install /opt/rocm/share/amd_smi

# 若权限有问题，可先拷贝目录：
cp -r /opt/rocm/share/amd_smi ./amdsmi_src
cd ./amdsmi_src
uv pip install .
cd ..
```

#### 6.2 安装编译依赖并拉取 vLLM 源码

```bash
uv pip install --upgrade \
  numba \
  scipy \
  "huggingface-hub[cli,hf_transfer]" \
  setuptools_scm

git clone https://github.com/vllm-project/vllm.git
cd vllm

# 安装 vLLM ROCm 专用依赖
uv pip install -r requirements/rocm.txt
uv pip install numpy setuptools wheel
```

#### 6.3 设置 ROCm 架构、编译安装 vLLM

```bash
export VLLM_TARGET_DEVICE="rocm"
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCM_HOME="/opt/rocm"

MAX_JOBS=16 uv pip install -e . --no-build-isolation
```

---

### 7. 在虚拟环境中启动 vLLM 模型服务

```bash
# 启用 FlashAttention
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_USE_HIP_DSA=1

# 启动服务
vllm serve ~/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf --dtype float16 --max-model-len 4096 --max-num-seqs 32 --tokenizer qwen/Qwen3-8B

# 使用 --enforce-eager 提升启动速度（会略微降低推理速度）
vllm serve ~/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf \
  --tokenizer qwen/Qwen3-8B \
  --dtype float16 \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-model-len 4096
```

---

### 8. 完整性能测试脚本（自动获取模型 ID）

```bash
# 1. 自动获取模型 ID（防止 404）
MODEL_ID=$(curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[0].id')
echo "探测到模型 ID: $MODEL_ID"

# 2. 准备随机 Prompt
RAND_PROMPT="随机码$(date +%N): 请详细介绍量子计算的未来，不少于500字。"

# 3. 记录精确开始时间
start=$(date +%s.%N)

# 4. 发起请求
response=$(curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
\"model\": \"$MODEL_ID\",
\"prompt\": \"$RAND_PROMPT\",
\"max_tokens\": 512,
\"temperature\": 0.8
}")

# 5. 记录结束时间
end=$(date +%s.%N)

# 6. 解析内容
content=$(echo "$response" | jq -r '.choices[0].text // "错误: 未获取到文本内容，请检查输出"')
tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
duration=$(echo "$end - $start" | bc)

# 7. 打印输出
echo "==================== 原始内容 ===================="
echo "$content"
echo "=================================================="

if (( $(echo "$duration < 0.05" | bc -l) )); then
  echo "响应过快 ($duration 秒)，可能是 404 报错或缓存。"
  echo "完整响应体: $response"
else
  tps=$(echo "scale=2; $tokens / $duration" | bc)
  echo "生成 Token 数: $tokens"
  echo "实际总耗时: $duration 秒"
  echo "真实推理速度: $tps tokens/s"
fi
echo "=================================================="
```

示例效果截图：

<img src="./images/media/image13.png" style="width:5.75in;height:3.25in" />
<img src="./images/media/image14.png" style="width:5.75in;height:3.25in" />
<img src="./images/media/image15.png" style="width:5.75in;height:3.25in" />


