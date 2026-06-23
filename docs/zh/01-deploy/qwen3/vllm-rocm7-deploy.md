## vLLM 零基础大模型部署（Ubuntu 24.04 + ROCm 7+）

本节介绍在 Ubuntu 24.04 + ROCm 7+ 环境下，使用 **vLLM** 部署和调用 **Qwen3** 模型。

当前 vLLM 官方文档推荐优先使用官方 ROCm Docker 镜像：

```text
vllm/vllm-openai-rocm:latest
```

旧的 `rocm/vllm`、`rocm/vllm-dev` 镜像已不再作为 vLLM 官方新文档的首选路径。若只想快速跑通 Qwen3 推理服务，Docker 方式最简单。

> 前置条件：已完成 [Ubuntu 24.04 + ROCm 7 环境准备](./env-prepare-ubuntu24-rocm7.md)。

---

## 一、方式一：官方 vLLM ROCm Docker 镜像（推荐）

### 1. 路线 A：ROCm 7.13 官方验证镜像（gfx1151）

ROCm 7.13 官方文档提供了针对 `gfx1151` 的 vLLM 0.19.1 Docker 镜像：

```bash
docker pull rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

> 注意：该镜像内置 PyTorch 2.10.0 + vLLM 0.19.1；PyTorch 2.11.0 属于 ROCm 7.13 pip 安装路线，不要把两条路线的版本混写。

启动容器并进入 shell：

```bash
docker run -it --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --network=host \
  --ipc=host \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ~/models:/app/models \
  -e HF_HOME="/app/models" \
  rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  bash
```

容器内可以继续执行 `vllm serve`。

---

### 2. 路线 B：vLLM upstream 通用镜像

vLLM 官方 upstream 文档推荐使用 `vllm/vllm-openai-rocm` 镜像。该路线适合希望跟随 vLLM 官方最新发布的用户。

```bash
docker pull vllm/vllm-openai-rocm:latest
```

---

### 3. 启动 Qwen3 服务

快速验证可使用较小的 Qwen3 模型：

```bash
docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model Qwen/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --max-model-len 4096
```

如果需要测试 Qwen3-8B，可替换模型：

```bash
docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --max-model-len 4096
```

服务启动后检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

---

### 4. 使用本地模型目录启动

如果已经提前下载好模型，可挂载本地目录：

```bash
docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/models:/app/models \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model /app/models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --max-model-len 4096
```

---

## 二、OpenAI-compatible API 调用

### 1. curl 调用

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "用一句话解释什么是大语言模型。"}
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 256
  }'
```

### 2. Python 调用

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "用一句话介绍 ROCm。"}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

---

## 三、性能测试脚本（tokens/s）

```bash
RAND_PROMPT="随机码$(date +%N): 请详细介绍 ROCm 的用途，要求内容丰富，不要重复。"
start=$(date +%s.%N)

response=$(curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen3-0.6B\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$RAND_PROMPT\"}],
    \"max_tokens\": 512,
    \"temperature\": 0.7
  }")

end=$(date +%s.%N)
content=$(echo "$response" | jq -r '.choices[0].message.content')
tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
duration=$(echo "$end - $start" | bc)

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
```

> [截图待补充：使用官方 vLLM ROCm Docker 镜像运行 Qwen3 的测试输出]

---

## 四、方式二：ROCm wheel 安装（可选）

如果不使用 Docker，也可以安装 vLLM ROCm wheel。该方式对 Python / ROCm / glibc 版本要求更严格。

官方当前要求重点：

- Python 3.12
- ROCm 7.0 或 ROCm 7.2.1 对应 wheel
- glibc >= 2.35

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

安装完成后启动服务：

```bash
vllm serve Qwen/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --max-model-len 4096
```

> 注意：如果 Python 版本不匹配，安装器可能回退到 CUDA wheel，随后在 AMD GPU 上出现 `libcudart.so` 相关错误。

---

## 五、方式三：源码编译（进阶）

源码编译适合需要修改 vLLM、调试算子或适配特殊硬件的场景。普通部署不建议优先选择该方式。

官方 ROCm 构建文档：

- https://docs.vllm.ai/en/stable/getting_started/installation/gpu/

查询 GPU 架构：

```bash
rocminfo | grep gfx
```

设置架构示例：

```bash
export PYTORCH_ROCM_ARCH="gfx1151"
```

具体依赖和构建步骤请以 vLLM 官方文档为准。

---

## 六、常见问题

### 1. 容器无法访问 GPU

确认 Docker 命令包含 `--device /dev/kfd --device /dev/dri`，并确认宿主机 `rocminfo` 能看到 GPU。

### 2. 显存不足

降低 `--max-model-len`，例如从 4096 降到 2048；或改用更小模型进行连通性验证。

### 3. 应该继续使用 `rocm/vllm-dev:nightly` 吗？

新文档优先使用 vLLM 官方镜像 `vllm/vllm-openai-rocm:latest`。旧 AMD 镜像只适合特定版本验证，不作为本教程主路径。
