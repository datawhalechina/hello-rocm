## llama.cpp 部署 MiniCPM-V（Ubuntu 24.04 + ROCm 7+）

### 模型简介

[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) 是由面壁智能（ModelBest）与清华大学自然语言处理实验室（OpenBMB）联合开发的端侧多模态大模型系列。MiniCPM-V 4.6 是该系列最新版本，仅 1.3B 参数（SigLIP2 视觉编码器 + Qwen3.5 语言主干），支持图像理解和文本对话。

- 模型仓库：[openbmb/MiniCPM-V-4_6](https://huggingface.co/openbmb/MiniCPM-V-4_6)
- GGUF 量化：[openbmb/MiniCPM-V-4_6-gguf](https://huggingface.co/openbmb/MiniCPM-V-4_6-gguf)

本节使用 **llama.cpp** 部署 **MiniCPM-V 4.6 Q4_K_M（GGUF）**，包括：

- 使用预构建的可执行文件（推荐）
- 使用 Docker + 官方 ROCm 镜像自行编译

与纯文本的 MiniCPM 不同，MiniCPM-V 是视觉语言模型，除 GGUF 权重外还需加载 **`mmproj` 多模态投影文件**，并使用 `llama-mtmd-cli` / `llama-server --mmproj` 进行推理。

> 前置条件：已完成 ROCm 7+ 系统安装与验证（见 `env-prepare-ubuntu24-rocm7.md`）。
> 已在 **AMD Ryzen AI MAX+ 395（Radeon 8060S，gfx1151），ROCm 7.13** 上验证。

---

### 一、方式一（推荐）：预构建的可执行文件

#### 1. 下载预构建版本

使用 Lemonade 提供的预构建版本，其中：

- **370** 对应 **gfx1150** 架构
- **395** 对应 **gfx1151** 架构

相关链接：

- https://github.com/lemonade-sdk/llamacpp-rocm
- https://github.com/lemonade-sdk/llamacpp-rocm/releases

```bash
mkdir -p ~/minicpmv-rocm && cd ~/minicpmv-rocm
# 选择与你架构匹配的文件（此处为 gfx1151）
curl -L -o llama-rocm-gfx1151.zip \
  https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1292/llama-b1292-ubuntu-rocm-gfx1151-x64.zip
mkdir -p llama-bin && unzip -q llama-rocm-gfx1151.zip -d llama-bin
```

---

#### 2. 确认 ROCm 7+ 安装（必须为系统版 ROCm）

```bash
amd-smi
```

应能看到 GPU 型号、驱动、ROCm 版本，例如：

```
MARKET_NAME: Radeon 8060S Graphics
TARGET_GRAPHICS_VERSION: gfx1151
ROCm version: 7.13.0
```

确认 llama.cpp 能识别到 GPU：

```bash
cd ~/minicpmv-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
./llama-cli --list-devices
# Available devices:
#   ROCm0: Radeon 8060S Graphics (65536 MiB, ... free)
```

---

#### 3. 设置权限和环境变量

```bash
cd ~/minicpmv-rocm/llama-bin
chmod +x llama-cli llama-server llama-mtmd-cli
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
```

> Lemonade 构建会把自带的 ROCm 运行库放在可执行文件旁边，因此除 `/opt/rocm/lib` 外，还需把 `$PWD` 加入 `LD_LIBRARY_PATH`。

---

#### 4. 下载 MiniCPM-V 4.6 GGUF + mmproj 投影文件

多模态模型需要**两个**文件：

- 量化后的 LLM 权重（`*Q4_K_M*.gguf`）
- 视觉投影文件（`mmproj-*.gguf`）

使用国内 Hugging Face 镜像下载：

```bash
mkdir -p ~/models/MiniCPM-V-4_6-gguf && cd ~/models/MiniCPM-V-4_6-gguf
export HF_ENDPOINT=https://hf-mirror.com

# 模型权重（Q4_K_M，约 505 MB）与多模态投影文件（约 1.1 GB）
for f in MiniCPM-V-4_6-Q4_K_M.gguf mmproj-model-f16.gguf; do
  curl -L --fail -o "$f" \
    "https://hf-mirror.com/openbmb/MiniCPM-V-4_6-gguf/resolve/main/$f"
done
```

> 也可使用 `hfd.sh` + `aria2` 进行断点续传下载。GGUF 文件名可能随上游更新而变化，请在 Hugging Face 搜索确认最新版本。

---

#### 5. CLI 多模态测试（`llama-mtmd-cli`）

`llama-mtmd-cli` 是多模态 CLI，用 `--mmproj` 传入投影文件，用 `--image` 传入图片：

```bash
cd ~/minicpmv-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH

./llama-mtmd-cli \
  -m ~/models/MiniCPM-V-4_6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf \
  --mmproj ~/models/MiniCPM-V-4_6-gguf/mmproj-model-f16.gguf \
  -ngl 99 -c 4096 --temp 0.2 \
  --image /path/to/image.jpeg \
  -p "请详细描述这张图片。"
```

---

#### 6. 启动 llama-server（OpenAI 兼容接口）

```bash
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
cd ~/minicpmv-rocm/llama-bin

./llama-server \
  -m ~/models/MiniCPM-V-4_6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf \
  --mmproj ~/models/MiniCPM-V-4_6-gguf/mmproj-model-f16.gguf \
  -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

---

#### 7. 测试接口

**文本补全：**

```bash
curl -s -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "MiniCPM-V-4_6",
  "prompt": "用一句话解释大语言模型",
  "max_tokens": 128
}' | jq -r '
.choices[0].text as $txt |
(.usage.completion_tokens / (.timings.predicted_ms / 1000)) as $tps |
"生成文本:\n\($txt)\n\ntokens/s: \($tps|tostring)"
'
```

**多模态对话**（通过 base64 传图）：

```bash
IMG_B64=$(base64 -w0 /path/to/image.jpeg)
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "MiniCPM-V-4_6",
  "max_tokens": 128,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "用一句话描述这张图片"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG_B64"'"}}
  ]}]
}' | jq -r '.choices[0].message.content'
```

参考性能（Radeon 8060S，gfx1151，ROCm 7.13，ctx=4096）：文本和多模态解码均约 **190 tokens/s**（首轮多模态请求另含图像编码时间）。实际速度取决于硬件。

---

### 二、方式二：Docker 方式（官方 ROCm llama.cpp 镜像）

> 若使用 Docker，需要安装 `amdgpu-dkms`：
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

#### 1. 下载容器镜像

```bash
export MODEL_PATH='~/models'

sudo docker run -it \
  --name=$(whoami)_llamacpp_minicpmv \
  --privileged --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 16G \
  -v $MODEL_PATH:/data \
  rocm/dev-ubuntu-24.04:7.0-complete
```

---

#### 2. 容器内准备工作区

```bash
apt-get update && apt-get install -y nano libcurl4-openssl-dev cmake git
mkdir -p /workspace && cd /workspace
```

---

#### 3. 克隆 llama.cpp 仓库

```bash
git clone https://github.com/ROCm/llama.cpp
cd llama.cpp
```

---

#### 4. 设定 ROCm 架构

```bash
# 以 AI MAX 395 (gfx1151) 为例
export LLAMACPP_ROCM_ARCH=gfx1151
```

---

#### 5. 编译 llama.cpp

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=$LLAMACPP_ROCM_ARCH \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=ON && \
cmake --build build --config Release -j$(nproc)
```

---

#### 6. 运行多模态测试

MiniCPM-V 需要同时传入模型权重和 mmproj 投影文件：

```bash
./build/bin/llama-mtmd-cli \
  -m /data/MiniCPM-V-4_6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf \
  --mmproj /data/MiniCPM-V-4_6-gguf/mmproj-model-f16.gguf \
  -ngl 99 -c 4096 \
  --image /data/image.jpeg \
  -p "这张图片里有什么？"
```

---

### 效果截图

<div align='center'>
    <img src="../../../public/images/01-deploy/minicpmv/minicpmv-example.png" alt="MiniCPM-V 4.6 llama.cpp 多模态示例" width="90%">
</div>
