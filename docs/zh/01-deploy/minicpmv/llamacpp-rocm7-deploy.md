## llama.cpp 部署 MiniCPM-V（Ubuntu 24.04 + ROCm 7+）

本节介绍如何在 AMD GPU（Ubuntu 24.04 + ROCm 7+）上，使用 **llama.cpp** 对**多模态**模型
**MiniCPM-V 4.6** 进行本地「图像 + 文本」推理。

整体流程与 `qwen3/llamacpp-rocm7-deploy.md` 的预构建方式一致，但有一个关键区别：
**MiniCPM-V 是视觉语言模型**，除 GGUF 权重外，还需加载一个 **`mmproj` 多模态投影文件**，
并使用带 `--mmproj` 的 `llama-mtmd-cli` / `llama-server`。

示例模型为 **MiniCPM-V 4.6 Q4_K_M（GGUF）** —— 一个 1.3B 的视觉语言模型，非常适合单卡 / 端侧部署。

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

确认 llama.cpp 通过 ROCm 识别到 GPU：

```bash
cd ~/minicpmv-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
./llama-cli --list-devices
# Available devices:
#   ROCm0: Radeon 8060S Graphics (65536 MiB, ... free)
```

---

#### 3. 进入 llama 后端目录并设置权限 / 环境变量

```bash
cd ~/minicpmv-rocm/llama-bin
chmod +x llama-cli llama-server llama-mtmd-cli
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
```

> 注意：Lemonade 构建会把自带的 ROCm 运行库放在可执行文件旁边，因此除 `/opt/rocm/lib` 外，
> 还需把后端目录本身（上面的 `$PWD`）加入 `LD_LIBRARY_PATH`。

---

#### 4. 下载 MiniCPM-V 4.6 GGUF + mmproj 投影文件

llama.cpp 使用 **GGUF 模型格式**。多模态模型需要**两个**文件：

- 量化后的 LLM 权重（`*Q4_K_M*.gguf`）
- 视觉投影文件（`mmproj-*.gguf`）

使用国内 Hugging Face 镜像 `https://hf-mirror.com/`：

```bash
mkdir -p ~/models/MiniCPM-V-4_6-gguf && cd ~/models/MiniCPM-V-4_6-gguf
export HF_ENDPOINT=https://hf-mirror.com

# 模型权重（Q4_K_M，约 505 MB）与多模态投影文件（约 1.1 GB）
for f in MiniCPM-V-4_6-Q4_K_M.gguf mmproj-model-f16.gguf; do
  curl -L --fail -o "$f" \
    "https://hf-mirror.com/openbmb/MiniCPM-V-4_6-gguf/resolve/main/$f"
done
```

> 提示：也可使用 `hfd.sh` + `aria2`（同 Qwen3 示例）进行更快、可断点续传的多线程下载。
> GGUF 仓库 / 文件名可能随上游更新而变化，使用前请在 Hugging Face 搜索 `MiniCPM-V-4_6-gguf`
> 选择最新的可信仓库。

---

#### 5. CLI 多模态测试（`llama-mtmd-cli`）

这是验证视觉链路最快的方式。`llama-mtmd-cli` 是多模态 CLI；用 `--mmproj` 传入投影文件，
用 `--image` 传入图片：

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

预期：在 `ROCm0` 上生成对图片内容连贯的自然语言描述。

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

#### 7. 测试接口（文本 + 图像，并计算 tokens/s）

**文本补全**（与 Qwen3 示例同样的公式
`completion_tokens / (timings.predicted_ms / 1000)`）：

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

**多模态对话**（通过 base64 data URL 在 OpenAI `chat/completions` 接口传图）：

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

在 **Radeon 8060S（gfx1151），ROCm 7.13，ctx=4096** 上的参考结果：

- 文本解码：**约 190 tokens/s**（MiniCPM-V 4.6 仅 1.3B，因此比 8B 模型快很多）
- 多模态对话：**约 190 tokens/s** 解码（首轮另含图像编码时间）
- **tokens/s 取决于你的实际硬件。**

---

### 二、方式二：Docker 方式（官方 ROCm llama.cpp 镜像）

如果你更习惯使用 Docker，可参考官方文档并在容器内从源码编译，步骤与 Qwen3 / Gemma4 示例完全
一致 —— MiniCPM-V 唯一的区别是启动时需同时传入 `-m <gguf>` 与 `--mmproj <mmproj-gguf>`：

- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/llama-cpp-install.html

```bash
./build/bin/llama-mtmd-cli \
  -m /data/MiniCPM-V-4_6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf \
  --mmproj /data/MiniCPM-V-4_6-gguf/mmproj-model-f16.gguf \
  -ngl 99 -c 4096 \
  --image /data/image.jpeg \
  -p "这张图片里有什么？"
```

> 注意：为 ROCm 从源码编译 llama.cpp（`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`）的完整步骤见
> `qwen3/llamacpp-rocm7-deploy.md`；MiniCPM-V 只需额外的多模态支持与 `mmproj` 文件。
