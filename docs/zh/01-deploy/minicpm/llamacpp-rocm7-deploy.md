## llama.cpp 部署 MiniCPM（Ubuntu 24.04 + ROCm 7+）

本节介绍如何在 AMD GPU（Ubuntu 24.04 + ROCm 7+）上，使用 **llama.cpp** 对**文本**大模型
**MiniCPM5-1B** 进行本地推理。

整体流程与 `qwen3/llamacpp-rocm7-deploy.md` 的预构建方式一致。与多模态的 `minicpmv/` 指南不同，
MiniCPM 是**纯文本**模型，因此**无需 `mmproj` 投影文件** —— 只加载单个 GGUF，使用标准的
`llama-cli` / `llama-server` 即可。

示例模型为 **MiniCPM5-1B Q4_K_M（GGUF）** —— 一个面向端侧 / 边缘部署的 1B 稠密模型。

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
mkdir -p ~/minicpm-rocm && cd ~/minicpm-rocm
# 选择与你架构匹配的文件（此处为 gfx1151）
curl -L -o llama-rocm-gfx1151.zip \
  https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1292/llama-b1292-ubuntu-rocm-gfx1151-x64.zip
mkdir -p llama-bin && unzip -q llama-rocm-gfx1151.zip -d llama-bin
```

> 如果你已为 `minicpmv/`（MiniCPM-V）指南配置过 llama.cpp，可直接复用同一个 `llama-bin/` 目录 ——
> 二进制完全相同，仅模型不同。

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
cd ~/minicpm-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
./llama-cli --list-devices
# Available devices:
#   ROCm0: Radeon 8060S Graphics (65536 MiB, ... free)
```

---

#### 3. 进入 llama 后端目录并设置权限 / 环境变量

```bash
cd ~/minicpm-rocm/llama-bin
chmod +x llama-cli llama-server
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
```

> 注意：Lemonade 构建会把自带的 ROCm 运行库放在可执行文件旁边，因此除 `/opt/rocm/lib` 外，
> 还需把后端目录本身（上面的 `$PWD`）加入 `LD_LIBRARY_PATH`。

---

#### 4. 下载 MiniCPM5-1B GGUF

llama.cpp 使用 **GGUF 模型格式**。MiniCPM5-1B 提供现成的 GGUF：

| 文件 | 大小 | 适用场景 |
| --- | --- | --- |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | 参考质量 |
| `MiniCPM5-1B-Q8_0.gguf` | 1.1 GB | 相比 F16 质量损失极小 |
| `MiniCPM5-1B-Q4_K_M.gguf` | 657 MB | 端侧 / 最小显存 |

使用国内 Hugging Face 镜像 `https://hf-mirror.com/`：

```bash
mkdir -p ~/models/MiniCPM5-1B-GGUF && cd ~/models/MiniCPM5-1B-GGUF
export HF_ENDPOINT=https://hf-mirror.com

curl -L --fail -o MiniCPM5-1B-Q4_K_M.gguf \
  "https://hf-mirror.com/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q4_K_M.gguf"
```

> 提示：也可使用 `huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf`，
> 或 `hfd.sh` + `aria2` 进行更快、可断点续传的下载。

---

#### 5. CLI 文本测试（`llama-cli`）

```bash
cd ~/minicpm-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH

# 交互式对话（自动套用聊天模板）
./llama-cli \
  -m ~/models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 4096 --temp 0.7 --top-p 0.95 -n 2048
```

预期：在 `ROCm0` 上生成连贯文本。MiniCPM5-1B 是推理模型，可能会在最终答案前输出
`[Start thinking]` 思考块。

---

#### 6. 启动 llama-server（OpenAI 兼容接口）

```bash
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
cd ~/minicpm-rocm/llama-bin

./llama-server \
  -m ~/models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 8192 --jinja --host 127.0.0.1 --port 8080
```

> `--jinja` 启用模型自带的聊天模板（MiniCPM5-1B 推荐开启）。

---

#### 7. 测试接口（对话 + tokens/s）

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "MiniCPM5-1B",
  "messages": [{"role": "user", "content": "1+1=? 然后用一句话解释"}],
  "temperature": 0.7, "top_p": 0.95, "max_tokens": 256
}' | jq -r '
.choices[0].message.content as $txt |
(.usage.completion_tokens / (.timings.predicted_ms / 1000)) as $tps |
"回答:\n\($txt)\n\ntokens/s: \($tps|tostring)"
'
```

在 **Radeon 8060S（gfx1151），ROCm 7.13，ctx=8192** 上的参考结果：

- 解码：**约 185 tokens/s**（MiniCPM5-1B 仅 1B，因此比 8B 模型快很多）
- **tokens/s 取决于你的实际硬件。**

#### 生成参数

| 模式 | `--temp` | `--top-p` | 适用场景 |
| --- | --- | --- | --- |
| 思考 | 0.9 | 0.95 | 推理、数学、代码、多步骤 |
| 不思考 | 0.7 | 0.95 | 快速助手、低时延 |

---

### 二、方式二：Docker / 从源码编译

如果你更习惯 Docker，或想为 ROCm 从源码编译 llama.cpp
（`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`），步骤与 Qwen3 指南
（`qwen3/llamacpp-rocm7-deploy.md`）完全一致；MiniCPM 为纯文本模型，运行时无需额外参数：

```bash
./build/bin/llama-cli \
  -m /data/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 4096 -p "用两句话解释 AMD ROCm。"
```

#### 从你自己的检查点构建 GGUF

如果你微调了自己的 MiniCPM5-1B 变体并希望得到 GGUF，可用 llama.cpp 转换 + 量化：

```bash
python ./convert_hf_to_gguf.py /path/to/your-MiniCPM5-fp16-hf --outfile F16.gguf --outtype f16
./build/bin/llama-quantize F16.gguf MiniCPM5-1B-Q4_K_M.gguf Q4_K_M
```

---

### 参见

- `minicpmv/llamacpp-rocm7-deploy.md` —— llama.cpp 上的**多模态** MiniCPM-V（需额外的 `mmproj`
  投影文件 + `llama-mtmd-cli` 进行图像输入）。
- `qwen3/llamacpp-rocm7-deploy.md` —— 本指南所参照的文本模型 llama.cpp 标准流程。
