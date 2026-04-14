## llama.cpp 零基础环境部署（Ubuntu 24.04 + ROCm 7+）

本节介绍如何在 Ubuntu 24.04 + ROCm 7+ 环境下，使用 **llama.cpp** 进行推理，包括：

- 使用预构建的可执行文件（推荐）
- 使用 Docker + 官方 ROCm 镜像自行编译

示例模型以 **Qwen3-8B Q4_K_M（GGUF 格式）** 为主。

> 前置条件：已完成 ROCm 7.1.0 系统安装与验证（见 `env-prepare-ubuntu24-rocm7.md`）。

---

### 一、方式一（推荐）：预构建的可执行文件

#### 1. 下载预构建版本

使用 Lemonade 提供的预构建版本，其中：

- **370** 对应 **gfx1150** 架构  
- **395** 对应 **gfx1151** 架构

相关链接：

- https://github.com/lemonade-sdk/llamacpp-rocm  
- https://github.com/lemonade-sdk/llamacpp-rocm/releases

<img src="./images/media/image7.png" style="width:5.75in;height:5.10417in" />

---

#### 2. 确认 ROCm 7+ 安装（必须为系统版 ROCm）

使用 `amd-smi` 确认 GPU 型号、驱动、ROCm 版本：

```bash
amd-smi
```

示例输出（可看到 GPU 型号、驱动版本、ROCm 版本）：

<img src="./images/media/image8.png" style="width:5.75in;height:4.01042in" />

若输出正常，说明可以使用 GPU 进行推理。

---

#### 3. 进入 llama 后端目录并设置权限 / 环境变量

```bash
cd llama-*x64/
sudo chmod +x *
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

---

#### 4. 下载 Qwen3-8B Q4_K_M GGUF 模型

Ollama / llama-server 通常使用 **GGUF 模型格式**。

这里使用国内 huggingface 镜像 `https://hf-mirror.com/`，需要登录  
[huggingface](https://huggingface.co/) 获取用户名与 token。

参考命令：

```bash
# 创建模型存放目录
mkdir -p ~/models
cd ~/models

# 下载 GGUF 文件（示例，需替换成官方/开源下载地址）
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
sudo apt update
sudo apt install aria2

# 需要登录 huggingface 获取你的 用户名 和 token
./hfd.sh netrunnerllm/Qwen3-8B-Q4_K_M-GGUF --hf_username <USERNAME> --hf_token hf_***
```

---

#### 5. 启动 llama-server

```bash
# rocm 驱动库链接
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
cd llama-*x64/
./llama-server -m ~/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf -ngl 99
```

---

#### 6. 测试接口（curl + jq 计算 tokens/s）

使用 `curl` 请求本地的 `llama-server` 接口，并统计 QPS / TPS：

```bash
curl -s -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen3-8b-q4_k_m",
  "prompt": "用一句话解释大语言模型",
  "max_tokens": 128
}' | jq -r '
# 打印生成文本
.choices[0].text as $txt |
# 计算 token/s
(.usage.completion_tokens / (.timings.predicted_ms / 1000)) as $tps |
"生成文本:\n\($txt)\n\ntokens/s: \($tps|tostring)"
'
```

截图示例：

<img src="./images/media/image9.png" style="width:5.75in;height:3.25in" />

测试结果示例（Qwen3-8B Q4_K_M，ctx=4096）：

- **约 40.71 tokens/s**

---

### 二、方式二：Docker 方式（官方 ROCm llama.cpp 镜像）

如果你更习惯使用 Docker，可以参考官方文档：

- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/llama-cpp-install.html

> 注意：若使用 Docker，需要安装 `amdgpu-dkms`：  
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html  
> 相关步骤在前文安装脚本中已包含；若未执行脚本，需要自行手动安装。

---

#### 1. 下载容器镜像

```bash
export MODEL_PATH='~/models'

sudo docker run -it \
  --name=$(whoami)_llamacpp \
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

进入容器后，设置你的工作目录与依赖：

```bash
apt-get update && apt-get install -y nano libcurl4-openssl-dev cmake git
mkdir -p /workspace && cd /workspace
```

---

#### 3. 容器内克隆 ROCm 官方 llama.cpp 仓库

```bash
git clone https://github.com/ROCm/llama.cpp
cd llama.cpp
```

---

#### 4. 设定 ROCm 架构（以 AI MAX 395 为例）

```bash
export LLAMACPP_ROCM_ARCH=gfx1151
```

如需同时为多种微架构编译，可使用：

```bash
export LLAMACPP_ROCM_ARCH=gfx803,gfx900,gfx906,gfx908,gfx90a,gfx942,gfx1010,gfx1030,gfx1032,gfx1100,gfx1101,gfx1102,gfx1150,gfx1151
```

---

#### 5. 编译并安装 llama.cpp

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=$LLAMACPP_ROCM_ARCH -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON && \
cmake --build build --config Release -j$(nproc)
```

---

#### 6. 测试安装

```bash
cd /workspace/llama.cpp
./build/bin/test-backend-ops
```

---

#### 7. 运行 Qwen3-8B Q4_K_M 测试

```bash
./build/bin/llama-cli -m /data/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf -ngl 99
```

截图示例：

<img src="./images/media/image10.png" style="width:5.75in;height:3.25in" />

测试结果（Qwen3-8B Q4_K_M，ctx=4096）：

- **约 39.60 tokens/s**


