## llama.cpp-omni 零基础部署（Ubuntu 24.04 + ROCm 7+）

本节介绍如何在 Ubuntu 24.04 + ROCm 7+ 环境下，使用 **llama.cpp-omni** 编译并运行 MiniCPM-o 4.5，实现语音输入、图像理解和 TTS 语音输出。

> 前置条件：
> - 已完成 [ROCm 基础环境安装](/zh/00-environment/)，系统已有 `/opt/rocm` 且 `rocminfo` 正常输出 GPU 信息。
> - 已阅读 [MiniCPM-o 4.5 模型介绍](./minicpm-o-model.md)，了解所需 GGUF 文件。

---

### 一、了解你的 GPU 架构

在编译之前，首先确认 GPU 的 gfx 架构编号，编译时需要用到：

```bash
rocminfo | grep -i "gfx"
# 或
amd-smi | grep -i "gfx"
```

常见 AMD GPU 对应的架构号：

| GPU 系列 | gfx 编号 |
|----------|----------|
| RX 7900 XTX / 7900 XT | gfx1100 |
| RX 7800 XT / 7700 XT | gfx1101 |
| RX 9070 XT / 9070 | gfx1150 |
| Ryzen AI MAX+ 395（Strix Halo APU） | **gfx1151** |
| Instinct MI300X | gfx942 |

---

### 二、克隆并编译 llama.cpp-omni

#### 1. 安装编译依赖

```bash
# Ubuntu 22.04 / 24.04
sudo apt update && sudo apt install -y \
    git cmake build-essential libcurl4-openssl-dev \
    python3-pip pkg-config
```

#### 2. 克隆仓库

```bash
mkdir -p ~/omni && cd ~/omni
git clone https://github.com/tc-mb/llama.cpp-omni.git repo
cd repo
```

#### 3. 配置 ROCm 编译（通用 AMD GPU）

以下命令适用于绝大多数 AMD GPU（gfx1100 / gfx1150 / gfx942 等），将 `AMDGPU_TARGETS` 替换为你的 GPU 架构号：

```bash
# 设置你的 GPU 架构
export LLAMACPP_ROCM_ARCH=gfx1100   # ← 按实际替换，例如 gfx1150、gfx1101

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="$LLAMACPP_ROCM_ARCH" \
    -DLLAMA_CURL=ON \
    -DHIP_PLATFORM=amd \
    --rocm-path=/opt/rocm

cmake --build build --target llama-server llama-omni-cli -j$(nproc)
```

编译成功后，`build/bin/` 目录下会生成 `llama-omni-cli` 和 `llama-server` 两个关键二进制。

> 若编译时报找不到 `hip` / `rocblas` 的 CMake 包，或运行时报 `hipErrorInvalidImage` / `Tensile` 相关错误，请参考第五节「常见故障排查」。

---

### 三、下载 GGUF 模型文件

MiniCPM-o 4.5 共需要 10 个 GGUF 文件（约 8.3 GB），目录结构须精确匹配（`llama-server` 按固定相对路径查找子模型）。

先创建目录：

```bash
cd ~/omni
mkdir -p models/vision models/audio models/tts models/token2wav-gguf
```

---

#### 方式一：从 Hugging Face 下载

模型主页：[OpenBMB/MiniCPM-o-4_5-gguf](https://huggingface.co/OpenBMB/MiniCPM-o-4_5-gguf)

**使用 huggingface-cli（推荐）**：

```bash
pip install huggingface_hub

huggingface-cli download OpenBMB/MiniCPM-o-4_5-gguf \
    --local-dir ~/omni/models \
    --local-dir-use-symlinks False
```

> `--local-dir-use-symlinks False` 确保文件直接写入目标目录，而非创建软链接，避免路径歧义。

**使用 hfd 脚本（支持 aria2 多线程加速）**：

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod +x hfd.sh
sudo apt install -y aria2

# 在 https://huggingface.co/settings/tokens 获取 token
./hfd.sh OpenBMB/MiniCPM-o-4_5-gguf \
    --hf_username <YOUR_HF_USERNAME> \
    --hf_token hf_*** \
    --local-dir ~/omni/models
```

---

#### 方式二：从 ModelScope 下载

模型主页：[OpenBMB/MiniCPM-o-4_5-gguf](https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf)

```bash
pip install modelscope

modelscope download --model OpenBMB/MiniCPM-o-4_5-gguf \
    --local_dir ~/omni/models
```

或使用 `curl` 逐文件下载（支持断点续传 `-C -`）：

```bash
BASE="https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf/resolve/master"
cd ~/omni/models

curl -C - -O "$BASE/MiniCPM-o-4_5-Q4_K_M.gguf"
curl -C - -o vision/MiniCPM-o-4_5-vision-F16.gguf "$BASE/vision/MiniCPM-o-4_5-vision-F16.gguf"
curl -C - -o audio/MiniCPM-o-4_5-audio-F16.gguf   "$BASE/audio/MiniCPM-o-4_5-audio-F16.gguf"
curl -C - -o tts/MiniCPM-o-4_5-tts-F16.gguf       "$BASE/tts/MiniCPM-o-4_5-tts-F16.gguf"
curl -C - -o tts/MiniCPM-o-4_5-projector-F16.gguf "$BASE/tts/MiniCPM-o-4_5-projector-F16.gguf"
for f in encoder flow_matching flow_extra hifigan2 prompt_cache; do
    curl -C - -o "token2wav-gguf/${f}.gguf" "$BASE/token2wav-gguf/${f}.gguf"
done
```

---

#### 2. 验证文件完整性

```bash
cd ~/omni/models
ls -lh . vision/ audio/ tts/ token2wav-gguf/
```

预期输出（文件名须完全一致）：

```
.
├── MiniCPM-o-4_5-Q4_K_M.gguf         ~4.9 GB
├── audio/
│   └── MiniCPM-o-4_5-audio-F16.gguf  ~1.2 GB
├── tts/
│   ├── MiniCPM-o-4_5-tts-F16.gguf    ~0.5 GB
│   └── MiniCPM-o-4_5-projector-F16.gguf
├── token2wav-gguf/
│   ├── encoder.gguf
│   ├── flow_matching.gguf
│   ├── flow_extra.gguf
│   ├── hifigan2.gguf
│   └── prompt_cache.gguf
└── vision/
    └── MiniCPM-o-4_5-vision-F16.gguf  ~0.9 GB
```

---

### 四、准备测试音频并运行推理

#### 1. 准备一段测试音频

`llama-omni-cli` 的 `--test` 模式需要一个 WAV 文件（16kHz 单声道）：

```bash
# 方式一：录制自己的声音（需要 sox）
sudo apt install -y sox
rec -r 16000 -c 1 /tmp/test.wav trim 0 5   # 录制 5 秒，Ctrl+C 停止

# 方式二：用已有音频转换格式（需要 ffmpeg）
ffmpeg -i input.mp3 -ar 16000 -ac 1 /tmp/test.wav
```

#### 2. 运行 CLI 推理（Turn-based 模式）

```bash
cd ~/omni/repo

# 设置运行时环境（通用 AMD GPU）
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0

# 运行推理（语音输入 + TTS 输出）
./build/bin/llama-omni-cli \
    -m ~/omni/models/MiniCPM-o-4_5-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    --test /tmp/test 1
```

> `--test <prefix> <n>` 模式会读取 `<prefix>0000.wav` 作为语音输入，并生成文本回复和 TTS 语音。
>
> 若只需文本输出、暂不需要 TTS 语音，可追加 `--no-tts`。

TTS 生成的语音文件会写入 `repo/tools/omni/output/round_000/tts_wav/`。

#### 3. Omni 模式（语音 + 图像）

```bash
# 在音频文件旁放一张图片（同名前缀，.jpg 结尾）
cp your_image.jpg /tmp/test0000.jpg

# 加上 --omni 参数
./build/bin/llama-omni-cli \
    -m ~/omni/models/MiniCPM-o-4_5-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    --omni \
    --test /tmp/test 1
```

#### 4. 参考性能指标

在 AMD Ryzen AI MAX+ 395（gfx1151，64 GB 统一内存）上：

| 阶段 | 速度 |
|------|------|
| Prompt 处理（prefill） | ~282–290 tokens/s |
| 文本生成（decode） | ~39 tokens/s |

---

### 五、常见故障排查

<details>
<summary>运行时报 "hipErrorInvalidImage" 或 "Tensile: hipModuleLoadData failed"（gfx1151 用户）</summary>

**现象**：编译能通过，但 `llama-omni-cli` / `llama-server` 一启动就报：

```
hipErrorInvalidImage: device kernel image is invalid
Tensile: hipModuleLoadData failed
```

**原因**：gfx1151（Strix Halo APU）是较新的架构。早期系统 `/opt/rocm`（如 7.12.0）的 rocBLAS Tensile 库缺少该 GPU 的完整 GEMM 内核。

> **先确认是否仍需修复**：自 ROCm 7.13 起，gfx1151 已进入官方支持列表。如果你的系统是 ROCm 7.13 或更新版本，建议先按第二节的通用流程直接编译运行；只有确实遇到上述报错时，再执行下面的修复步骤。

**修复方案**：安装与系统 ROCm 主版本匹配的 [TheRock nightly SDK](https://rocm.nightlies.amd.com/v2/gfx1151/)（含完整的 gfx1151 Tensile 内核），用合并前缀重新编译，并在运行时指向其 rocBLAS 目录。

**步骤 1：安装与系统主版本匹配的 TheRock nightly SDK**

```bash
mkdir -p ~/omni/rocm_sdk && cd ~/omni/rocm_sdk

# 选择与系统 ROCm 主版本匹配的 alpha 版本（系统 7.12 → 7.12.0a，系统 7.13 → 7.13.0a）
# 关键：SDK 的 .so soname 必须与系统驱动一致，否则会出现 hipMemcpy 等运行时错误
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    "rocm-sdk-core==7.13.0a*" \
    "rocm-sdk-devel==7.13.0a*" \
    "rocm-sdk-libraries-gfx1151==7.13.0a*" \
    --target ./pkg --no-deps

# rocm-sdk-devel 的头文件/cmake 在 tar 包中，手动解压
find ./pkg -name "_devel.tar" -exec tar xf {} -C . \; 2>/dev/null || true
```

**步骤 2：构建合并 ROCm 前缀并重新编译**

```bash
mkdir -p ~/omni/rocm_merged
ln -sfn /opt/rocm/* ~/omni/rocm_merged/ 2>/dev/null || true

SDK_LIB=$(find ~/omni/rocm_sdk/pkg -path "*/_rocm_sdk_libraries_gfx1151" -type d | head -1)
SDK_CORE=$(find ~/omni/rocm_sdk/pkg -path "*/_rocm_sdk_core" -type d | head -1)
cp -rn "$SDK_LIB/lib/cmake" ~/omni/rocm_merged/lib/ 2>/dev/null || true

cd ~/omni/repo
cmake -B build_fix \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1151 \
    -DLLAMA_CURL=ON \
    -DHIP_PLATFORM=amd \
    -DCMAKE_PREFIX_PATH="$HOME/omni/rocm_merged;$SDK_LIB" \
    --rocm-path=/opt/rocm \
    -DAMD_DEVICE_LIBS_PREFIX=/opt/rocm/lib/llvm/amdgcn/bitcode

cmake --build build_fix --target llama-server llama-omni-cli -j$(nproc)
```

**步骤 3：运行时注入修复后的 rocBLAS 环境**

```bash
SDK_LIB=$(find ~/omni/rocm_sdk/pkg -path "*/_rocm_sdk_libraries_gfx1151" -type d | head -1)
SDK_CORE=$(find ~/omni/rocm_sdk/pkg -path "*/_rocm_sdk_core" -type d | head -1)

export LD_LIBRARY_PATH="$SDK_LIB/lib:$SDK_CORE/lib"
export ROCBLAS_TENSILE_LIBPATH="$SDK_LIB/lib/rocblas/library"
export HIP_VISIBLE_DEVICES=0

cd ~/omni/repo
./build_fix/bin/llama-omni-cli \
    -m ~/omni/models/MiniCPM-o-4_5-Q4_K_M.gguf \
    -ngl 99 -c 4096 \
    --test /tmp/test 1
```

> 部署 Web Demo 时，把上面三个环境变量放进启动脚本即可（见 [Web Demo 部署](./webdemo-rocm7-deploy.md) 中的 `start_amd.sh`）。

</details>

<details>
<summary>找不到音频 / 视觉 / TTS 子模型文件</summary>

`llama-omni-cli` 在 `-m` 参数指定的 GGUF 文件的**同级目录**下，按固定相对路径查找子模型。请确认以下文件名和目录层级与第三节完全一致：

```
models/
├── MiniCPM-o-4_5-Q4_K_M.gguf      ← -m 指向此文件
├── vision/MiniCPM-o-4_5-vision-F16.gguf
├── audio/MiniCPM-o-4_5-audio-F16.gguf
├── tts/MiniCPM-o-4_5-tts-F16.gguf
└── ...
```

</details>

<details>
<summary>编译时报 "Cannot find cmake/hip" 等 CMake 错误</summary>

系统 `/opt/rocm` 可能缺少 CMake 配置包（部分发行版的 ROCm 包不含这些文件）。尝试：

```bash
# 确认 hip-config.cmake 存在
find /opt/rocm -name "hip-config.cmake" 2>/dev/null

# 若不存在，尝试加装
sudo apt install rocm-cmake hip-dev 2>/dev/null || \
pip install rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1100/
```

</details>

<details>
<summary>TTS 语音没有生成</summary>

不加 `--no-tts` 时，TTS 默认开启，但需要所有 5 个 `token2wav-gguf/` 文件都存在。确认：

```bash
ls ~/omni/models/token2wav-gguf/
# 应输出：encoder.gguf  flow_extra.gguf  flow_matching.gguf  hifigan2.gguf  prompt_cache.gguf
```

</details>

---

### 参考资源

- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [OpenBMB/MiniCPM-o-4_5-gguf（ModelScope）](https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf)
- [TheRock Nightly SDK（gfx1151）](https://rocm.nightlies.amd.com/v2/gfx1151/)
- [GPU 架构对照表](/zh/00-environment/rocm-gpu-architecture-table)

---

> CLI 推理跑通后，如需搭建完整的 Web 前端和多种对话模式，请继续阅读 [Web Demo 全双工部署](./webdemo-rocm7-deploy.md)。
