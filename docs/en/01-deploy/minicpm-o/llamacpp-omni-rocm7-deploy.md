## llama.cpp-omni Deployment on Ubuntu 24.04 + ROCm 7+

This section explains how to build and run **llama.cpp-omni** on AMD GPU under Ubuntu 24.04 + ROCm 7+, enabling voice input, image understanding, and TTS voice output for MiniCPM-o 4.5.

> Prerequisites:
> - [ROCm environment setup](/00-environment/) completed — `/opt/rocm` is present and `rocminfo` reports your GPU.
> - [MiniCPM-o 4.5 model introduction](./minicpm-o-model.md) read — you know which GGUF files are required.

---

### 1. Identify Your GPU Architecture

First confirm your GPU's gfx architecture number — you'll need it for the build:

```bash
rocminfo | grep -i "gfx"
# or
amd-smi | grep -i "gfx"
```

Common AMD GPU architecture codes:

| GPU Series | gfx code |
|------------|----------|
| RX 7900 XTX / 7900 XT | gfx1100 |
| RX 7800 XT / 7700 XT | gfx1101 |
| RX 9070 XT / 9070 | gfx1150 |
| Ryzen AI MAX+ 395 (Strix Halo APU) | **gfx1151** |
| Instinct MI300X | gfx942 |

---

### 2. Clone and Build llama.cpp-omni

#### 2.1 Install build dependencies

```bash
# Ubuntu 22.04 / 24.04
sudo apt update && sudo apt install -y \
    git cmake build-essential libcurl4-openssl-dev \
    python3-pip pkg-config
```

#### 2.2 Clone the repository

```bash
mkdir -p ~/omni && cd ~/omni
git clone https://github.com/tc-mb/llama.cpp-omni.git repo
cd repo
```

#### 2.3 Configure and build (generic AMD GPU)

Replace `AMDGPU_TARGETS` with your GPU's gfx code:

```bash
# Set your GPU architecture
export LLAMACPP_ROCM_ARCH=gfx1100   # ← replace as needed, e.g. gfx1150, gfx1101

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="$LLAMACPP_ROCM_ARCH" \
    -DLLAMA_CURL=ON \
    -DHIP_PLATFORM=amd \
    --rocm-path=/opt/rocm

cmake --build build --target llama-server llama-omni-cli -j$(nproc)
```

After a successful build, `build/bin/` will contain both `llama-omni-cli` and `llama-server`.

> If the build can't find the `hip` / `rocblas` CMake packages, or you hit `hipErrorInvalidImage` / `Tensile` errors at runtime, see Section 5, "Troubleshooting".

---

### 3. Download GGUF Model Files

MiniCPM-o 4.5 requires 10 GGUF files (~8.3 GB total). The directory structure must match exactly, as `llama-server` looks up sub-models via fixed relative paths.

Create directories first:

```bash
cd ~/omni
mkdir -p models/vision models/audio models/tts models/token2wav-gguf
```

---

#### Option A: Download from Hugging Face

Model page: [OpenBMB/MiniCPM-o-4_5-gguf](https://huggingface.co/OpenBMB/MiniCPM-o-4_5-gguf)

**Using huggingface-cli (recommended)**:

```bash
pip install huggingface_hub

huggingface-cli download OpenBMB/MiniCPM-o-4_5-gguf \
    --local-dir ~/omni/models \
    --local-dir-use-symlinks False
```

> `--local-dir-use-symlinks False` writes files directly into the target directory instead of creating symlinks, avoiding any path confusion.

**Using the hfd script (aria2 multi-thread acceleration)**:

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod +x hfd.sh
sudo apt install -y aria2

# Get your token from https://huggingface.co/settings/tokens
./hfd.sh OpenBMB/MiniCPM-o-4_5-gguf \
    --hf_username <YOUR_HF_USERNAME> \
    --hf_token hf_*** \
    --local-dir ~/omni/models
```

---

#### Option B: Download from ModelScope

Model page: [OpenBMB/MiniCPM-o-4_5-gguf](https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf)

```bash
pip install modelscope

modelscope download --model OpenBMB/MiniCPM-o-4_5-gguf \
    --local_dir ~/omni/models
```

Or use `curl` per file (supports `-C -` resume):

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

#### 3.2 Verify files

```bash
cd ~/omni/models
ls -lh . vision/ audio/ tts/ token2wav-gguf/
```

Expected output (file names must match exactly):

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

### 4. Prepare Test Audio and Run Inference

#### 4.1 Prepare a test audio file

`llama-omni-cli`'s `--test` mode requires a WAV file (16 kHz, mono):

```bash
# Option A: record your voice (requires sox)
sudo apt install -y sox
rec -r 16000 -c 1 /tmp/test.wav trim 0 5   # record 5 seconds, Ctrl+C to stop

# Option B: convert an existing file (requires ffmpeg)
ffmpeg -i input.mp3 -ar 16000 -ac 1 /tmp/test.wav
```

#### 4.2 Run CLI inference (Turn-based mode)

```bash
cd ~/omni/repo

# Set runtime environment (generic AMD GPU)
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0

# Run inference (voice input + TTS output)
./build/bin/llama-omni-cli \
    -m ~/omni/models/MiniCPM-o-4_5-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    --test /tmp/test 1
```

> `--test <prefix> <n>` reads `<prefix>0000.wav` as voice input and generates a text reply with TTS audio.
>
> Add `--no-tts` if you want text output only (skips TTS generation).

TTS-generated audio is written to `repo/tools/omni/output/round_000/tts_wav/`.

#### 4.3 Omni mode (voice + image)

```bash
# Place an image next to the audio (same prefix, .jpg extension)
cp your_image.jpg /tmp/test0000.jpg

# Add --omni flag
./build/bin/llama-omni-cli \
    -m ~/omni/models/MiniCPM-o-4_5-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    --omni \
    --test /tmp/test 1
```

#### 4.4 Reference performance

On AMD Ryzen AI MAX+ 395 (gfx1151, 64 GB unified memory):

| Phase | Speed |
|-------|-------|
| Prompt processing (prefill) | ~282–290 tokens/s |
| Text generation (decode) | ~39 tokens/s |

---

### 5. Troubleshooting

<details>
<summary>"hipErrorInvalidImage" or "Tensile: hipModuleLoadData failed" at runtime (gfx1151 users)</summary>

**Symptom**: The build succeeds, but `llama-omni-cli` / `llama-server` crashes immediately on launch with:

```
hipErrorInvalidImage: device kernel image is invalid
Tensile: hipModuleLoadData failed
```

**Cause**: gfx1151 (Strix Halo APU) is a relatively new architecture. Early system `/opt/rocm` releases (e.g. 7.12.0) shipped a rocBLAS Tensile library missing the complete GEMM kernels for this GPU.

> **Check whether you still need this fix**: As of ROCm 7.13, gfx1151 is on the official support list. If your system is ROCm 7.13 or later, try the generic build/run flow from Section 2 first — only apply the fix below if you actually hit the error above.

**Fix**: Install the [TheRock nightly SDK](https://rocm.nightlies.amd.com/v2/gfx1151/) matching your system's ROCm major version (it includes complete gfx1151 Tensile kernels), rebuild with a merged prefix, and point the runtime at its rocBLAS directory.

**Step 1: Install the TheRock nightly SDK matching your system version**

```bash
mkdir -p ~/omni/rocm_sdk && cd ~/omni/rocm_sdk

# Choose the alpha version matching your system ROCm major (system 7.12 → 7.12.0a, system 7.13 → 7.13.0a)
# Key: the SDK .so soname must match the system driver, or you'll hit runtime errors like hipMemcpy failures
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    "rocm-sdk-core==7.13.0a*" \
    "rocm-sdk-devel==7.13.0a*" \
    "rocm-sdk-libraries-gfx1151==7.13.0a*" \
    --target ./pkg --no-deps

# Extract the devel tar (headers/cmake are bundled inside)
find ./pkg -name "_devel.tar" -exec tar xf {} -C . \; 2>/dev/null || true
```

**Step 2: Build with a merged ROCm prefix**

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

**Step 3: Run with the fixed rocBLAS environment**

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

> For the Web Demo, put these three environment variables into the launch script (see `start_amd.sh` in [Web Demo Deployment](./webdemo-rocm7-deploy.md)).

</details>

<details>
<summary>Cannot find audio / vision / TTS sub-model files</summary>

`llama-omni-cli` looks for sub-models at fixed relative paths from the same directory as the main `-m` GGUF file. Ensure the file names and directory hierarchy exactly match Section 3:

```
models/
├── MiniCPM-o-4_5-Q4_K_M.gguf      ← -m points here
├── vision/MiniCPM-o-4_5-vision-F16.gguf
├── audio/MiniCPM-o-4_5-audio-F16.gguf
├── tts/MiniCPM-o-4_5-tts-F16.gguf
└── ...
```

</details>

<details>
<summary>CMake error: "Cannot find cmake/hip"</summary>

The system `/opt/rocm` may be missing CMake config files:

```bash
find /opt/rocm -name "hip-config.cmake" 2>/dev/null

# If not found, try installing
sudo apt install rocm-cmake hip-dev 2>/dev/null || \
pip install rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1100/
```

</details>

<details>
<summary>No TTS audio generated</summary>

TTS is enabled by default (use `--no-tts` to skip). It requires all 5 `token2wav-gguf/` files:

```bash
ls ~/omni/models/token2wav-gguf/
# Should show: encoder.gguf  flow_extra.gguf  flow_matching.gguf  hifigan2.gguf  prompt_cache.gguf
```

</details>

---

### References

- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [OpenBMB/MiniCPM-o-4_5-gguf (ModelScope)](https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf)
- [TheRock Nightly SDK (gfx1151)](https://rocm.nightlies.amd.com/v2/gfx1151/)
- [GPU Architecture Table](/00-environment/rocm-gpu-architecture-table)

---

> Once CLI inference is working, see [Web Demo Full-Duplex Deployment](./webdemo-rocm7-deploy.md) to set up a complete web frontend with all four conversation modes.
