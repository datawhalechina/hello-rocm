## llama.cpp Deployment of MiniCPM (Ubuntu 24.04 + ROCm 7+)

### Model Overview

[MiniCPM](https://github.com/OpenBMB/MiniCPM) is an on-device large language model series developed by ModelBest and Tsinghua University NLP Lab (OpenBMB). MiniCPM5-1B is the latest text-only model in the series, with only 1B parameters and support for Think / No-think reasoning modes.

- Model: [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
- GGUF: [openbmb/MiniCPM5-1B-GGUF](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)

This guide deploys **MiniCPM5-1B Q4_K_M (GGUF)** using **llama.cpp**, covering:

- Prebuilt executables (recommended)
- Docker + official ROCm image for building from source

MiniCPM is text-only — a single GGUF file with standard `llama-cli` / `llama-server`. For multimodal (image + text), see the `minicpmv/` directory.

> Prerequisite: ROCm 7+ system installation and verification is complete
> (see `env-prepare-ubuntu24-rocm7.md`). Verified on **AMD Ryzen AI MAX+ 395 (Radeon 8060S,
> gfx1151), ROCm 7.13**.

---

### Method 1 (Recommended): Prebuilt Executables

#### 1. Download the Prebuilt Version

Use the prebuilt llama.cpp provided by Lemonade, where:

- **370** corresponds to the **gfx1150** architecture
- **395** corresponds to the **gfx1151** architecture

Related links:

- https://github.com/lemonade-sdk/llamacpp-rocm
- https://github.com/lemonade-sdk/llamacpp-rocm/releases

```bash
mkdir -p ~/minicpm-rocm && cd ~/minicpm-rocm
# Pick the asset matching your architecture (gfx1151 shown here)
curl -L -o llama-rocm-gfx1151.zip \
  https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1292/llama-b1292-ubuntu-rocm-gfx1151-x64.zip
mkdir -p llama-bin && unzip -q llama-rocm-gfx1151.zip -d llama-bin
```

---

#### 2. Verify ROCm 7+ Installation (Must Be System-level ROCm)

```bash
amd-smi
```

You should see your GPU model, driver, and ROCm version, e.g.:

```
MARKET_NAME: Radeon 8060S Graphics
TARGET_GRAPHICS_VERSION: gfx1151
ROCm version: 7.13.0
```

Confirm llama.cpp can see the GPU:

```bash
cd ~/minicpm-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
./llama-cli --list-devices
# Available devices:
#   ROCm0: Radeon 8060S Graphics (65536 MiB, ... free)
```

---

#### 3. Set Permissions and Environment Variables

```bash
cd ~/minicpm-rocm/llama-bin
chmod +x llama-cli llama-server
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
```

> The Lemonade build bundles its own ROCm runtime libraries next to the binaries, so add `$PWD` to `LD_LIBRARY_PATH` in addition to `/opt/rocm/lib`.

---

#### 4. Download the MiniCPM5-1B GGUF

llama.cpp uses the **GGUF model format**. Available quantizations:

| File | Size | Notes |
| --- | --- | --- |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | Full precision |
| `MiniCPM5-1B-Q8_0.gguf` | 1.1 GB | Minimal quality loss |
| `MiniCPM5-1B-Q4_K_M.gguf` | 657 MB | For limited VRAM |

Using the Chinese Hugging Face mirror:

```bash
mkdir -p ~/models/MiniCPM5-1B-GGUF && cd ~/models/MiniCPM5-1B-GGUF
export HF_ENDPOINT=https://hf-mirror.com

curl -L --fail -o MiniCPM5-1B-Q4_K_M.gguf \
  "https://hf-mirror.com/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q4_K_M.gguf"
```

> You can also use `huggingface-cli download` or `hfd.sh` + `aria2` for resumable downloads.

---

#### 5. CLI Text Test (`llama-cli`)

```bash
cd ~/minicpm-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH

./llama-cli \
  -m ~/models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 4096 --temp 0.7 --top-p 0.95 -n 2048
```

MiniCPM5-1B supports reasoning mode and may emit a `[Start thinking]` block before its final answer.

---

#### 6. Start llama-server (OpenAI-compatible API)

```bash
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
cd ~/minicpm-rocm/llama-bin

./llama-server \
  -m ~/models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 8192 --jinja --host 127.0.0.1 --port 8080
```

> `--jinja` enables the model's bundled chat template (recommended for MiniCPM5-1B).

---

#### 7. Test the API

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "MiniCPM5-1B",
  "messages": [{"role": "user", "content": "1+1=? Then explain in one short sentence."}],
  "temperature": 0.7, "top_p": 0.95, "max_tokens": 256
}' | jq -r '
.choices[0].message.content as $txt |
(.usage.completion_tokens / (.timings.predicted_ms / 1000)) as $tps |
"Answer:\n\($txt)\n\ntokens/s: \($tps|tostring)"
'
```

Reference: **~185 tokens/s** decode on Radeon 8060S (gfx1151), ROCm 7.13, ctx=8192. Actual speed depends on hardware.

#### Generation Parameters

| Mode | `--temp` | `--top-p` | When to use |
| --- | --- | --- | --- |
| Think | 0.9 | 0.95 | reasoning, math, code |
| No-think | 0.7 | 0.95 | fast assistant, low latency |

---

### Method 2: Docker (Official ROCm llama.cpp Image)

> Docker requires `amdgpu-dkms`:
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

#### 1. Start the Container

```bash
export MODEL_PATH='~/models'

sudo docker run -it \
  --name=$(whoami)_llamacpp_minicpm \
  --privileged --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 16G \
  -v $MODEL_PATH:/data \
  rocm/dev-ubuntu-24.04:7.0-complete
```

---

#### 2. Prepare the Workspace

```bash
apt-get update && apt-get install -y nano libcurl4-openssl-dev cmake git
mkdir -p /workspace && cd /workspace
```

---

#### 3. Clone llama.cpp

```bash
git clone https://github.com/ROCm/llama.cpp
cd llama.cpp
```

---

#### 4. Set ROCm Architecture

```bash
# AI MAX 395 (gfx1151) example
export LLAMACPP_ROCM_ARCH=gfx1151
```

---

#### 5. Build llama.cpp

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

#### 6. Run the Test

```bash
./build/bin/llama-cli \
  -m /data/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 4096 -p "Explain AMD ROCm in two sentences."
```

#### Build a GGUF from Your Own Checkpoint

If you fine-tuned your own MiniCPM5-1B variant:

```bash
python ./convert_hf_to_gguf.py /path/to/your-MiniCPM5-fp16-hf --outfile F16.gguf --outtype f16
./build/bin/llama-quantize F16.gguf MiniCPM5-1B-Q4_K_M.gguf Q4_K_M
```

---

### Screenshot Example

<div align='center'>
    <img src="../../../public/images/01-deploy/minicpm/minicpm5-example.jpg" alt="MiniCPM5-1B llama.cpp example" width="90%">
</div>
