## vLLM Deployment of MiniCPM-V (Ubuntu 24.04 + ROCm 7+)

This section explains how to serve the **multimodal** model **MiniCPM-V 4.6** with **vLLM** on
an AMD GPU (Ubuntu 24.04 + ROCm 7+), including:

- Quick start using the official ROCm vLLM Docker image
- Manually building ROCm vLLM from source (for environments without Docker)

It follows the same structure as the `qwen3/vllm-rocm7-deploy.md` example. The MiniCPM-V
specifics are: it is a **vision-language model**, so you serve it with `--trust-remote-code`
and (for multi-image / video) `--limit-mm-per-prompt`, and you send images through the OpenAI
`chat/completions` API.

The example model is **MiniCPM-V 4.6** (1.3B vision-language; pairs a SigLIP2 vision encoder
with a Qwen3.5 backbone).

> Prerequisite: ROCm 7+ installation and verification is complete
> (see `env-prepare-ubuntu24-rocm7.md`). Reference machine: **AMD Ryzen AI MAX+ 395
> (Radeon 8060S, gfx1151), ROCm 7.13**.

---

### Model / Version Compatibility (read first)

MiniCPM-V 4.6 is **natively supported** in vLLM as the architecture
**`MiniCPMV4_6ForConditionalGeneration`**. Two hard requirements:

- **vLLM ≥ 0.22.0** — earlier releases do not contain the `minicpmv4_6` model module.
- **transformers ≥ 5.7** — the 4.6 architecture was merged into transformers as a standalone
  class (this repo's environment uses transformers 5.12.1).

> Important version note for ROCm source builds: vLLM 0.22.0+ pins **`torch == 2.11.0`** in its
> build metadata. If your ROCm PyTorch is older (e.g. `torch 2.9.x+rocm`), build vLLM in a
> **separate virtual environment** with a matching torch (a ROCm `torch 2.10/2.11` wheel), so you
> don't disturb a working inference/fine-tune environment. The Docker method (below) avoids this
> entirely by shipping a matched torch + vLLM inside the image.

---

## Method 1: Docker Method (Recommended)

Refer to the official Quickstart documentation:

- https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

> Note: If using Docker, you need `amdgpu-dkms`:
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

### 1. Start the vLLM Container

```bash
sudo docker pull rocm/vllm-dev:nightly # Get the latest image

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

The container's `/app/models` is mounted to the host's `~/models`.

> Tip: confirm the image's vLLM is new enough: `python -c "import vllm; print(vllm.__version__)"`
> inside the container should report ≥ 0.22.0. If not, pull a newer tag.

### 2. Download the Model (HF format, NOT the GGUF)

vLLM serves the full Hugging Face checkpoint (safetensors), not the llama.cpp GGUF:

```bash
# On the host (or inside the container), into ~/models
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openbmb/MiniCPM-V-4_6 \
  --local-dir ~/models/MiniCPM-V-4_6
```

### 3. Start the Model Service Inside the Container

Launch command from the official MiniCPM-V cookbook, adapted:

```bash
# Run inside the container
vllm serve /app/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8

# Quick start (--enforce-eager): disables HIP graph capture,
# faster startup but slightly slower inference. Useful for first bring-up.
vllm serve /app/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 8192 \
  --max-num-seqs 8
```

> - Use `--dtype bfloat16` on gfx1151 (fp16 can overflow to NaN on this model family; bf16 is
>   stable on Strix Halo).
> - 4.6 supports up to 256K context; start small (`--max-model-len 8192`) and raise it as VRAM allows.
> - For multi-image / video per request, add e.g. `--limit-mm-per-prompt '{"image": 4, "video": 1}'`.
> - The `MiniCPM-V-4_6-Thinking` checkpoint injects a `<think>` block by default; serve the plain
>   `MiniCPM-V-4_6` checkpoint for direct answers, or pass
>   `--chat-template-kwargs '{"enable_thinking": false}'`.

### 4. Test the API (text + image, with tokens/s)

Auto-detect the model id, then call the OpenAI-compatible endpoints (vLLM serves on port 8000):

```bash
MODEL_ID=$(curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[0].id')
echo "Detected model ID: $MODEL_ID"
```

**Text completion + tokens/s** (wall-clock method, same as the Qwen3 example):

```bash
start=$(date +%s.%N)
response=$(curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_ID\", \"prompt\": \"Explain large language models in one sentence\", \"max_tokens\": 128}")
end=$(date +%s.%N)

tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
duration=$(echo "$end - $start" | bc)
echo "$response" | jq -r '.choices[0].text'
echo "tokens: $tokens | time: ${duration}s | tokens/s: $(echo "scale=2; $tokens / $duration" | bc)"
```

**Multimodal chat** (image via base64 data URL on `chat/completions`):

```bash
IMG_B64=$(base64 -w0 /app/models/image.jpeg)
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
  \"model\": \"$MODEL_ID\",
  \"max_tokens\": 128,
  \"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"text\", \"text\": \"Describe this image in one sentence.\"},
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMG_B64\"}}
  ]}]
}" | jq -r '.choices[0].message.content'
```

---

## Method 2: Manual Build of vLLM (No Docker / Advanced Users)

Use this when Docker is unavailable. It mirrors the Qwen3 source-build flow; the steps that
differ for MiniCPM-V are called out.

### 1. Environment and Version Requirements

- vLLM **≥ 0.22.0** (required for MiniCPM-V 4.6); this guide uses a release tag, e.g. `v0.22.0`.
- GPU support includes Ryzen AI MAX / AI 300 (**gfx1151**/1150); ROCm **7.0.2+**.
- A ROCm PyTorch matching vLLM's pin (`torch == 2.11.0`). Build in a dedicated venv.

### 2. Set Up an Isolated Python venv with uv

Building in a separate venv protects any existing ROCm torch (e.g. a llama.cpp / fine-tune env):

```bash
uv venv --python 3.12 --seed ~/vllm-venv
source ~/vllm-venv/bin/activate
```

### 3. Install ROCm PyTorch Matching vLLM's Pin

```bash
# A ROCm torch wheel matching vLLM's build pin (2.11 nightly shown; use the closest available)
uv pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0 \
  "torch==2.11.0.dev*" torchvision
```

> If a `torch 2.11` ROCm wheel is unavailable for your ROCm, use the nearest (e.g. `2.10+rocm7.0`)
> and build with `--no-build-isolation` so vLLM compiles against the installed torch.

### 4. Triton for ROCm

ROCm PyTorch wheels already ship a matching Triton (`import triton` to confirm). Only build
Triton from source if `import triton` fails — see the Qwen3 guide's Triton section.

### 5. (Optional) FlashAttention for ROCm

vLLM runs on gfx1151 without a custom FlashAttention build; vLLM will fall back to a supported
attention backend. To build it anyway, follow the Qwen3 guide's FlashAttention section.

### 6. Build vLLM (ROCm, gfx1151)

```bash
# AMD SMI (from the local ROCm install)
cp -r /opt/rocm/share/amd_smi ./amdsmi_src && (cd ./amdsmi_src && uv pip install .)

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.22.0          # any tag >= 0.22.0 that contains minicpmv4_6.py

uv pip install -r requirements/rocm.txt
uv pip install numba scipy "huggingface-hub[cli,hf_transfer]" setuptools_scm setuptools wheel ninja cmake

export VLLM_TARGET_DEVICE="rocm"
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCM_HOME="/opt/rocm"

MAX_JOBS=16 uv pip install -e . --no-build-isolation
```

> This compiles vLLM's HIP kernels for gfx1151 (a few hundred source files); expect a long build.
> Verify the architecture is registered afterwards:
> ```bash
> python -c "from vllm import ModelRegistry; print('MiniCPMV4_6ForConditionalGeneration' in ModelRegistry.get_supported_archs())"
> ```

### 7. Serve and Test

```bash
vllm serve ~/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 8192 \
  --max-num-seqs 8
```

Then use the same text + multimodal API tests as Method 1 (Section 4).

---

### Notes / Status

- **Model-side compatibility is confirmed**: vLLM exposes `MiniCPMV4_6ForConditionalGeneration`
  (vLLM ≥ 0.22.0, transformers ≥ 5.7), and gfx1151 is a supported vLLM target on ROCm 7.0.2+.
- **The build is the heavy part on consumer ROCm**: without Docker, vLLM must be compiled from
  source for gfx1151, and the only versions with 4.6 support pin `torch 2.11`. Use an isolated venv
  so you don't disturb a working `torch 2.9.x+rocm` environment (such as the llama.cpp / fine-tune
  setup in this repo).
- For llama.cpp deployment of the same model (lighter weight, prebuilt binaries), see
  `minicpmv/llamacpp-rocm7-deploy.md`.
