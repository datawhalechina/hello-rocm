## vLLM Deployment of MiniCPM-V (Ubuntu 24.04 + ROCm 7+)

### Model Overview

[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) is an on-device multimodal model series developed by ModelBest and Tsinghua University NLP Lab (OpenBMB). MiniCPM-V 4.6 has only 1.3B parameters (SigLIP2 vision encoder + Qwen3.5 language backbone) and supports image understanding and text conversation.

- Model: [openbmb/MiniCPM-V-4_6](https://huggingface.co/openbmb/MiniCPM-V-4_6)

This guide deploys **MiniCPM-V 4.6** using **vLLM**, covering:

- Quick start with the official ROCm vLLM Docker image
- Manually building ROCm vLLM from source (for environments without Docker)

> Prerequisite: ROCm 7+ installation and verification is complete
> (see `env-prepare-ubuntu24-rocm7.md`). Reference machine: **AMD Ryzen AI MAX+ 395
> (Radeon 8060S, gfx1151), ROCm 7.13**.

---

### Version Requirements

MiniCPM-V 4.6 is natively supported in vLLM as the architecture `MiniCPMV4_6ForConditionalGeneration`. Requirements:

- **vLLM >= 0.22.0**
- **transformers >= 5.7**

> vLLM 0.22.0+ requires `torch == 2.11.0`. If your ROCm PyTorch is older (e.g. `torch 2.9.x+rocm`), build vLLM in a **separate virtual environment** to avoid breaking your existing setup. The Docker method avoids this entirely.

---

## Method 1: Docker (Recommended)

Reference: https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

> Docker requires `amdgpu-dkms`:
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

### 1. Start the vLLM Container

```bash
sudo docker pull rocm/vllm-dev:nightly

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

> Verify version inside the container: `python -c "import vllm; print(vllm.__version__)"` should report >= 0.22.0.

### 2. Download the Model (HF format, NOT GGUF)

vLLM needs the full Hugging Face checkpoint (safetensors), not the llama.cpp GGUF:

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openbmb/MiniCPM-V-4_6 \
  --local-dir ~/models/MiniCPM-V-4_6
```

### 3. Start the Model Service

```bash
vllm serve /app/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8
```

For quick bring-up, add `--enforce-eager` to skip HIP graph capture (faster startup, slightly slower inference).

> - Use `--dtype bfloat16` on gfx1151 (fp16 may produce NaN).
> - For multi-image / video per request, add `--limit-mm-per-prompt '{"image": 4, "video": 1}'`.
> - For reasoning mode, use the `MiniCPM-V-4_6-Thinking` checkpoint; otherwise use `MiniCPM-V-4_6`.

### 4. Test the API

Get the model id:

```bash
MODEL_ID=$(curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[0].id')
echo "Model ID: $MODEL_ID"
```

**Text completion:**

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

**Multimodal chat** (image via base64):

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

## Method 2: Build vLLM from Source (No Docker)

### 1. Requirements

- vLLM **>= 0.22.0**
- ROCm **7.0.2+**, GPU support for gfx1151/1150
- `torch == 2.11.0` (ROCm version), in an isolated venv

### 2. Create an Isolated Python venv

```bash
uv venv --python 3.12 --seed ~/vllm-venv
source ~/vllm-venv/bin/activate
```

### 3. Install ROCm PyTorch

```bash
uv pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0 \
  "torch==2.11.0.dev*" torchvision
```

> If no `torch 2.11` wheel is available, use the closest version and build with `--no-build-isolation`.

### 4. Install Triton

ROCm PyTorch wheels usually include Triton. Verify:

```bash
python -c "import triton; print(triton.__version__)"
```

If `import triton` fails, build from source:

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -e python
```

### 5. (Optional) FlashAttention

vLLM runs on gfx1151 without a custom FlashAttention build. If needed:

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
pip install -e .
```

### 6. Build vLLM

```bash
# AMD SMI
cp -r /opt/rocm/share/amd_smi ./amdsmi_src && (cd ./amdsmi_src && uv pip install .)

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.22.0

uv pip install -r requirements/rocm.txt
uv pip install numba scipy "huggingface-hub[cli,hf_transfer]" setuptools_scm setuptools wheel ninja cmake

export VLLM_TARGET_DEVICE="rocm"
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCM_HOME="/opt/rocm"

MAX_JOBS=16 uv pip install -e . --no-build-isolation
```

This compiles HIP kernels for gfx1151 and takes a while. Verify afterwards:

```bash
python -c "from vllm import ModelRegistry; print('MiniCPMV4_6ForConditionalGeneration' in ModelRegistry.get_supported_archs())"
```

### 7. Serve and Test

```bash
vllm serve ~/models/MiniCPM-V-4_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 8192 \
  --max-num-seqs 8
```

Then use the same API tests as Method 1 (Section 4).

---

### Notes

- Without Docker, vLLM must be compiled from source for gfx1151, and versions with 4.6 support require `torch 2.11`. Use an isolated venv.
- For llama.cpp deployment of the same model (lighter weight, prebuilt binaries), see `minicpmv/llamacpp-rocm7-deploy.md`.
