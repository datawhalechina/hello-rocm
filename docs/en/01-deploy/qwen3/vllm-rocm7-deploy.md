## vLLM LLM Deployment from Scratch (Ubuntu 24.04 + ROCm 7+)

This section explains how to use **vLLM** for inference on Ubuntu 24.04 + ROCm 7+, including:

- Quick start using the official ROCm vLLM Docker image
- Manually compiling ROCm version vLLM (including Triton, FlashAttention, and other dependencies)

The example model is **Qwen3-8B Q4_K_M (GGUF format)**.

> Prerequisite: ROCm 7.1.0 installation and verification is complete (see `env-prepare-ubuntu24-rocm7.md`).

---

## Method 1: Docker Method (Recommended)

Refer to the official Quickstart documentation:

- https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

> Note: If using Docker, you need to install `amdgpu-dkms`:  
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html  
> These steps are included in the script mentioned earlier; otherwise, install manually.

---

### 1. Start the vLLM Container

Start the container from the official ROCm vLLM image:

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

The container's `/app/models` is mounted to the host's `~/models` (for storing GGUF models, etc.).

---

### 2. Start the Model Service Inside the Container

Inside the container, start the Qwen3-8B Q4_K_M GGUF model:

```bash
# Run inside the container
vllm serve /app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf --dtype float16 --max-model-len 4096 --max-num-seqs 32 --tokenizer qwen/Qwen3-8B

# Quick start (--enforce-eager): disables CUDA graph,
# faster startup but slightly slower inference (typically 10–20% slower, usually worth it for testing GGUF).
vllm serve /app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf \
  --tokenizer qwen/Qwen3-8B \
  --dtype float16 \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-model-len 4096
```

---

### 3. Performance Testing Script (tokens/s)

Below is a complete Bash script example for measuring the actual inference speed of **Qwen3-8B-Q4_K_M**:

```bash
# 1. Prepare a random prompt
RAND_PROMPT="Random code $(date +%N): Please describe the future of quantum computing in detail, with rich content and no repetition."

# 2. Record precise start time
start=$(date +%s.%N)

# 3. Send request and store in variable
response=$(curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
\"model\": \"/app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf\",
\"prompt\": \"$RAND_PROMPT\",
\"max_tokens\": 512,
\"temperature\": 0.8
}")

# 4. Record end time
end=$(date +%s.%N)

# 5. Parse content
# Extract raw generated text
content=$(echo "$response" | jq -r '.choices[0].text')
# Extract token count
tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
# Calculate duration
duration=$(echo "$end - $start" | bc)

# 6. Print output
echo "==================== Raw Content ===================="
echo "$content"
echo "=================================================="

if (( $(echo "$duration < 0.05" | bc -l) )); then
  echo "Abnormally fast response detected ($duration seconds), possibly cached."
else
  tps=$(echo "scale=2; $tokens / $duration" | bc)
  echo "Generated tokens: $tokens"
  echo "Total elapsed time: $duration seconds"
  echo "Actual inference speed: $tps tokens/s"
fi
echo "=================================================="
```

Test result example (Qwen3-8B Q4_K_M, ctx=4096):

- **Approximately 30.21 tokens/s**

Screenshot examples:

<div align='center'>
    <img src="../../../../public/images/01-deploy/qwen3/image11.png" alt="" width="90%">
</div>
<div align='center'>
    <img src="../../../../public/images/01-deploy/qwen3/image12.png" alt="" width="90%">
</div>

---

## Method 2: Manual Build of vLLM (For Advanced Users)

This section is longer and is intended for users with in-depth requirements for ROCm / Triton / FlashAttention / vLLM.

### 1. Environment and Version Requirements

Refer to the official documentation:

- https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

Key version information (example):

- vLLM 0.13.0
- GPU support: MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 (gfx1100/1101),
  Radeon RX 9000 (gfx1200/1201), Ryzen AI MAX / AI 300 (gfx1151/1150)
- ROCm 6.3 or above
  - MI350 requires ROCm 7.0+
  - Ryzen AI MAX / AI 300 requires ROCm 7.0.2+

---

### 2. Set Up Python Virtual Environment with uv

Reference: `https://www.runoob.com/python3/uv-tutorial.html`

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate
```

---

### 3. Install PyTorch with ROCm 7+ Support

```bash
# Install PyTorch
uv pip uninstall torch
uv pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

---

### 4. Install Triton for ROCm

Repository: `https://github.com/ROCm/triton.git`

```bash
uv pip install ninja cmake wheel pybind11
uv pip uninstall triton

git clone https://github.com/ROCm/triton.git
cd triton
# git checkout $TRITON_BRANCH
git checkout f9e5bf54

# Start 16-core parallel compilation with real-time output
# --no-build-isolation ensures using ninja, cmake from your current environment
if [ ! -f setup.py ]; then cd python; fi
MAX_JOBS=16 uv pip install --no-build-isolation -e .
cd ..
```

> Note: Triton dependencies are large. The `Preparing packages...` phase may download 10+GB, requiring a good network connection. Download may take several hours, while compilation takes a few minutes.

---

### 5. Build FlashAttention (ROCm Version)

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout origin/main_perf
git submodule update --init

export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_USE_HIP_DSA=1
python setup.py bdist_wheel --dist-dir=dist
uv pip install dist/*.whl
```

After building, you can run tests in the `benchmarks` directory:

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_USE_HIP_DSA=1

cd benchmarks/
python benchmark_flash_attention.py
```

---

### 6. Build vLLM (ROCm 7.0+)

#### 6.1 Install AMD SMI

```bash
# Update uv itself (optional)
uv self update

# Install AMD SMI (pointing to local path)
# uv pip install /opt/rocm/share/amd_smi

# If there are permission issues, copy the directory first:
cp -r /opt/rocm/share/amd_smi ./amdsmi_src
cd ./amdsmi_src
uv pip install .
cd ..
```

#### 6.2 Install Build Dependencies and Clone vLLM Source

```bash
uv pip install --upgrade \
  numba \
  scipy \
  "huggingface-hub[cli,hf_transfer]" \
  setuptools_scm

git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install vLLM ROCm-specific dependencies
uv pip install -r requirements/rocm.txt
uv pip install numpy setuptools wheel
```

#### 6.3 Set ROCm Architecture and Build/Install vLLM

```bash
export VLLM_TARGET_DEVICE="rocm"
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCM_HOME="/opt/rocm"

MAX_JOBS=16 uv pip install -e . --no-build-isolation
```

---

### 7. Start vLLM Model Service in Virtual Environment

```bash
# Enable FlashAttention
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_USE_HIP_DSA=1

# Start service
vllm serve ~/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf --dtype float16 --max-model-len 4096 --max-num-seqs 32 --tokenizer qwen/Qwen3-8B

# Use --enforce-eager for faster startup (slightly reduces inference speed)
vllm serve ~/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf \
  --tokenizer qwen/Qwen3-8B \
  --dtype float16 \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-model-len 4096
```

---

### 8. Complete Performance Testing Script (Auto-detect Model ID)

```bash
# 1. Auto-detect model ID (prevent 404)
MODEL_ID=$(curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[0].id')
echo "Detected model ID: $MODEL_ID"

# 2. Prepare a random prompt
RAND_PROMPT="Random code $(date +%N): Please describe the future of quantum computing in detail, at least 500 words."

# 3. Record precise start time
start=$(date +%s.%N)

# 4. Send request
response=$(curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
\"model\": \"$MODEL_ID\",
\"prompt\": \"$RAND_PROMPT\",
\"max_tokens\": 512,
\"temperature\": 0.8
}")

# 5. Record end time
end=$(date +%s.%N)

# 6. Parse content
content=$(echo "$response" | jq -r '.choices[0].text // "Error: No text content retrieved, please check the output"')
tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
duration=$(echo "$end - $start" | bc)

# 7. Print output
echo "==================== Raw Content ===================="
echo "$content"
echo "=================================================="

if (( $(echo "$duration < 0.05" | bc -l) )); then
  echo "Response too fast ($duration seconds), possibly a 404 error or cache hit."
  echo "Full response body: $response"
else
  tps=$(echo "scale=2; $tokens / $duration" | bc)
  echo "Generated tokens: $tokens"
  echo "Total elapsed time: $duration seconds"
  echo "Actual inference speed: $tps tokens/s"
fi
echo "=================================================="
```

Screenshot examples:

<div align='center'>
    <img src="../../../../public/images/01-deploy/qwen3/image13.png" alt="" width="90%">
</div>
<div align='center'>
    <img src="../../../../public/images/01-deploy/qwen3/image14.png" alt="" width="90%">
</div>
<div align='center'>
    <img src="../../../../public/images/01-deploy/qwen3/image15.png" alt="" width="90%">
</div>
