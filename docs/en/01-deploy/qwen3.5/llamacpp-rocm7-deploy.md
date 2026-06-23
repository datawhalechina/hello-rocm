## llama.cpp Deployment for Qwen3.5 (Ubuntu 24.04 + ROCm 7+)

This guide shows how to run Qwen3.5 GGUF models with **llama.cpp** on ROCm.

> Prerequisite: complete [ROCm 7.13 environment setup](./env-prepare-ubuntu24-rocm7.md).

### 1. Download a ROCm llama.cpp Build

Lemonade provides prebuilt ROCm builds:

- `370` corresponds to `gfx1150`
- `395` corresponds to `gfx1151`

Links:

- https://github.com/lemonade-sdk/llamacpp-rocm
- https://github.com/lemonade-sdk/llamacpp-rocm/releases

<div align='center'>
    <img src="../../../public/images/01-deploy/qwen3/image7.png" alt="llama.cpp ROCm releases" width="90%">
</div>

### 2. Verify ROCm

```bash
amd-smi
```

<div align='center'>
    <img src="../../../public/images/01-deploy/qwen3/image8.png" alt="amd-smi output" width="90%">
</div>

### 3. Prepare the Runtime

```bash
cd llama-*x64/
sudo chmod +x *
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### 4. Download a Qwen3.5 GGUF Model

Download a Qwen3.5 GGUF quantized model from Hugging Face or another trusted source.

```bash
mkdir -p ~/models/qwen3.5
cd ~/models/qwen3.5
wget https://huggingface.co/Manojb/Qwen3.5-4B-UD-Q4_K_XL.gguf/blob/main/Qwen3.5-4B-UD-Q4_K_XL.gguf
```

### 5. Start llama-server

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
cd llama-*x64/

./llama-server \
  -m ~/models/qwen3.5/qwen3.5-4b-q4_k_m.gguf \
  -ngl 99
```

### 6. Test the API

```bash
curl -s -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen3.5-4b-q4_k_m",
  "prompt": "Explain large language models in one sentence.",
  "max_tokens": 128
}' | jq -r '.choices[0].text'
```

