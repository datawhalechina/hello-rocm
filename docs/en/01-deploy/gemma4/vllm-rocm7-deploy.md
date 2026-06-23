## vLLM Deployment on ROCm (Ubuntu 24.04 + ROCm 7+)

This guide shows how to deploy and call **Gemma 4** with **vLLM** on ROCm.

For ROCm 7.13 / gfx1151, there are two practical Docker routes:

- AMD ROCm validated image: `rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`
- Upstream vLLM image: `vllm/vllm-openai-rocm:latest`

> Prerequisite: complete [ROCm 7.13 environment setup](./env-prepare-ubuntu24-rocm7.md). Gemma models may require accepting the license on Hugging Face and passing `HF_TOKEN`.

---

## 1. ROCm 7.13 Validated Docker Image (gfx1151)

```bash
docker pull rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

> This image includes PyTorch 2.10.0 + vLLM 0.19.1. PyTorch 2.11.0 belongs to the pip installation path.

```bash
docker run -it --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --network=host \
  --ipc=host \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ~/models:/app/models \
  -e HF_HOME="/app/models" \
  rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  bash
```

Inside the container:

```bash
vllm serve google/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --max-model-len 4096
```

---

## 2. Upstream vLLM ROCm Image

```bash
docker pull vllm/vllm-openai-rocm:latest
```

```bash
export HF_TOKEN="hf_***"

docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model google/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --max-model-len 4096
```

---

## 3. API Calls

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "Introduce ROCm in one sentence."}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
response = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=[{"role": "user", "content": "Introduce deep learning in one sentence."}],
)
print(response.choices[0].message.content)
```

---

## 4. ROCm Wheel Path (Optional)

ROCm wheels are stricter about Python / ROCm / glibc versions. For the official vLLM ROCm wheel path, use Python 3.12 and the vLLM ROCm wheel index. For ROCm 7.13 AMD framework wheels, follow the AMD documentation for the exact gfx1151 wheel URLs.

---

## 5. Troubleshooting

- If Gemma downloads fail, accept the model license on Hugging Face and pass `HF_TOKEN`.
- If the container cannot see the GPU, verify `/dev/kfd` and `/dev/dri` on the host.
- If memory is insufficient, reduce `--max-model-len`, for example from 4096 to 2048.
