## vLLM Deployment on ROCm (Ubuntu 24.04 + ROCm 7+)

This guide shows how to deploy and call **Qwen3** with **vLLM** on ROCm.

For ROCm 10.0.0, there are two practical Docker routes:

- AMD ROCm validated image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- Upstream vLLM image: `vllm/vllm-openai-rocm:latest`

> Prerequisite: complete [ROCm 10.0.0 environment setup](./env-prepare-ubuntu24-rocm7.md).

---

## 1. ROCm 10.0.0 Validated Docker Image

```bash
docker pull rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
```

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
  rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
  bash
```

Inside the container:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --max-model-len 4096
```

---

## 2. Upstream vLLM ROCm Image

```bash
docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model Qwen/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --max-model-len 4096
```

---

## 3. API Calls

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Introduce ROCm in one sentence."}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "Introduce deep learning in one sentence."}],
)
print(response.choices[0].message.content)
```

---

## 4. ROCm Wheel Path (Optional)

ROCm wheels are stricter about Python / ROCm / glibc versions. Use Docker first unless there is a specific reason to manage vLLM in a local Python environment.
