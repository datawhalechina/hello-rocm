## Qwen3.5-4B vLLM Deployment on ROCm (Ubuntu 24.04 + ROCm 7+)

This guide shows how to serve and call **Qwen3.5-4B** with **vLLM** on ROCm.

For ROCm 7.13 / gfx1151, there are two practical Docker routes:

- AMD ROCm validated image: `rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`
- Upstream vLLM image: `vllm/vllm-openai-rocm:latest`

> Prerequisite: complete [ROCm 7.13 environment setup](./env-prepare-ubuntu24-rocm7.md). Qwen3.5 is a newer model family, so prefer recent vLLM / Transformers versions.

---

## 1. ROCm 7.13 Validated Docker Image (gfx1151)

```bash
docker pull rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
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
  rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  bash
```

Inside the container:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --max-model-len 4096 \
  --trust-remote-code
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
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model Qwen/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --max-model-len 4096 \
  --trust-remote-code
```

---

## 3. API Calls and Thinking Mode

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Introduce ROCm in one sentence."}],
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 256,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}
  }'
```

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
response = client.chat.completions.create(
    model="Qwen3.5-4B",
    messages=[{"role": "user", "content": "Introduce deep learning in one sentence."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

Use `enable_thinking=false` for short demos and role-play; use `true` for reasoning-heavy tasks.
