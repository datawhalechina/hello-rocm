## Ollama Deployment for Qwen3.5 (Ubuntu 24.04 + ROCm 7+)

This guide shows how to use **Ollama** with the ROCm llama.cpp backend to run Qwen3.5 models.

> Prerequisite: complete [ROCm 7.13 environment setup](./env-prepare-ubuntu24-rocm7.md).

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the service:

```bash
curl http://localhost:11434
```

### 2. Prepare a Qwen3.5 Model

If Ollama provides an official Qwen3.5 tag (https://ollama.com/library/qwen3.5), use that tag directly:

```bash
ollama pull qwen3.5:4b
ollama run qwen3.5:4b
```

If there is no suitable official tag, create a local model from a GGUF file:

```bash
mkdir -p ~/models/qwen3.5
cd ~/models/qwen3.5
```

Create a `Modelfile`:

```text
FROM ./qwen3.5-4b-q4_k_m.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.8
```

Create and run the model:

```bash
ollama create qwen3.5-4b-local -f Modelfile
ollama run qwen3.5-4b-local
```

### 3. Measure tokens/s

```bash
curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen3.5-4b-local",
  "prompt": "Explain large language models in one sentence.",
  "stream": false
}' | jq '.eval_count, .eval_duration' | \
awk 'NR==1{count=$1} NR==2{duration=$1/1e9} END{printf "tokens/s: %.2f\n", count/duration}'
```

