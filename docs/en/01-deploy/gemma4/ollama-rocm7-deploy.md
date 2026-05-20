## Ollama Deployment from Scratch (Ubuntu 24.04 + ROCm 7+)

This section explains how to install and use **Ollama (ROCm version llama.cpp backend)** on Ubuntu 24.04 + ROCm 7+, with performance testing using Gemma 4 E4B-it Q4_K_M as an example.

> Prerequisite: ROCm 7.1.0 environment setup is complete (refer to `env-prepare-ubuntu24-rocm7.md`).

---

### 1. Install Ollama (System Service)

Step one: One-click installation managed via `systemctl`, which starts the service on local port `11434`.  
For more information, refer to the official documentation: https://docs.ollama.com/linux

Installation command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

### 2. Verify Service Connectivity

After installation, verify the service is running properly:

```bash
curl http://localhost:11434
```

If JSON information is returned (such as version number, etc.), the service has started successfully.

---

### 3. Basic Usage Commands

Common basic commands:

```bash
# List all models
ollama list

# Download a model (using Gemma 4 E4B-it Q4_K_M as an example)
ollama pull gemma4:e4b-it-q4_K_M

# Test model execution (interactive)
ollama run gemma4:e4b-it-q4_K_M
```

> Note: Gemma 4 tag naming in the Ollama official library may change with upstream updates. Please check [ollama.com/library/gemma4](https://ollama.com/library/gemma4) for the latest tags before use. If you have sufficient VRAM, you can also switch to larger variants like `gemma4:31b` or `gemma4:26b-a4b`.

---

### 4. Benchmarking with curl (Calculating tokens/s)

The following example command calls Ollama's REST API and uses `jq` to parse the evaluation information from the response to calculate inference speed:

```bash
curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
  "model": "gemma4:e4b-it-q4_K_M",
  "prompt": "Explain what a large language model is in one sentence",
  "stream": false
}' | jq '.eval_count, .eval_duration' | \
awk 'NR==1{count=$1} NR==2{duration=$1/1e9} END{printf "tokens/s: %.2f\n", count/duration}'
```

- `eval_count`: Number of tokens generated during inference  
- `eval_duration`: Inference time (in nanoseconds)  
- `tokens/s`: Calculated as `count / (duration / 1e9)`

---

### 5. Gemma 4 E4B-it Q4_K_M Performance Example

Testing the **Gemma 4 E4B-it Q4_K_M** model in the above environment (context length 4096):

- **tokens/s depends on your actual hardware** (Gemma 4 E4B activates only 4.5B effective parameters during inference, typically faster than comparable 8B models at the same Q4_K_M quantization)

Screenshot example:

<div align='center'>
    <img src="../../../../public/images/01-deploy/gemma4/image6.png" alt="" width="90%">
</div>

> To experience Gemma 4's multimodal capabilities (image / video / audio input), select a Gemma 4 tag in Ollama that is labeled as supporting `vision` / `multimodal`, and pass images via the `images` field (Base64-encoded) in the `/api/chat` API endpoint. Please refer to the latest Ollama documentation for specific parameters.
