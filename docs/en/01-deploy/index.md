<div align=center>
  <h1>01-Deploy</h1>
  <strong>🚀 ROCm LLM Deployment in Practice</strong>
</div>

<div align="center">

*Get started with LLM deployment on AMD GPUs from scratch*

[Back to Home](/en/) · [中文](/zh/01-deploy/)

</div>

## Introduction

&emsp;&emsp;This module provides comprehensive tutorials for deploying large language models on AMD GPUs. Whether you are a beginner or an experienced developer, you can quickly learn how to deploy and run LLMs on the ROCm platform through these tutorials.

&emsp;&emsp;Since ROCm 7.10.0, ROCm supports seamless installation in Python virtual environments just like CUDA, significantly lowering the barrier for LLM deployment on AMD GPUs.

&emsp;&emsp;This module uses **Google Gemma 4** (`gemma-4-E4B-it` primarily) as the example model by default, and also provides parallel tutorials for **Qwen3** as a reference. The directory structure is as follows:

```
01-Deploy/
└── models/
    ├── Gemma4/           # Deployment tutorials with Gemma 4 as the primary model (recommended)
    ├── Qwen3/            # Qwen3 series deployment tutorials (reference/comparison)
    └── Qwen3.5/          # Qwen3.5 series deployment tutorials (new architecture reference)
```

## Tutorial List

### Ubuntu 24.04 + ROCm 7 Environment Setup Tutorial

&emsp;&emsp;This tutorial walks you through installing and verifying ROCm 7.1.0 on Ubuntu 24.04 step by step, including uninstalling old ROCm environments, running the official script to install ROCm, and using tools like `rocminfo` / `rocm-smi` / `amd-smi` to confirm GPU and driver status. It is recommended to complete this tutorial before starting any deployment tutorial.

- **Target Audience**: Users setting up a ROCm environment on an AMD GPU for the first time
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1 hour

📖 [Start the Environment Setup Tutorial (Gemma4)](/en/01-deploy/gemma4/env-prepare-ubuntu24-rocm7.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/env-prepare-ubuntu24-rocm7.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/env-prepare-ubuntu24-rocm7.md)

---

### Gemma 4 Model Introduction

&emsp;&emsp;Before starting deployment, it is recommended to read the Gemma 4 model introduction to understand the architecture characteristics, capability differences, and hardware selection recommendations for the four versions: Gemma 4 E2B / E4B / 31B / 26B A4B, so you can choose the right model for your environment.

- **Target Audience**: Users trying Gemma 4 for the first time
- **Difficulty Level**: ⭐
- **Estimated Time**: 15 minutes

📖 [Read the Gemma 4 Model Introduction](/en/01-deploy/gemma4/gemma4_model.md)

---

### LM Studio LLM Deployment from Scratch

&emsp;&emsp;LM Studio is a user-friendly desktop application that supports running large language models locally. This tutorial uses **Gemma 4 E4B-it Q4_K_M** as an example to guide you through deploying and running LLMs on AMD GPUs using LM Studio with the ROCm version of the llama.cpp backend.

- **Target Audience**: Beginners and users who want to quickly experience LLMs
- **Difficulty Level**: ⭐
- **Estimated Time**: 30 minutes

📖 [Start the LM Studio Deployment Tutorial (Gemma4)](/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/lm-studio-rocm7-deploy.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/lm-studio-rocm7-deploy.md)

---

### vLLM LLM Deployment from Scratch

&emsp;&emsp;vLLM is a high-performance LLM inference and serving framework that supports efficient PagedAttention and continuous batching. This tutorial uses **Gemma 4 E4B-it** as an example, covering both a quick start method using the official ROCm vLLM Docker image, and an advanced method for manually compiling Triton / FlashAttention / vLLM from source.

- **Target Audience**: Developers who need to set up inference services
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1 hour

📖 [Start the vLLM Deployment Tutorial (Gemma4)](/en/01-deploy/gemma4/vllm-rocm7-deploy.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/vllm-rocm7-deploy.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/vllm-rocm7-deploy.md)

---

### Ollama LLM Deployment from Scratch

&emsp;&emsp;Ollama is a framework for quickly serving large language models and vision-language models with an efficient backend runtime. This tutorial uses **Gemma 4 E4B-it Q4_K_M** as an example to guide you through deploying LLMs on AMD GPUs using Ollama (ROCm version llama.cpp backend), with tokens/s benchmark examples included.

- **Target Audience**: Developers who want to spin up a local inference service with a single command
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1 hour

📖 [Start the Ollama Deployment Tutorial (Gemma4)](/en/01-deploy/gemma4/ollama-rocm7-deploy.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/ollama-rocm7-deploy.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/ollama-rocm7-deploy.md)

---

### llama.cpp LLM Deployment from Scratch

&emsp;&emsp;llama.cpp is a lightweight and high-performance inference backend that supports multiple model formats including GGUF, with optimized versions available for ROCm. This tutorial uses **Gemma 4 E4B-it Q4_K_M (GGUF)** as an example, showing how to deploy mainstream models on Ubuntu 24.04 + ROCm 7+ using both pre-built binaries and Docker.

- **Target Audience**: Developers who want to freely orchestrate inference workflows via CLI / REST API
- **Difficulty Level**: ⭐⭐⭐
- **Estimated Time**: 1.5 hours

📖 [Start the llama.cpp Deployment Tutorial (Gemma4)](/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/llamacpp-rocm7-deploy.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/llamacpp-rocm7-deploy.md)

---

## Requirements

### Hardware Requirements

- AMD GPU (ROCm-supported GPUs such as RX 7000 / 9000 series, Ryzen AI MAX / AI 300, Instinct MI series, etc.)
- At least 8GB VRAM recommended (Gemma 4 E4B Q4_K_M quantized version can run with 8GB VRAM; for native bfloat16 inference or larger models, please refer to the VRAM recommendations in the corresponding tutorials)

### Software Requirements

- Operating System: Linux (Ubuntu 22.04+) or Windows 11
- ROCm 7.10.0 or higher
- Python 3.10+

## FAQ

<details>
<summary>Q: How do I check if my AMD GPU supports ROCm?</summary>

Please refer to the [ROCm official support list](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) to check supported GPU models.

</details>

<details>
<summary>Q: What should I do if I encounter a "HIP error" during deployment?</summary>

1. Confirm that ROCm is properly installed
2. Check that environment variables are correctly set
3. Try restarting the system and running again

</details>

<details>
<summary>Q: I get a permission denied error when downloading Gemma 4?</summary>

Gemma series models require you to first click **Agree & Access** on the corresponding model page on Hugging Face (e.g., <a href="https://huggingface.co/google/gemma-4-E4B-it">google/gemma-4-E4B-it</a>), then log in with a Hugging Face Token that has `read` permissions or inject it via `HF_TOKEN` into the container / process.

</details>

## Reference Resources

- [ROCm Official Documentation](https://rocm.docs.amd.com/)
- [vLLM Official Documentation](https://docs.vllm.ai/)
- [Ollama Official Documentation](https://docs.ollama.com/)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [Hugging Face Gemma 4 Model Collection](https://huggingface.co/collections/google/gemma-4)

---

<div align="center">

**Contributions for more deployment tutorials are welcome!** 🎉

[Submit an Issue](https://github.com/datawhalechina/hello-rocm/issues) | [Submit a PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
