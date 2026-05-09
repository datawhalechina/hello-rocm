<div align=center>
  <h1>01-Deploy</h1>
  <strong>🚀 ROCm LLM deployment tutorials</strong>
</div>

<div align="center">

*Get LLMs running on AMD GPUs with ROCm—from zero to deployed*

[Back to project home](../README_en.md) · [简体中文](./README.md)

</div>

## Overview

&emsp;&emsp;This section is a complete guide to deploying large language models on **AMD GPUs** with **ROCm**. Whether you are new to ROCm or already ship models in production, you can follow these tutorials to install, validate, and serve models on AMD hardware.

&emsp;&emsp;Since **ROCm 7.10.0**, ROCm can be installed into Python virtual environments in a CUDA-like workflow, which lowers the barrier to LLM deployment on AMD GPUs.

&emsp;&emsp;Tutorials primarily use **Google Gemma 4** (`gemma-4-E4B-it` as the main example), with **parallel Qwen3** guides for comparison. Layout:

```
01-Deploy/
└── models/
    ├── Gemma4/           # Gemma 4–centric guides (recommended)
    └── Qwen3/            # Qwen3 deployment guides (reference)
```

## Tutorials

### Ubuntu 24.04 + ROCm 7 — environment setup

&emsp;&emsp;Step-by-step setup on **Ubuntu 24.04**: remove old ROCm if needed, install ROCm 7.1.x with the official flow, and verify the stack with `rocminfo`, `rocm-smi`, `amd-smi`, etc. **Complete this before** the framework-specific deployment guides.

- **Audience**: First-time ROCm users on AMD GPUs  
- **Level**: ⭐⭐  
- **Time**: ~1 hour  

📖 [Environment setup (Gemma4)](./models/Gemma4/env-prepare-ubuntu24-rocm7.md)  
📎 Also see: [Qwen3 variant](./models/Qwen3/env-prepare-ubuntu24-rocm7.md)

---

### Gemma 4 model overview

&emsp;&emsp;Before you deploy, read the Gemma 4 introduction for **E2B / E4B / 31B / 26B A4B**—architecture, capability differences, and hardware hints so you can pick a variant that fits your GPU.

- **Audience**: New to Gemma 4  
- **Level**: ⭐  
- **Time**: ~15 minutes  

📖 [Gemma 4 model overview](./models/Gemma4/gemma4_model.md)

---

### LM Studio — LLM deployment from scratch

&emsp;&emsp;**LM Studio** is a desktop app for running LLMs locally. The guide uses **Gemma 4 E4B-it Q4_K_M** and LM Studio with the **ROCm build of llama.cpp** on AMD GPUs.

- **Audience**: Beginners who want a GUI-first path  
- **Level**: ⭐  
- **Time**: ~30 minutes  

📖 [LM Studio tutorial (Gemma4)](./models/Gemma4/lm-studio-rocm7-deploy.md)  
📎 Also see: [Qwen3 version](./models/Qwen3/lm-studio-rocm7-deploy.md)

---

### vLLM — LLM deployment from scratch

&emsp;&emsp;**vLLM** is a high-performance inference and serving stack (PagedAttention, continuous batching, etc.). The guide uses **Gemma 4 E4B-it**, covers the **official ROCm vLLM Docker** quick path, and an **advanced** path (build Triton / FlashAttention / vLLM from source).

- **Audience**: Developers who need a serving stack  
- **Level**: ⭐⭐  
- **Time**: ~1 hour  

📖 [vLLM tutorial (Gemma4)](./models/Gemma4/vllm-rocm7-deploy.md)  
📎 Also see: [Qwen3 version](./models/Qwen3/vllm-rocm7-deploy.md)

---

### Ollama — LLM deployment from scratch

&emsp;&emsp;**Ollama** runs LLMs and VLMs with a simple CLI/API and an efficient runtime. The guide uses **Gemma 4 E4B-it Q4_K_M** on AMD GPUs (ROCm **llama.cpp** backend) and includes a **tokens/s** benchmark example.

- **Audience**: Developers who want a one-command local server  
- **Level**: ⭐⭐  
- **Time**: ~1 hour  

📖 [Ollama tutorial (Gemma4)](./models/Gemma4/ollama-rocm7-deploy.md)  
📎 Also see: [Qwen3 version](./models/Qwen3/ollama-rocm7-deploy.md)

---

### llama.cpp — LLM deployment from scratch

&emsp;&emsp;**llama.cpp** is a lightweight, high-performance backend (GGUF and more) with ROCm-optimized builds. The guide uses **Gemma 4 E4B-it Q4_K_M (GGUF)** on **Ubuntu 24.04 + ROCm 7+** via **prebuilt binaries** and **Docker**.

- **Audience**: Developers who prefer CLI / REST and full control  
- **Level**: ⭐⭐⭐  
- **Time**: ~1.5 hours  

📖 [llama.cpp tutorial (Gemma4)](./models/Gemma4/llamacpp-rocm7-deploy.md)  
📎 Also see: [Qwen3 version](./models/Qwen3/llamacpp-rocm7-deploy.md)

---

## Requirements

### Hardware

- AMD GPU supported by ROCm (e.g. RX 7000 / 9000, Ryzen AI MAX / AI 300, Instinct MI series)  
- **≥8 GB VRAM** recommended (Gemma 4 E4B **Q4_K_M** fits in ~8 GB; for native **bfloat16** or larger models, follow each guide’s VRAM notes)

### Software

- OS: **Linux (Ubuntu 22.04+)** or **Windows 11**  
- **ROCm 7.10.0** or newer  
- **Python 3.10+**

## FAQ

<details>
<summary>Q: How do I know if my AMD GPU is supported by ROCm?</summary>

See the [ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported GPUs.

</details>

<details>
<summary>Q: I hit a “HIP error” while deploying—what should I check?</summary>

1. Confirm ROCm installed cleanly and matches your driver stack  
2. Verify environment variables (`PATH`, `LD_LIBRARY_PATH`, HIP/ROCm paths)  
3. Reboot and retry if the stack was just upgraded  

</details>

<details>
<summary>Q: Hugging Face says I have no access to download Gemma 4?</summary>

For Gemma weights, open the model page (e.g. <a href="https://huggingface.co/google/gemma-4-E4B-it">google/gemma-4-E4B-it</a>), click **Agree & Access**, then authenticate with a Hugging Face token that has **read** access (CLI login or `HF_TOKEN` in your shell/container).

</details>

## References

- [ROCm documentation](https://rocm.docs.amd.com/)  
- [vLLM documentation](https://docs.vllm.ai/)  
- [Ollama documentation](https://docs.ollama.com/)  
- [llama.cpp repository](https://github.com/ggerganov/llama.cpp)  
- [Hugging Face Gemma 4 collection](https://huggingface.co/collections/google/gemma-4)

---

<div align="center">

**Contributions welcome—more deployment guides appreciated!** 🎉

[Open an issue](https://github.com/datawhalechina/hello-rocm/issues) | [Open a PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
