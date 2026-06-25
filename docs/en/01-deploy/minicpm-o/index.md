<div align=center>
  <h1>MiniCPM-o 4.5 Deployment</h1>
  <strong>🎙️ Running a Full-Omni Model on AMD GPU</strong>
</div>

<div align="center">

*Audio · Vision · TTS · Full-Duplex · ROCm 7+ · llama.cpp-omni*

[Home](/) · [中文](/zh/01-deploy/minicpm-o/) · [Deploy Overview](/01-deploy/)

</div>

## Overview

&emsp;&emsp;**MiniCPM-o 4.5** is an end-side omni-modal model from OpenBMB that combines **text, voice input (audio encoder), image understanding (vision encoder), and voice output (TTS)** into a single model. Its Omni full-duplex mode enables real-time, phone-call-like conversations — all running on AMD ROCm without an NVIDIA GPU.

&emsp;&emsp;This module is based on OpenBMB's official [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) inference engine and [`OpenBMB/MiniCPM-o-Demo`](https://github.com/OpenBMB/MiniCPM-o-Demo) (`Comni` branch), verified on **AMD Ryzen AI MAX+ 395 (gfx1151 / Strix Halo APU)** and applicable to other ROCm-supported AMD GPUs.

```
01-Deploy/minicpm-o/
├── minicpm-o-model.md               # MiniCPM-o 4.5 model introduction
├── llamacpp-omni-rocm7-deploy.md    # llama.cpp-omni CLI deployment
└── webdemo-rocm7-deploy.md          # MiniCPM-o Web Demo full-duplex deployment
```

---

## Tutorial List

### MiniCPM-o 4.5 Model Introduction

&emsp;&emsp;Understand the omni-modal architecture of MiniCPM-o 4.5 — audio input, visual encoding, TTS synthesis, and full-duplex design. After reading, you'll know exactly which GGUF sub-model files are required and the VRAM requirements for different GPUs.

- **Audience**: Readers who want to understand the model before deploying
- **Difficulty**: ⭐
- **Estimated time**: 15 minutes

📖 [Read MiniCPM-o 4.5 Model Introduction](./minicpm-o-model.md)

---

### llama.cpp-omni CLI Deployment

&emsp;&emsp;Build and run MiniCPM-o 4.5 on AMD GPU using OpenBMB's official llama.cpp-omni inference engine. This tutorial covers HIP compilation, GGUF model download, audio input testing, and TTS voice output — letting you run full omni-modal inference from the command line.

- **Audience**: Developers who want to verify inference capabilities or prepare a backend for the Web Demo
- **Difficulty**: ⭐⭐⭐
- **Estimated time**: 2 hours (including model download)

📖 [Start llama.cpp-omni Deployment Tutorial](./llamacpp-omni-rocm7-deploy.md)

---

### MiniCPM-o Web Demo Full-Duplex Deployment

&emsp;&emsp;Deploy the complete Web Demo using `OpenBMB/MiniCPM-o-Demo` (`Comni` branch) on AMD GPU. Supports **Turn-based / Half-Duplex / Omni full-duplex / Audio full-duplex** — experience real-time voice and video conversation with the model directly in your browser. Prerequisite: llama.cpp-omni must already be compiled.

- **Audience**: Developers who want to build a full interactive demonstration
- **Difficulty**: ⭐⭐⭐
- **Estimated time**: 1 hour (with inference backend already ready)

📖 [Start Web Demo Full-Duplex Deployment](./webdemo-rocm7-deploy.md)

---

## Requirements

### Hardware

| Scenario | Minimum | Recommended |
|----------|---------|-------------|
| Text inference (LLM only) | 8 GB VRAM | — |
| Voice input (audio encoder) | 12 GB VRAM | — |
| Full omni (LLM + vision + audio + TTS) | **16 GB VRAM** | 64 GB unified memory (Strix Halo APU) |

> AMD Ryzen AI MAX+ 395 / 890M with unified memory architecture is ideal for MiniCPM-o — 64 GB unified memory can fully load all GGUF sub-models (~8.3 GB) into the GPU.

### Software

- OS: Linux (Ubuntu 22.04 / 24.04)
- ROCm 7.10.0 or later (system installation)
- CMake 3.21+, GCC / Clang (for HIP compilation)
- Python 3.10+ (Web Demo dependency)

### Test Environment for This Tutorial

```
Hardware:  AMD Ryzen AI MAX+ PRO 395 / Radeon™ 890M (Strix Halo)
GPU arch:  gfx1151
Memory:    64 GB unified (all usable as VRAM)
ROCm:      7.12.0 (system) + TheRock 7.12.0a alpha (Tensile fix)
OS:        Ubuntu 24.04
```

---

## FAQ

<details>
<summary>Q: How is MiniCPM-o different from a regular LLM, and why can't I use vLLM / Ollama?</summary>

MiniCPM-o 4.5 includes independent sub-modules — an audio encoder, a vision encoder, and a TTS token2wav pipeline — that mainstream inference frameworks don't natively support yet. llama.cpp-omni is a specialized inference engine fork that can load and schedule all these sub-models concurrently.

</details>

<details>
<summary>Q: Does gfx1151 (Strix Halo) require special handling?</summary>

Yes. gfx1151 was introduced in late 2025. The system `/opt/rocm` rocBLAS Tensile library doesn't include complete GEMM kernels for gfx1151, causing a crash at runtime (`hipErrorInvalidImage`). The fix is to install the TheRock 7.12.0a alpha SDK and point the runtime at its rocBLAS library directory — see the [llama.cpp-omni tutorial](./llamacpp-omni-rocm7-deploy.md) for details.

Other AMD GPUs (gfx1100 / RX 7900 XTX, gfx1150 / RX 9070 XT, etc.) are not affected.

</details>

<details>
<summary>Q: How do I find my AMD GPU architecture (gfx number)?</summary>

```bash
rocminfo | grep -i "gfx"
# or
amd-smi
```

</details>

---

## References

- [MiniCPM-o Official Repository (OpenBMB)](https://github.com/OpenBMB/MiniCPM-o)
- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [MiniCPM-o-Demo Official Repository](https://github.com/OpenBMB/MiniCPM-o-Demo)
- [MiniCPM-o-4_5-gguf (ModelScope)](https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf)
- [ROCm Official Documentation](https://rocm.docs.amd.com/)
- [TheRock Nightly SDK (gfx1151 fix)](https://rocm.nightlies.amd.com/v2/gfx1151/)

---

<div align="center">

**Contributions welcome!** 🎉

[Submit Issue](https://github.com/datawhalechina/hello-rocm/issues) | [Submit PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
