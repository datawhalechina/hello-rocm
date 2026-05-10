<div align=center>
  <h1>03-Infra</h1>
  <strong>⚙️ ROCm Operator Optimization & GPU Programming</strong>
</div>

<div align="center">

*From AMD AI hardware landscape to HIP operators and performance profiling*

[Back to Home](/)

</div>

## Introduction

&emsp;&emsp;This module is designed for developers who want to build a systematic understanding of **AMD GPU + ROCm**: from the AI hardware and ROCm software stack panorama, to the PyTorch call chain and GPU architecture, to writing **HIP** operators by hand and combining **rocBLAS / MIOpen** with performance measurement — covering the ROCm software stack, GPU programming, and operator practice with hands-on examples.

&emsp;&emsp;The content corresponds one-to-one with the **Chapter 1–4** tutorial series in this repository. The default lab environment is **Ubuntu 22.04 / 24.04 + ROCm 7.x**, with example devices including **AMD AI+ MAX395 / Radeon 8060S (gfx1151)** and others. Readers can adapt according to their own GPU and ROCm version. The directory structure is as follows:

```
03-Infra/
├── 01-embrace-amd-ai/
│   ├── README.md
│   └── images/
├── 02-decode-ai-accelerator/
│   ├── README.md
│   ├── code/
│   └── images/
├── 03-handwrite-rocm-operator/
│   ├── README.md
│   ├── code/
│   └── images/
├── 04-custom-pytorch-operator/
│   ├── README.md
│   ├── code/
│   └── images/
└── README.md
```

## Tutorial List

### Chapter 1: Embracing the New Era of AMD AI Computing

&emsp;&emsp;From Ryzen AI (NPU + GPU), Radeon discrete GPUs, to Instinct data center accelerators — an overview of AMD's AI product line and typical use cases. Explains **ROCm**'s role in the stack and includes hands-on experiments such as ResNet training and Qwen large model inference with **PyTorch**, building an overall picture of "what you can run on AMD."

- **Target Audience**: Developers and students new to AMD AI / ROCm
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 2–3 hours

📖 [Read Chapter 1](/03-infra/embrace-amd-ai)

---

### Chapter 2: Decoding AI Accelerators — From Software Stack to Hardware Architecture

&emsp;&emsp;Use tools like `ldd` to trace the **PyTorch → HIP → HSA → Driver → GPU** call chain, understand the CPU's "low latency" vs GPU's "high throughput" paradigm, and the **SIMT** execution model. Combined with concepts like **CU, LDS, and VRAM bandwidth**, learn to read how AMD GPUs behave under AI workloads.

- **Target Audience**: Developers who can already run models and want to understand the underlying stack and hardware
- **Difficulty Level**: ⭐⭐⭐
- **Estimated Time**: 3–4 hours

📖 [Read Chapter 2](/03-infra/decode-ai-accelerator)

---

### Chapter 3: Entering the World of ROCm Programming — Writing a "PyTorch Operator" by Hand

&emsp;&emsp;Introduces the correspondence between **HIP** and CUDA, Host/Device code structure. From **writing a Kernel** to reproduce Tensor addition, to using **`hipEvent`** for timing, and an initial look at **rocBLAS** and **MIOpen** — completing the transition from Python to device-level code.

- **Target Audience**: Developers with C++ basics who are ready to write or read custom operators
- **Difficulty Level**: ⭐⭐⭐
- **Estimated Time**: 2–3 hours

📖 [Read Chapter 3](/03-infra/handwrite-rocm-operator)

---

### Chapter 4: Writing Custom ROCm Operators for PyTorch

&emsp;&emsp;Starting from PyTorch's Python call chain, use the **C++ Extension** mechanism to register HIP Kernels as custom operators. Hands-on with **Fused Swish** operator, **Grid-Stride Loop** optimization, **Autograd** integration for automatic differentiation, and **memory-wall benchmarks** to quantify bandwidth bottlenecks — completing the full loop from "hand-written Kernel" to "PyTorch-callable operator."

- **Target Audience**: Developers who want to integrate custom HIP operators into PyTorch training/inference workflows
- **Difficulty Level**: ⭐⭐⭐⭐
- **Estimated Time**: 3–4 hours

📖 [Read Chapter 4](/03-infra/custom-pytorch-operator)

---

## Requirements

### Hardware Requirements

- AMD GPU (ROCm-supported GPU, such as RX 7000 / 9000 series, Ryzen AI MAX / AI 300, Instinct MI series, etc.)
- VRAM requirements vary by chapter: Chapter 1 involves ResNet / large model inference, so ensure sufficient VRAM; Chapters 3 and 4 focus on small HIP programs and matrix experiments, so a typical consumer GPU will suffice

### Software Requirements

- Operating System: Linux (Ubuntu 22.04+; tutorials use 22.04 / 24.04 as examples)
- ROCm 7.10.0 or later (chapters may also reference 7.x; verify with your local `rocm-smi` / release notes)
- Chapter 1: Python 3.10+, PyTorch (ROCm build)
- Chapters 3 & 4: Recommended to install **ROCm HIP development components** (e.g., `hipcc`, `rocm-dev`, etc. — package names may vary by distribution), CMake 3.16+, GCC 9+ or Clang 12+

## FAQ

<details>
<summary>Q: How do I check if my AMD GPU supports ROCm?</summary>

Please refer to the [ROCm Official Support List](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported GPU models and distribution combinations.

</details>

<details>
<summary>Q: What is the relationship between Chapters 3 & 4 and Chapters 1 & 2?</summary>

Chapters 1 & 2 focus on the **big picture and stack/hardware understanding**; Chapters 3 & 4 focus on **HIP programming and operator-level practice**. The chapter titles are closely related: you can read Chapter 3 first to build a minimal end-to-end workflow, then read Chapter 4 for mapping and performance deep-dives; if time is limited, you can also selectively read based on your background.

</details>

<details>
<summary>Q: Getting "header not found" or "linking library not found" errors when compiling HIP programs?</summary>

1. Confirm the corresponding ROCm development packages are installed and `hipcc` is in your `PATH`
2. Check that `HIP_PATH`, `ROCM_PATH`, and other environment variables match the installation path
3. Refer to the [HIP Installation & Getting Started](https://rocm.docs.amd.com/projects/HIP/en/latest/) guide to verify dependencies

</details>

## Reference Resources

- [ROCm Official Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [rocBLAS Documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [PyTorch for ROCm](https://pytorch.org/get-started/locally/)

---

<div align="center">

**Contributions of more operator optimization and GPU programming practice content are welcome!** 🎉

[Submit an Issue](https://github.com/datawhalechina/hello-rocm/issues) | [Submit a PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
