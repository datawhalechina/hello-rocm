<div align=center>
  <h1>04-References</h1>
  <strong>📚 Curated ROCm Resources</strong>
</div>

<div align="center">

*Selected official and community resources for AMD ROCm*

[Back to Home](/)

</div>

## Introduction

&emsp;&emsp;This section collects high-quality learning resources related to ROCm and AMD GPUs, including official documentation, community tutorials, technical blogs, and relevant news. Use it to quickly find the references you need.

## hello-rocm Skill

The hello-rocm Skill is the AI-assistant navigation layer built into this project. It exposes the project’s learning path, reference index, GPU architecture table, deployment tutorials, and troubleshooting checklist to AI coding tools that support Skills, Rules, or Agent configuration.

| If you ask | The Skill indexes |
|-----------|-------------------|
| Which architecture / gfx target does my GPU use? | `docs/en/00-environment/rocm-gpu-architecture-table.md` |
| What is the fastest path to run my first model? | `src/hello-rocm-skill/references/quick-deploy/SKILL.md` |
| How do I install PyTorch / vLLM / Ollama / llama.cpp on ROCm? | The “Frameworks and Inference Services” table on this page |
| How do I debug ROCm / PyTorch / HIP errors? | `src/hello-rocm-skill/references/troubleshooting/SKILL.md` |
| Which chapter should I read first? | README and chapter `index.md` files |

### Copy-and-use Skill prompt

Copy the sentence below into your AI coding tool and let it decide how to load the Skill through its Skills, Rules, or Agent configuration system:

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

If you prefer manual installation, copy the Skill to the matching directory for your tool:

::: code-group

```bash [Claude Code]
mkdir -p .claude/skills
cp -r src/hello-rocm-skill .claude/skills/hello-rocm
```

```bash [Cursor]
mkdir -p .cursor/skills
cp -r src/hello-rocm-skill .cursor/skills/hello-rocm
```

```bash [Generic Agent]
mkdir -p .agents/skills
cp -r src/hello-rocm-skill .agents/skills/hello-rocm
```

:::

Then start a new conversation and try:

```text
Load the hello-rocm skill and help me choose the right ROCm tutorial for my AMD GPU.
```

For troubleshooting and FAQs, you can also join the [Feishu community discussion](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO).

## Official Resources

### AMD Official Documentation

| Resource | Description | Link |
|----------|-------------|------|
| ROCm Documentation | Official ROCm platform docs | [rocm.docs.amd.com](https://rocm.docs.amd.com/) |
| ROCm Release Notes | Release notes for each version | [Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) |
| HIP Programming Guide | HIP API and programming guide | [HIP Docs](https://rocm.docs.amd.com/projects/HIP/en/latest/) |
| AMD GitHub | AMD open-source repositories | [github.com/amd](https://github.com/amd) |
| ROCm GitHub | ROCm project repositories | [github.com/ROCm](https://github.com/ROCm) |

### AMD GPU / APU Architecture and Official Technical Resources

> hello-rocm focuses on local learning, Radeon GPUs, and Ryzen AI PC scenarios, so this table prioritizes RDNA / Ryzen / Radeon resources. CDNA / Instinct resources are kept in the lower half for data center GPU comparison. Not every generation has a traditional whitepaper; for Radeon / Ryzen, AMD often publishes ISA references, ROCm support docs, and official technical articles instead.

| Architecture / resource | Focus | Architecture overview / support entry | Whitepaper / official technical resource |
|--------------|-------|-----------------------|--------------------------------|
| AMD RDNA 4 Architecture | Radeon RX 9000 / Radeon AI PRO R9000 series, latest consumer and workstation GPUs | [AMD RDNA Architecture](https://www.amd.com/en/technologies/rdna.html) | [RDNA 4 ISA Reference Guide](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture) |
| AMD RDNA 3.5 Architecture | Ryzen AI Max / AI 300 / AI 400 APU integrated GPUs (`gfx1150` / `gfx1151` / `gfx1152`) | [AMD RDNA3.5 system optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html) | [RDNA 3.5 ISA Reference Guide](https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture) |
| AMD RDNA 3 Architecture | Radeon RX 7000 / Radeon PRO W7000 series and some Ryzen APUs | [AMD RDNA Architecture](https://www.amd.com/en/technologies/rdna.html) | [RDNA 3 ISA Reference Guide](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0) |
| AMD RDNA 2 Architecture | Radeon RX 6000 / Radeon PRO W6000 series | [AMD RDNA Architecture](https://www.amd.com/en/technologies/rdna.html) | [RDNA 2 Explained: Radeon PRO W6000](https://www.amd.com/content/dam/amd/en/documents/products/graphics/workstation/rdna2-explained-radeon-pro-W6000.pdf) / [RDNA 2 ISA Reference Guide](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture) |
| AMD RDNA Architecture | Radeon RX 5000 / Radeon PRO W5000 series, foundational RDNA architecture | [AMD RDNA Architecture](https://www.amd.com/en/technologies/rdna.html) | [GPUOpen AMD GPU architecture programming documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation/) |
| ROCm on Radeon and Ryzen | ROCm + PyTorch support entry for Radeon RX / Radeon PRO / Ryzen APUs | [Use ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html) | [GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) |
| AMD XDNA / Ryzen AI NPU | Ryzen AI on-device NPU inference, distinct from ROCm/HIP GPU compute | [AMD XDNA Architecture](https://www.amd.com/en/technologies/xdna.html) | [AMD Ryzen AI Software](https://www.amd.com/en/developer/resources/ryzen-ai-software.html) / [Ryzen AI Docs PDF](https://ryzenai.docs.amd.com/_/downloads/en/latest/pdf/) |
| AMD CDNA 4 Architecture | Instinct MI350 series and next-generation AI compute acceleration | [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html#overview) | [AMD CDNA 4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) |
| AMD CDNA 3 Architecture | Instinct MI300 series for generative AI and HPC acceleration | [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html#overview) | [AMD CDNA 3 White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) |
| AMD CDNA 2 Architecture | Instinct MI200 series, scientific computing, and machine learning acceleration | [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html#overview) | [AMD CDNA 2 White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf) |
| AMD CDNA Architecture | Instinct MI100 series and Exascale-class GPU compute | [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html#overview) | [AMD CDNA White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf) |

### Architecture, Product, and LLVM Target Quick Map

> For beginners, start from the product name, identify the architecture, then use the LLVM Target (`gfx`) to choose the ROCm / PyTorch installation index. See the “Supported GPU List” below for the full GPU list.

#### CDNA: Data Center Instinct GPUs

| Architecture | Typical products | LLVM Target | Main use |
|--------------|------------------|-------------|----------|
| CDNA 4 | AMD Instinct MI350 series (MI355X, MI350X, MI350P) | `gfx950` | Next-generation AI training / inference and HPC |
| CDNA 3 | AMD Instinct MI300 series (MI325X, MI300X, MI300A) | `gfx942` | Generative AI and HPC acceleration |
| CDNA 2 | AMD Instinct MI200 series (MI250X, MI250, MI210) | `gfx90a` | Scientific computing and machine learning acceleration |
| CDNA | AMD Instinct MI100 series | `gfx908` | Exascale-class GPU compute |

#### RDNA: Radeon GPUs and Ryzen APUs

| Architecture | Typical products / Graphics model | LLVM Target | Main use |
|--------------|-----------------------------------|-------------|----------|
| RDNA 4 | Radeon RX 9000 series (RX 9070 XT / 9070 GRE / 9070) and Radeon AI PRO R9000 series | `gfx1201` | Gaming GPUs, workstation graphics, and AI capabilities |
| RDNA 4 | Radeon RX 9060 XT LP / 9060 XT / 9060 series | `gfx1200` | Mainstream gaming GPUs |
| RDNA 3.5 | Ryzen AI Max / Max PRO 300 (Radeon 8060S / 8050S) incl. AI Max+ 392/388 | `gfx1151` | Mobile / APU integrated GPUs |
| RDNA 3.5 | High-end Ryzen AI 300 / AI 400 / AI PRO 400 models (Radeon 890M / 880M) | `gfx1150` | Mobile / APU integrated GPUs |
| RDNA 3.5 | Mid-range Ryzen AI 300 / AI 400 / AI PRO 400 models (Radeon 860M and similar) | `gfx1152` | Mobile / APU integrated GPUs |
| RDNA 3 | Radeon RX 7900 / PRO W7900 / PRO W7800 series | `gfx1100` | High-end consumer and workstation GPUs |
| RDNA 3 | Radeon RX 7800 / 7700 / PRO W7700 / V710 series | `gfx1101` | Consumer and workstation GPUs |
| RDNA 3 | Radeon RX 7600 series | `gfx1102` | Mainstream consumer GPUs |
| RDNA 3 | Ryzen 200 series (Radeon 780M / 760M / 740M) | `gfx1103` | Mobile / APU integrated GPUs |

### Ryzen Series and On-Device AI PC Official Resources

> Terminology note: what users sometimes call `gfx3.5` is more accurately **RDNA 3.5 / GFX11.5**. In ROCm / LLVM target names, use `gfx1150`, `gfx1151`, `gfx1152`, and related targets instead of `gfx3.5`.

| Topic | Devices / focus | Official resource | Main use |
|-------|-----------------|-------------------|----------|
| ROCm on Radeon and Ryzen | Radeon 9000/7000 series and Ryzen APUs | [Use ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html) | ROCm / PyTorch entry point for consumer Radeon and Ryzen APU platforms |
| RDNA 3.5 system optimization | Ryzen AI Max / AI 300 / AI 400 APUs (`gfx1150` / `gfx1151` / `gfx1152`) | [AMD RDNA3.5 system optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html) | GPUVM, kernel version, and system tuning requirements for RDNA 3.5 APUs |
| GPU hardware specifications | Instinct, Radeon PRO/RX, and Ryzen APUs | [GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) | Look up CUs, wavefronts, caches, LLVM targets, and GFXIP versions |
| ROCm compatibility matrix | GPUs / APUs supported by ROCm releases | [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) | Check hardware, OS, driver, and ROCm version combinations |
| AMDGPU LLVM targets | Compiler and low-level target lookup | [LLVM AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html) | Check LLVM target names such as `gfx1150`, `gfx1151`, and `gfx1152` |
| Ryzen AI Max local LLM inference | Ryzen AI Max+ 395 / Radeon 8060S | [AI Inference on AMD Ryzen AI Max Processor](https://rocm.blogs.amd.com/artificial-intelligence/ryzen-uma-llm/README.html) | Local LLM inference practice on unified-memory APUs |
| Ryzen AI Max+ 395 product and architecture notes | Ryzen AI Max+ 395, Radeon 8060S, XDNA 2 NPU | [AMD Ryzen AI Max+ 395 Processor](https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html) | CPU / GPU / NPU positioning for on-device AI PCs |
| Ryzen AI Software | Ryzen AI NPU / iGPU application development | [AMD Ryzen AI Software](https://www.amd.com/en/developer/resources/ryzen-ai-software.html) / [Ryzen AI Docs PDF](https://ryzenai.docs.amd.com/_/downloads/en/latest/pdf/) | NPU-side ONNX Runtime / Vitis AI EP software stack, distinct from ROCm/HIP |
| AMD XDNA NPU architecture | Ryzen AI NPU | [AMD XDNA Architecture](https://www.amd.com/en/technologies/xdna.html) | Background on spatial dataflow / AI Engine architecture for on-device NPUs |

### Frameworks and Inference Services (ROCm Quick Install Links)

> This section is designed as a quick lookup index for the hello-rocm Skill: it prioritizes framework or AMD ROCm official installation links, with AMD ROCm Blog searches as practical cross-references for examples and version updates.

| Type | Project | ROCm quick install / official notes | AMD official practice reference | hello-rocm entry |
|------|---------|-------------------------------------|---------------------------------|------------------|
| Deep learning framework | PyTorch | [Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html) | [AMD ROCm Blog - PyTorch](https://rocm.blogs.amd.com/search.html?q=PyTorch) | [Environment setup](../00-environment/index.md) |
| Deep learning framework | TensorFlow | [Install TensorFlow for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html) | [AMD ROCm Blog - TensorFlow](https://rocm.blogs.amd.com/search.html?q=TensorFlow) | [Environment setup](../00-environment/index.md) |
| Deep learning framework | JAX | [Install JAX for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html) | [AMD ROCm Blog - JAX](https://rocm.blogs.amd.com/search.html?q=JAX) | [Environment setup](../00-environment/index.md) |
| Inference service | vLLM | [vLLM AMD ROCm installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#amd-rocm) | [AMD ROCm Blog - vLLM](https://rocm.blogs.amd.com/search.html?q=vLLM) | [vLLM deployment tutorials](../01-deploy/index.md) |
| Inference service | Ollama | [Ollama GPU docs](https://github.com/ollama/ollama/blob/main/docs/gpu.md) | [AMD ROCm Blog - Ollama](https://rocm.blogs.amd.com/search.html?q=Ollama) | [Ollama deployment tutorials](../01-deploy/index.md) |
| Inference service | llama.cpp | [llama.cpp build docs - HIP/ROCm](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) | [AMD ROCm Blog - llama.cpp](https://rocm.blogs.amd.com/search.html?q=llama.cpp) | [llama.cpp deployment tutorials](../01-deploy/index.md) |
| Inference service | LM Studio | [LM Studio GPU docs](https://lmstudio.ai/docs/app/advanced/gpu) | [AMD ROCm Blog - LM Studio](https://rocm.blogs.amd.com/search.html?q=LM%20Studio) | [LM Studio deployment tutorials](../01-deploy/index.md) |
| Inference runtime | ONNX Runtime | [Install ONNX Runtime for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/onnxruntime-install.html) | [AMD ROCm Blog - ONNX Runtime](https://rocm.blogs.amd.com/search.html?q=ONNX%20Runtime) | [Environment setup](../00-environment/index.md) |

### Library Documentation

| Library | Purpose | Docs |
|---------|---------|------|
| rocBLAS | Basic linear algebra | [rocBLAS Docs](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/) |
| MIOpen | Deep learning primitives | [MIOpen Docs](https://rocm.docs.amd.com/projects/MIOpen/en/latest/) |
| RCCL | Collective communication | [RCCL Docs](https://rocm.docs.amd.com/projects/rccl/en/latest/) |
| rocFFT | Fast Fourier transforms | [rocFFT Docs](https://rocm.docs.amd.com/projects/rocFFT/en/latest/) |
| rocSPARSE | Sparse matrix operations | [rocSPARSE Docs](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/) |

## Hardware Support

### Supported GPU List

#### Instinct Series (Data Center)

| Series | Models | Architecture | LLVM Target | ROCm Support |
|--------|--------|--------------|-------------|--------------|
| MI350 | MI355X, MI350X, MI350P | CDNA 4 | `gfx950` | ✅ |
| MI300 | MI325X, MI300X, MI300A | CDNA 3 | `gfx942` | ✅ |
| MI200 | MI250X, MI250, MI210 | CDNA 2 | `gfx90a` | ✅ |
| MI100 | MI100 | CDNA | `gfx908` | ✅ |

#### Radeon PRO Series (Workstation)

| Series | Models | Architecture | LLVM Target | ROCm Support |
|--------|--------|--------------|-------------|--------------|
| AI PRO R9000 | R9700S, R9700, R9600D | RDNA 4 | `gfx1201` | ✅ |
| PRO W7000 | W7900 Dual Slot, W7900, W7800 48GB, W7800 | RDNA 3 | `gfx1100` | ✅ |
| PRO W7700 | W7700, V710 | RDNA 3 | `gfx1101` | ✅ |

#### Radeon RX Series (Consumer)

| Series | Models | Architecture | LLVM Target | ROCm Support |
|--------|--------|--------------|-------------|--------------|
| RX 9000 | RX 9070 XT, 9070 GRE, 9070 | RDNA 4 | `gfx1201` | ✅ |
| RX 9000 | RX 9060 XT LP, 9060 XT, 9060 | RDNA 4 | `gfx1200` | ✅ |
| RX 7000 | RX 7900 XTX, 7900 XT, 7900 GRE | RDNA 3 | `gfx1100` | ✅ |
| RX 7000 | RX 7800 XT, 7700 XT, 7700 XE, 7700 | RDNA 3 | `gfx1101` | ✅ |
| RX 7000 | RX 7600 | RDNA 3 | `gfx1102` | ✅ |

#### Ryzen APU Series (Laptop / Mobile)

| Series | Models | Graphics model (iGPU) | Architecture | LLVM Target | ROCm Support |
|--------|--------|------------------------|--------------|-------------|--------------|
| Ryzen AI Max PRO 300 | AI Max+ PRO 395, Max PRO 390/385/380 | Radeon 8060S | RDNA 3.5 | `gfx1151` | ✅ |
| Ryzen AI Max 300 | AI Max+ 395, AI Max+ 392, AI Max+ 388, Max 390, Max 385 | Radeon 8060S / 8050S | RDNA 3.5 | `gfx1151` | ✅ |
| Ryzen AI PRO 400 | AI 9 HX PRO 475/470, AI 9 PRO 465, AI 7 PRO 450, AI 5 PRO 440 | Radeon 890M / 880M / 860M | RDNA 3.5 | `gfx1150` / `gfx1152` | ✅ |
| Ryzen AI 400 | AI 9 HX 475/470, AI 9 465, AI 7 450 | Radeon 890M / 880M / 860M | RDNA 3.5 | `gfx1150` / `gfx1152` | ✅ |
| Ryzen AI 300 | AI 9 HX 375/370, AI 9 365, AI 7 350/345, AI 5 340/330 | Radeon 890M / 880M | RDNA 3.5 | `gfx1150` / `gfx1152` | ✅ |
| Ryzen 200 | 9 270, 7 260/250, 5 240/230/220, 3 210 and PRO series | Radeon 780M / 760M / 740M | RDNA 3 | `gfx1103` | ✅ |

> For the full support list, follow the [ROCm 7.13.0 Compatibility Matrix](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html).

## Community Resources

### Tutorials & Blogs

- [AMD ROCm Blog](https://rocm.blogs.amd.com/) - Official AMD technical blog
- [AMD Developer](https://developer.amd.com/) - AMD developer resource center
- [Datawhale](https://github.com/datawhalechina) - Open-source learning community

### Video Tutorials

> Coming soon...

### Forums & Communities

| Platform | Description | Link |
|----------|-------------|------|
| AMD Community | Official AMD community forum | [community.amd.com](https://community.amd.com/) |
| GitHub Discussions | ROCm project discussions | [ROCm Discussions](https://github.com/ROCm/ROCm/discussions) |
| Reddit r/Amd | AMD-related discussions | [r/Amd](https://www.reddit.com/r/Amd/) |

## Common Tools

### Development Tools

| Tool | Purpose | Install Command |
|------|---------|-----------------|
| hipcc | HIP compiler | `sudo apt install hip-dev` |
| rocprof | Performance profiler | `sudo apt install rocprofiler` |
| rocgdb | GPU debugger | `sudo apt install rocgdb` |
| hipify-clang | CUDA-to-HIP converter | `sudo apt install hipify-clang` |

### AI Frameworks

| Framework | ROCm Support | Installation |
|-----------|--------------|--------------|
| PyTorch | ✅ | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| TensorFlow | ✅ | See official docs |
| JAX | ✅ | See official docs |
| ONNX Runtime | ✅ | See official docs |

## News

### 2026

- **2026.07.15** - [ROCm 7.14.0 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) 🚀 **Milestone release: ROCm officially transitions to TheRock**
  - **TheRock becomes the future build and release foundation of ROCm**: ROCm 7.14.0 officially migrates ROCm to [TheRock](https://github.com/ROCm/TheRock), a modular build and release system. This is the most significant architectural turning point since 7.10.0 introduced Windows / pip support, marking ROCm's shift from a monolithic package to a modular ecosystem. Going forward, ROCm's evolution, community hardware enablement, and independent component releases will all be centered on TheRock. See the [TheRock transition guide](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html)
  - **Three design principles**: ① Leaner core (Core SDK keeps only essential runtime and development components); ② Use case-specific expansions (optional domain SDKs for AI, data science, HPC); ③ Modular installation (install only what your workflow needs — smaller footprint, faster innovation)
  - **Installation and packaging changes**: install directory moves from `/opt/rocm/` to `/opt/rocm/core`; package prefixes are unified from `rocm-*` / `roc*` / `hip*` to `amdrocm-*` (e.g., hipBLAS and rocBLAS are combined into `amdrocm-blas`); a new `/opt/rocm/extras-7/` shared prefix is added. **ABI/API compatibility with ROCm 7.2 legacy is maintained, so recompilation is not required**; for package-manager installs, the `amdrocm` meta package configures `update-alternatives` and provides backward-compatible symlinks
  - **AI framework updates**: PyTorch 2.12.0, JAX 0.10.0, vLLM 0.23.0, SGLang 0.5.13, TensorFlow 2.21 (replacing the previous PyTorch 2.9.1 / JAX 0.8.2 / vLLM 0.19.1 / SGLang 0.5.9)
  - **New hardware support**: Ryzen AI MAX+ PRO 495 / MAX PRO 490/485 (gfx1151), Ryzen AI 5 435/430, AI 5 PRO 435, AI 7 445 (gfx1153) APUs
  - **New OS support**: RHEL 10.2 / 9.8 (replacing 10.1 / 9.7 respectively); SLES 15 SP7, SLES 16, and Debian 13 on MI350P
  - **Component reorganization (TheRock layering)**: ROCm SMI removed (replaced by AMD SMI); ROCm Bandwidth Test reaches EOL (use TransferBench or ROCm Validation Suite instead); ROCm Validation Suite, TransferBench, and MIGraphX move to ROCm-Extras; ONNX runtime becomes Standalone/ONNX; new hipFile direct storage I/O library added
  - **Performance and tooling**: ROCprofiler-SDK becomes the default backend for PyTorch Profiler (2.12+); new SPM (Streaming Performance Monitors, Beta); ROCm Compute Profiler / Systems Profiler now pip-installable
  - > **Note**: 7.14.0 follows the versioning discontinuity that began with the 7.9.0 preview; ASAN packages are not available in 7.14.0 and are planned for a future release

- **2026.05.15** - [ROCm 7.13.0 Preview Release Notes](https://rocm.docs.amd.com/en/7.13.0-preview/about/release-notes.html)
  - Compatibility information should follow the [ROCm 7.13.0 Compatibility Matrix](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html)
  - New hardware support: Instinct MI350P (gfx950), Radeon AI PRO R9700S (gfx1201), Radeon RX 9060 XT LP (gfx1200), Ryzen AI Max+ 392/388 (gfx1151), Ryzen AI PRO 400 / AI 400 series models (gfx1150/gfx1152)
  - New OS support: Ubuntu 26.04 (kernel 7.0), RHEL 10.0/10.1, Debian 13, Oracle Linux 10, Rocky Linux 9, SLES 16.0
  - Latest amdgpu driver version 31.30.0; firmware PLDM bundle 01.26.00.02

- **2026.03.11** - [ROCm 7.12.0 Preview Release Notes](https://rocm.docs.amd.com/en/7.12.0-preview/about/release-notes.html)
  - Updated ROCm 7.12.0 preview release notes covering ROCm components, installation paths, and platform support changes
  - Compatibility information should follow the [ROCm 7.12.0 Compatibility Matrix](https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html?fam=instinct&gpu=mi355x&os=ubuntu&os-version=11_25h2&i=pip)
  - pip index URLs are split by GPU architecture, making it easier to choose the matching wheel source in a virtual environment

### 2025

- **2025.12.11** - [ROCm 7.10.0 Released](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)
  - Windows platform support
  - pip install into Python virtual environments
  - TheRock project restructured underlying architecture

> More news coming soon...

## Recommended Books

> Coming soon...

## Contributing Resources

&emsp;&emsp;If you have quality ROCm-related resources to share, feel free to submit a PR or Issue!

### Submission Requirements

- Links must be valid and content must be high-quality
- Provide a short description of the resource
- Organize according to existing categories

---

<div align="center">

**Contributions welcome!** 🎉

[Open an Issue](https://github.com/datawhalechina/hello-rocm/issues) | [Submit a PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
