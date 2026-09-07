# ROCm 10.0.0 Release Notes: From 7.14.0 to ROCm.AI

<div align="center">

*2026-08-26 · ROCm Core SDK 10.0.0 · [Official docs (latest)](https://rocm.docs.amd.com/en/latest/) · [Official Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)*

[Back to Environment](/00-environment/) · [中文](/zh/00-environment/rocm-10-0-0-release-notes)

</div>

ROCm 10.0.0 is the first major version jump after the 7.x series. 7.14.0 made [TheRock](https://github.com/ROCm/TheRock) the production build and release foundation. 10.0.0 keeps that foundation and, for the first time, folds install, coding, and performance work into an AI-assistant-native workflow called **ROCm.AI**.

For hello-rocm readers, the version number matters less than these three new entry points:

| Component | In one sentence | When you use it |
|:---|:---|:---|
| **AMD Skills** | Official AMD optimization knowledge packaged as Skills for Claude, Cursor, and Codex | You are writing ROCm / HIP / vLLM code with an AI assistant and want AMD best practices by default |
| **Hyperloom** | Open-source auto-optimization engine: profile, find bottlenecks, rewrite kernels, tune parameters, then validate | Inference already runs and you want more throughput |
| **ROCm CLI** | Install, verify, deploy, and manage with one command surface | You do not want to hand-copy pip, driver, and image commands |

This page explains what changed from 7.14.0 to 10.0.0, then unpacks those three entry points. For install steps, see [00-Environment](/00-environment/).

---

## 1. 7.14.0 → 10.0.0 at a glance

| Area | ROCm 7.14.0 (2026-07-15) | ROCm 10.0.0 (2026-08-26) |
|:---|:---|:---|
| Positioning | TheRock becomes the build / release foundation | Decade milestone; **ROCm.AI** lands on top of TheRock |
| Developer experience | Docs + wheels + Docker, assembled by hand | **AMD Skills + Hyperloom + ROCm CLI** |
| pip index | `https://repo.amd.com/rocm/whl-multi-arch/` | `https://stable.repo.amd.com/rocm/whl-next/` |
| System packages | Distributed from the older `repo.amd.com` layout | Consolidated at [stable.repo.amd.com](https://stable.repo.amd.com) |
| PyTorch | 2.12.0 | **2.13.0** (also 2.12.0 / 2.11.0) |
| torchvision / torchaudio | 0.27.0 / 2.11.0 | **0.28.0 / 2.11.0.2** |
| JAX | 0.10.0 | **0.11.0** (also 0.10.2 / 0.10.0) |
| vLLM | 0.23.0 | **0.27.0** |
| SGLang | 0.5.13 | **0.5.15** |
| TensorFlow | 2.21 | 2.21 (also 2.20 / 2.19.1) |
| MIGraphX / ONNX Runtime | 2.16 / 1.23.2 generation | **2.17 / 1.29.0** |
| Windows driver | Adrenalin **26.5.1** | Adrenalin **26.8.1** |
| Linux driver | amdgpu 31.30.0 generation | amdgpu **31.50.0** |
| New hardware | gfx1153 (Ryzen AI 7 445 / AI 5 435, etc.) | **Radeon RX 9050 / 9050 4GB (gfx1200)** |
| Windows SDK | HIP SDK on a separate track from Linux | **HIP SDK retired**; Windows and Linux share the Core SDK |
| ASAN packages | Not shipped in 7.14.0 | **Shipped alongside standard packages** |
| Recommended install | `uv pip` + official images | `uv pip` still works; new path is **`rocm install sdk`** |

> Treat [ROCm 10.0.0 docs](https://rocm.docs.amd.com/en/latest/) as the canonical entry. Compatibility is defined by the [Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

---

## 2. Why the jump from 7 to 10

7.14.0 answered “how do we split ROCm, install only what we need, and ship Windows / Linux from one TheRock pipeline.” 10.0.0 answers the next layer:

- After install, developers still had to remember gfx extras, pick images, match drivers, and read optimization docs by hand.
- AI coding assistants default to CUDA / NVIDIA habits and do not automatically take AMD kernel, scheduling, or quantization paths.
- Inference optimization was still “human reads a profile → human edits a kernel → human re-benchmarks,” on a week-scale loop.

So 10.0.0 is not just another library bump. It moves the platform from **shipping libraries** to **shipping workflows**. The official blog puts it directly: older ROCm releases mainly gave better primitives; ROCm.AI covers the install, validate, serve, and optimize loop around those primitives. [ROCm 10.0 official write-up](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html)

TheRock is not replaced. The 10.0.0 Core SDK, framework wheels, and validated images all come from the same TheRock pipeline, with minor releases about every six weeks.

---

## 3. Focus 1: AMD Skills

### 3.1 What it actually is

[AMD Skills](https://github.com/amd/skills) is not another documentation site. It is a catalog of directories written to the [Agent Skills](https://github.com/anthropics/skills) standard: each skill has a `SKILL.md`, plus optional scripts and references. Claude Code, Cursor, Codex, and Gemini CLI load the matching skill when the task description fits.

Docs describe every API option. A skill encodes the opinionated path an AMD engineer would take — which image, which `gfx`, which environment variables, and what to check before serving. When you write ROCm code with an assistant, it follows those validated paths instead of improvising from CUDA muscle memory.

Install:

```bash
npx skills add amd/skills
```

Pin a skill and an agent:

```bash
npx skills add amd/skills --skill local-ai-use --agent cursor
npx skills add amd/skills --list
```

You can also copy folders into `~/.cursor/skills/`, `~/.claude/skills/`, or `$HOME/.agents/skills`.

### 3.2 How the catalog is split

The official catalog has three layers that match the two hello-rocm paths most readers care about: local models on a Ryzen AI PC, and production inference on Instinct.

**Client-native (Ryzen AI / local AI PC)**

| Skill | What it does |
|:---|:---|
| `local-ai-use` | Route image generation, TTS, and STT through a local server to cut cloud token cost |
| `local-ai-app-integration` | Add offline, privacy, and local inference to an existing cloud LLM app |

**Cross-stack (client to cloud)**

| Skill | What it does |
|:---|:---|
| `rocm-doctor` | Diagnose ROCm / HIP / PyTorch / llama.cpp against known misconfigurations; thin driver over `rocm examine` / `diagnose` / `fix` (planned) |
| `lemonade-router-builder` | Route requests through Lemonade by content, sensitivity, or capability |
| `hrr-replay-analysis` | Replay and analyze GPU workloads with HIP Record and Replay (planned) |

**Server-native (Instinct / EPYC)**

| Skill | What it does |
|:---|:---|
| `serving-llms-on-instinct` | Detect hardware, check model fit, apply a vLLM recipe, launch a benchmarked endpoint |
| `serving-llms-on-epyc` | Serve LLMs on EPYC with zentorch / vLLM |
| `hyperloom-workload-optimizer` | Install Hyperloom and autonomously optimize LLM inference throughput on Instinct |
| `magpie-kernel-evaluator` | Evaluate kernel correctness and performance against vLLM / SGLang |
| `tracelens-analysis-orchestrator` | Split a PyTorch profile with TraceLens and write a prioritized report |

### 3.3 How this relates to the hello-rocm Skill

This repo’s [`src/hello-rocm-skill`](https://github.com/datawhalechina/hello-rocm/tree/main/src/hello-rocm-skill) answers “which tutorial should I read, how do I deploy, how do I debug.” AMD Skills answers “what is the official AMD-validated path.” Install both:

- “Which hello-rocm chapter should I start with, and which gfx extra do I use?” → hello-rocm Skill
- “Stand up vLLM on MI300X the official way / squeeze throughput with Hyperloom” → AMD Skills

---

## 4. Focus 2: Hyperloom

### 4.1 The problem it closes

[Hyperloom](https://rocm.docs.amd.com/projects/hyperloom/en/latest/) is an open-source agentic optimizer (source: [AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom)). It is not another manual profiler GUI. It closes the loop of “read a trace → find the bottleneck → rewrite a kernel or retune → validate correctness and performance.”

The official loop is:

```text
Profile → Analyze → Plan → Optimize → Validate → repeat
```

You no longer drive every cycle. Give it a model, framework, GPU, parallelism, and sequence lengths; it profiles, proposes, edits, benchmarks, and checks correctness. AMD’s claim is that week-scale manual optimization can compress into hours, while covering search space a human would skip under time pressure.

### 4.2 The parts underneath

| Component | Role |
|:---|:---|
| TraceLens-Agent | Mark bottlenecks from a profile automatically |
| Magpie | Evaluate / compare kernels and benchmark vLLM or SGLang |
| IntelliKit | Conversational profiling |
| GEAK | Multi-agent kernel rewrites across Triton, HIP, CK, FlyDSL, and TileLang |
| Arbor | Self-evolving search over the optimization space |
| AgentKernelArena | A/B-test optimization agents on standard tasks |

The public 10.0.0 surface targets **MI300X / MI325X / MI355X**, with **vLLM and SGLang**. On Radeon / Ryzen, hello-rocm still starts with “get deploy working.” Hyperloom is for readers who already have Instinct hardware and need production throughput.

### 4.3 How to attach it to your workflow

Two paths:

1. **Via AMD Skills**: install `hyperloom-workload-optimizer` and tell the assistant to raise inference throughput on Instinct with Hyperloom.
2. **Via the official CLI**: start a session from [Run a Hyperloom optimization](https://rocm.docs.amd.com/projects/hyperloom/en/latest/how-to/optimize.html), passing framework, model path, `TP` / `CONC` / `ISL` / `OSL`, and a target gain.

Hyperloom does not replace the rocprofiler-sdk work from 7.14.0. That release made rocprofiler-sdk the PyTorch Profiler backend. 10.0.0 adds rocSHMEM / hipFile / OpenMP tracing, per-node HIP Graph attribution, and Roofline on gfx1150 / 1151 / 1152. Hyperloom consumes those traces, then takes the extra step of changing the code.

---

## 5. Focus 3: ROCm CLI

### 5.1 The one-command layer

[ROCm CLI](https://github.com/ROCm/rocm-cli) is the command-line face of ROCm.AI. It ships as a single prebuilt binary for Linux and Windows x86_64 and **does not require an existing ROCm, Python, or Rust install**. It is a **Technology Preview**: commands and UI can still change. It can already manage runtimes from 7.13 onward; official ROCm 10 support is listed as coming next.

Linux / WSL:

```bash
curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
```

Day-to-day commands:

```bash
rocm                 # launcher: setup / serve / diagnose / chat / dashboard
rocm examine         # GPU, driver, ROCm, and engine readiness
rocm install sdk     # TheRock wheels into a CLI-managed Python environment
rocm install driver  # amdgpu driver on Linux
rocm serve qwen      # local OpenAI-compatible model server
rocm dash            # full-screen telemetry (Linux / WSL)
```

The 7.14.0 path was “create a venv, fill extras, match image tags.” 10.0.0 adds a managed path: `rocm install sdk` fetches TheRock wheels and a matching PyTorch, `rocm examine` tells you whether the gap is driver, permissions, or runtime, and `rocm serve` picks an engine from the GPU — Lemonade (GGUF) on Ryzen / Radeon, vLLM (safetensors) on Instinct.

### 5.2 What you get beyond another install script

- **Side-by-side runtimes**: `rocm runtimes activate` / `rollback` so one machine can hold multiple ROCm installs and roll back an upgrade.
- **Engine adapters**: Lemonade on client GPUs, vLLM on Instinct. Windows currently has the CLI and Lemonade, not the live dashboard or vLLM.
- **A hook for agents**: `rocm-doctor` is a thin layer over `examine` / `diagnose` / `fix`. Skills and the CLI share one state machine.
- **Isolation from legacy stacks**: an existing 7.14.0 `/opt/rocm` or hand-built `.venv` is reported as `legacy_rocm_status: detected_unmanaged`. `rocm install sdk` creates a managed runtime next to it instead of overwriting it.

Minimum Linux is **Ubuntu 24.04** (glibc 2.38+). Ubuntu 22.04 cannot run the prebuilt Lemonade engine. There is no official macOS package.

hello-rocm still documents `uv pip` as the reproducible baseline so wheel versions stay explicit. On a new machine, try ROCm CLI first if you just want to verify the GPU, then cross-check versions against this page and the [environment baseline](/00-environment/).

---

## 6. What else changed in the stack

### 6.1 Install and distribution

- GPU software starts consolidating on [stable.repo.amd.com](https://stable.repo.amd.com): ROCm packages, the amdgpu driver, and public tools share one layout, with multiple versions and architectures side by side.
- The pip index moves from `repo.amd.com/rocm/whl-multi-arch/` to `stable.repo.amd.com/rocm/whl-next/`. The `[device-gfxXXXX]` extras stay the same.
- Installed RPM / DEB / runfile packages now embed **RPATH** instead of RUNPATH, so a ROCm binary prefers its own library tree and is less likely to be hijacked by `LD_LIBRARY_PATH` on a multi-version machine. Tarballs still use RUNPATH for compose-your-own environments.
- On Windows, the HIP SDK is retired. Windows and Linux share the Core SDK and release cadence. Windows ships as a tarball today; a native installer is planned later in 2026.

### 6.2 Frameworks and images

This project’s pip baseline follows the official recommendation:

```bash
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" \
  "torchaudio==2.11.0.2+rocm10.0.0"
```

Official validated vLLM image (the image bundles PyTorch 2.12.0 — do not mix that with the pip 2.13.0 line):

```bash
docker pull rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
```

10.0.0 also adds Unsloth local fine-tuning on Ryzen AI MAX, plus ComfyUI tuning for Wan2.2, FLUX.2 KLEIN, SD 3.5, and related models. Those will land in Fine-tune / practice chapters later and do not block the 01-Deploy environment bump.

### 6.3 Communication, math, and tools

- **RCCL / rocSHMEM**: the largest single investment in this release. RCCL’s upstream NCCL merge advances to 2.30.4, with symmetric memory, GPU-initiated networking (GIN), and Python APIs. rocSHMEM continues closing the NVSHMEM 3.6.5 API gap.
- **hipBLASLt**: tune GEMM kernel selection locally for your problem shapes; weights never leave the machine.
- **ROCm Optiq 1.0 GA**: Systems Profiler timelines and Compute Profiler analysis in one visualization environment.
- **ASAN packages**: the Address Sanitizer builds that 7.14.0 omitted now install like any other ROCm package.

---

## 7. Upgrade guidance for hello-rocm readers

1. **New machine**: follow the [environment baseline](/00-environment/) with `uv pip` and ROCm 10.0.0. If you only want to verify the GPU first, use `rocm examine` / `rocm install sdk`.
2. **Already running 7.14.0**: do not rebuild just for the version number. Upgrade when you need the new frameworks (PyTorch 2.13 / vLLM 0.27) or ROCm.AI. Remove old HIP SDK / old wheels first, and move Windows drivers to Adrenalin 26.8.1.
3. **Writing code with an assistant**: keep the hello-rocm Skill and add [amd/skills](https://github.com/amd/skills). They can coexist.
4. **Instinct and throughput work**: treat Hyperloom as the headline of 10.0.0, not another handwritten kernel-tuning note.
5. **Local Radeon / Ryzen deploy**: environment commands now target 10.0.0. LM Studio / Ollama / llama.cpp steps stay mostly the same; the ROCm and driver versions underneath change.

---

## 8. Official links

| Resource | Link |
|:---|:---|
| ROCm 10.0.0 docs home | <https://rocm.docs.amd.com/en/latest/> |
| Core SDK Release Notes | <https://rocm.docs.amd.com/en/latest/about/release-notes.html> |
| Install selector | <https://rocm.docs.amd.com/en/latest/install/rocm.html> |
| Compatibility matrix | <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html> |
| TheRock transition guide | <https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html> |
| Official blog | <https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html> |
| ROCm.AI announcement | <https://newsroom.amd.com/news/rocm-10-software-ai-native-developer-experiences/> |
| AMD Skills | <https://github.com/amd/skills> |
| ROCm CLI | <https://github.com/ROCm/rocm-cli> |
| Hyperloom docs | <https://rocm.docs.amd.com/projects/hyperloom/en/latest/> |
| Hyperloom source | <https://github.com/AMD-AGI/Hyperloom> |
