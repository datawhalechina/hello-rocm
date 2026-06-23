<div align=center>
  <img src="./docs/public/images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*Open Source · Community Driven · Making the AMD AI Ecosystem More Accessible*

<p align="center">
  <a href="./README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="./docs-readme/zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="./docs-readme/ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="./docs-readme/es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="./docs-readme/fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="./docs-readme/ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="./docs-readme/ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="./docs-readme/vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="./docs-readme/de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>
<div align="center">

<a href="https://datawhalechina.github.io/hello-rocm/"><img src="https://img.shields.io/badge/hello--rocm_Full_Tutorial-Try_Online-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="25"></a>

</div>

&emsp;&emsp;Starting with **ROCm 7.10.0** (released December 11, 2025), ROCm installs cleanly into Python virtual environments—just like CUDA—and officially supports both **Linux and Windows**. This is a big deal: if you're learning AI or tinkering with large language models, you're no longer locked into NVIDIA hardware. AMD GPUs are now a genuinely practical option.

&emsp;&emsp;That said, **easier hardware access doesn't mean the learning path is obvious**. If you already know your way around LLMs and want to run them on AMD, the real questions start here: How do you actually deploy a model on an AMD GPU? How do you fine-tune or train on top of that? How does ROCm's programming model work, and what does migrating from CUDA look like? And ultimately, how do all these pieces fit together into a production-ready AI app?

&emsp;&emsp;**hello-rocm** is built for exactly that journey. This project walks you through the full LLM workflow on AMD's ROCm platform—from **getting your first model running** to **building real AI applications on AMD GPUs**—covering fine-tuning, training, and GPU programming at every step along the way. An AMD GPU isn't just a graphics card; it's your on-ramp to serious AI development.

&emsp;&emsp;**At its core, this project is a collection of hands-on tutorials** so students and practitioners can learn AMD ROCm in a structured way. **Anyone can open an issue or submit a PR**—we build and maintain this together.

> &emsp;&emsp;***Recommended path: Start with [00-Environment](./docs/en/00-environment/index.md) (ROCm + PyTorch + **uv**), then move to deployment and fine-tuning, and finally explore operator optimization and GPU programming. Once your environment is set up, LM Studio or vLLM is a great first deployment to try.***

### hello-rocm Skill: put this project inside your AI coding assistant

&emsp;&emsp;If your AI coding tool supports Skills, Rules, or Agent configs, you can load the built-in **hello-rocm Skill**. It uses this repo's directory structure, reference index, GPU architecture table, deployment guides, and troubleshooting checklist to point you to the right doc or official link.

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place (e.g., .claude/skills, .cursor/skills, or .agents/skills), then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;Example questions: Can my AMD GPU run ROCm? What's the fastest way to get a local LLM running? How do I install vLLM / Ollama / llama.cpp on ROCm? Why is `torch.cuda.is_available()` returning False? See the [hello-rocm Skill guide](./docs/en/04-references/index.md#hello-rocm-skill) for more.

### Latest updates

- *2026.5.15:* [*ROCm 7.13.0 Release Notes*](https://rocm.docs.amd.com/en/7.13.0-preview/index.html)

- *2026.3.11:* [*ROCm 7.12.0 Release Notes*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025.12.11:* [*ROCm 7.10.0 Release Notes*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### Supported models & tutorials

<p align="center">
  <strong>✨ Popular LLMs: environment setup · multi-framework inference · fine-tuning ✨</strong><br>
  <em>Unified ROCm setup (Windows / Ubuntu) + ROCm 7+ · per-model tutorial directories (growing)</em><br>
 <a href="./docs/en/00-environment/index.md">00 — Environment setup</a>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">

  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="./docs/en/01-deploy/qwen3/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="./docs/en/01-deploy/qwen3/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="./docs/en/01-deploy/qwen3/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="./docs/en/01-deploy/qwen3/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="./docs/en/02-fine-tune/qwen3/qwen3-0.6b-lora-swanlab.md">Qwen3-0.6B LoRA + SwanLab</a><br>
      • <a href="./src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb">Qwen3-8B LoRA</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3.5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="./docs/en/01-deploy/qwen3.5/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="./docs/en/01-deploy/qwen3.5/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="./docs/en/01-deploy/qwen3.5/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="./docs/en/01-deploy/qwen3.5/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="./docs/en/02-fine-tune/qwen3.5/qwen3.5-4b-lora-swanlab.md">Qwen3.5-4B LoRA + SwanLab</a><br>
      • <a href="./src/fine-tune/models/qwen3.5/Qwen3.5-4B-LoRA.ipynb">Qwen3.5-4B LoRA Notebook</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Gemma4</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="./docs/en/01-deploy/gemma4/gemma4_model.md">Gemma 4 model overview</a><br>
      • <a href="./docs/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="./docs/en/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="./docs/en/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="./docs/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="./src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb">Gemma4 E4B LoRA fine-tuning (ModelScope, single GPU, Notebook)</a><br>
    </td>
  </tr>
</table>

## Why this project

&emsp;&emsp;What is ROCm?

> ROCm (Radeon Open Compute) is AMD's open-source GPU computing stack for HPC and machine learning. It enables parallel workloads on AMD GPUs and is the primary alternative to CUDA on AMD hardware.

&emsp;&emsp;Open-source LLMs are exploding, but most tutorials and tooling still assume NVIDIA's CUDA stack. If you pick AMD, finding end-to-end, ROCm-native learning material has been a real gap.

&emsp;&emsp;With **ROCm 7.10.0** (December 11, 2025), AMD's **TheRock** initiative decoupled the compute runtime from the OS, so the same ROCm APIs now work on both **Linux and Windows**, and you can `pip install` ROCm packages directly into virtual environments—just like CUDA. ROCm has gone from "Linux-only plumbing" to a true cross-platform AI compute platform. **hello-rocm** is the practical guide that helps more people actually use AMD GPUs for training and inference.

&emsp;&emsp;***We want to bridge the gap between AMD GPUs and everyday developers—open, inclusive, and aimed at a broader AI future.***

## Who this is for

&emsp;&emsp;This project is for you if you:

* Own an AMD GPU and want to try running LLMs locally;
* Want to develop on AMD but can't find a structured ROCm learning path;
* Care about cost-effective model deployment and inference;
* Are curious about ROCm and prefer learning by doing.

## Roadmap and structure

&emsp;&emsp;The repo is organized around the full ROCm LLM workflow: **unified environment baseline (00-Environment)**, deployment, fine-tuning, and operator optimization / GPU programming:


### Repository layout

```
hello-rocm/
├── docs/                   # VitePress documentation source
│   ├── en/                 # English docs
│   │   ├── 00-environment/    # ROCm baseline install & config
│   │   ├── 01-deploy/         # LLM deployment on ROCm
│   │   ├── 02-fine-tune/      # LLM fine-tuning on ROCm
│   │   ├── 03-infra/          # Operator optimization / GPU programming on ROCm
│   │   ├── 04-references/     # Curated ROCm references
│   │   └── 05-amd-yes/        # Community AMD project showcases
│   └── zh/                 # 中文文档
├── src/                    # Source code & scripts
└── assets/                 # Shared assets
```

### 00. Environment — ROCm baseline

<p align="center">
  <strong>🛠️ ROCm environment install & configuration</strong><br>
  <em>Single baseline · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="./docs/en/00-environment/index.md">Getting Started with ROCm Environment</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="./docs/en/00-environment/rocm-gpu-architecture-table.md">GPU architecture & pip index map</a><br>
      • Windows 11: drivers, security prerequisites, install flow<br>
      • Ubuntu 24.04: uv-based install and optional one-liner script<br>
      • Verification, uninstall, and switching GPU targets
    </td>
  </tr>
</table>

### 01. Deploy — LLM deployment on ROCm

<p align="center">
  <strong>🚀 LLM deployment on ROCm</strong><br>
  <em>From zero to a running model on AMD GPUs</em><br>
  📖 <strong><a href="./docs/en/01-deploy/index.md">Getting Started with ROCm Deploy</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio from scratch<br>
      • vLLM from scratch<br>
      • Ollama from scratch<br>
      • llama.cpp from scratch<br>
      • ATOM from scratch
    </td>
  </tr>
</table>

### 02. Fine-tune — LLM fine-tuning on ROCm

<p align="center">
  <strong>🔧 LLM fine-tuning on ROCm</strong><br>
  <em>Efficient fine-tuning on AMD GPUs</em><br>
  📖 <strong><a href="./docs/en/02-fine-tune/index.md">Getting Started with ROCm Fine-tune</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Fine-tuning from scratch<br>
      • Single-machine fine-tuning scripts<br>
      • Multi-node, multi-GPU fine-tuning
    </td>
  </tr>
</table>

### 03. Infra — operator optimization & GPU programming

<p align="center">
  <strong>⚙️ Operator optimization & GPU programming</strong><br>
  <em>From AMD AI hardware overview to HIP kernels & performance profiling</em><br>
  📖 <strong><a href="./docs/en/03-infra/index.md">Getting Started with ROCm Operator Optimization</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • AMD AI hardware landscape & ROCm ecosystem<br>
      • GPU software stack & hardware architecture deep dive<br>
      • HIP programming intro & hand-written kernel walkthrough<br>
      • Custom PyTorch operators & Autograd integration
    </td>
  </tr>
</table>

### 04. References

<p align="center">
  <strong>📚 ROCm references</strong><br>
  <em>Official and community resources</em><br>
  📖 <strong><a href="./docs/en/04-references/index.md">ROCm References</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">ROCm official docs</a><br>
      • <a href="https://github.com/amd">AMD on GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCm release notes</a><br>
      • <a href="./docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">AMD GPU architecture whitepapers (CDNA / RDNA)</a><br>
      • <a href="./docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">Framework & inference service ROCm quick-install links</a><br>
      • Related news
    </td>
  </tr>
</table>

### 05. AMD-YES — community showcases

<p align="center">
  <strong>✨ AMD project showcases</strong><br>
  <em>Community-driven projects on AMD GPUs</em><br>
  📖 <strong><a href="./docs/en/05-amd-yes/index.md">Getting Started with ROCm AMD-YES</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • toy-cli — lightweight terminal LLM assistant<br>
      • Minesweeper Agent — play Minesweeper with a local LLM (Agent loop practice)<br>
      • WeChat "Jump Jump" with YOLOv10 — game AI in action (train & run YOLOv10 under ROCm)<br>
      • Chat-甄嬛 — period-drama dialogue model<br>
      • Travel planner — HelloAgents agent demo<br>
      • Torch-RecHub — recommender systems (CTR, retrieval, multi-task, ONNX export)<br>
      • happy-llm — distributed LLM training
    </td>
  </tr>
</table>

## Contributing

&emsp;&emsp;We welcome all kinds of contributions:

* Add or improve tutorials
* Fix bugs and errors
* Share your AMD projects
* Suggest ideas and new directions

&emsp;&emsp;Please read the **[Content Guide](./CONTENT_GUIDE_en.md)** (directory layout, naming, images—aligned with tutorials like Qwen3 / Qwen3.5) and then **[CONTRIBUTING_en.md](./CONTRIBUTING_en.md)** (issue/PR workflow and per-model directory conventions).

&emsp;&emsp;If you run into issues while using ROCm, deploying models, or following the tutorials, feel free to join our **[community discussion](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** to share experiences, report problems, and help improve the docs.

&emsp;&emsp;Want to help maintain the project long-term? Reach out—we'd be happy to add you as a maintainer.

## Acknowledgments

### Core contributors

- [Zhixue Song (不要葱姜蒜) — project lead](https://github.com/KMnO4-zx) (Datawhale member; self-llm & happy-llm project lead)
- [Yu Chen — project lead](https://github.com/lucachen) (content creator; Google Developer Expert in ML)
- [Sizhou Chen — contributor](https://github.com/jjyaoao) (Datawhale member; hello-agents project lead)
- [Jiahang Pan — contributor](https://github.com/amdjiahangpan) (content creator; AMD software engineer)
- [Weihong Liu — contributor](https://github.com/Weihong-Liu) (Datawhale member)
- [Dongbo Hao — contributor](https://github.com/wlkq151172) (content creator; Jiangnan University grad student)
- [Muling Ke — contributor](https://github.com/1985312383) (Datawhale member; Torch-RecHub project lead)

> More contributors are always welcome!

### Others

- Have an idea? Open an issue—we'd love to hear it.
- Big thanks to everyone who has contributed tutorials.
- Special thanks to the **AMD University Program** for supporting this project.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>


## License

[MIT License](./LICENSE)

---

<div align="center">

**Let's build the future of AMD AI together.** 💪

Made with ❤️ by the hello-rocm community

</div>
