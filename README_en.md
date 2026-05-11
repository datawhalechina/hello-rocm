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

&emsp;&emsp;Since **ROCm 7.10.0** (released December 11, 2025), ROCm can be installed seamlessly in Python virtual environments much like CUDA, with official support for both **Linux and Windows**. This is a major step for AMD in AI: learners and LLM enthusiasts are no longer limited to NVIDIA hardware—AMD GPUs are a strong, practical choice.

&emsp;&emsp;AMD has committed to a **roughly six-week** ROCm release cadence with a strong focus on AI. The roadmap is exciting.

&emsp;&emsp;There is still a shortage of systematic tutorials worldwide for ROCm LLM inference, deployment, training, fine-tuning, and operator optimization / GPU programming topics. **hello-rocm** exists to fill that gap.

&emsp;&emsp;**This project is primarily tutorials** so students and future practitioners can learn AMD ROCm in a structured way. **Anyone is welcome to open issues or submit pull requests** to grow and maintain the project together.

> &emsp;&emsp;***Learning path: Finish [00-Environment](./docs/en/00-environment/index.md) first (ROCm + PyTorch + **uv**), then deployment and fine-tuning, and finally operator optimization and GPU programming topics. After your environment works, LM Studio or vLLM is a good place to start.***

### hello-rocm Skill: put this project inside your AI assistant

&emsp;&emsp;If you use an AI coding tool that supports Skills, Rules, or Agent configuration, you can use the built-in **hello-rocm Skill**. It uses this repository’s structure, reference index, GPU architecture table, deployment tutorials, and troubleshooting checklist to point you to the right document and official link.

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;Try asking: Does my AMD GPU support ROCm? What is the fastest path to run my first local LLM? How do I install vLLM / Ollama / llama.cpp on ROCm? How do I debug `torch.cuda.is_available()` returning False? See the [hello-rocm Skill guide](./docs/en/04-references/index.md#hello-rocm-skill).

### Latest updates

- *2026.3.11:* [*ROCm 7.12.0 Release Notes*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025.12.11:* [*ROCm 7.10.0 Release Notes*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### Supported models & tutorials

<p align="center">
  <strong>✨ Mainstream LLMs: environment · multi-framework inference · fine-tuning ✨</strong><br>
  <em>Unified ROCm setup (Windows / Ubuntu) + ROCm 7+ · per-model tutorials (growing)</em><br>
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
      • <a href="./src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb">Qwen3-8B LoRA (Notebook)</a><br>
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
      • <a href="./src/fine-tune/models/gemma4/01-Gemma4-E4B-LoRA及SwanLab可视化记录.ipynb">Gemma4 E4B LoRA fine-tuning (TRL, Notebook)</a><br>
    </td>
  </tr>
</table>

## Why this project

&emsp;&emsp;What is ROCm?

> ROCm (Radeon Open Compute) is AMD’s open GPU computing stack for HPC and machine learning. It lets you run parallel workloads on AMD GPUs and is the primary CUDA-alternative path on AMD hardware.

&emsp;&emsp;Open LLMs are everywhere, yet most tutorials and tools assume the NVIDIA CUDA stack. Developers who choose AMD often lack end-to-end, ROCm-native learning material.

&emsp;&emsp;From **ROCm 7.10.0** (December 11, 2025), AMD’s **TheRock** work decouples the compute runtime from the OS so the same ROCm interfaces run on **Linux and Windows**, and ROCm can be installed into Python environments similarly to CUDA. ROCm is no longer “Linux-only plumbing”—it is a cross-platform AI compute platform. **hello-rocm** collects practical guides so more people can actually use AMD GPUs for training and inference.

&emsp;&emsp;***We hope to be a bridge between AMD GPUs and everyday builders—open, inclusive, and aimed at a wider AI future.***

## Who it is for

&emsp;&emsp;You may find this project useful if you:

* Have an AMD GPU and want to run LLMs locally;
* Want to build on AMD but lack a structured ROCm curriculum;
* Care about cost-effective deployment and inference;
* Are curious about ROCm and prefer hands-on learning.

## Roadmap and structure

&emsp;&emsp;The repo follows the full ROCm LLM workflow: **unified baseline (00-Environment)**, deployment, fine-tuning, and operator optimization / GPU programming topics:


### Repository layout

```
hello-rocm/
├── docs/                   # VitePress documentation source
│   ├── en/                 # English docs
│   │   ├── environment/    # ROCm baseline install & config
│   │   ├── deploy/         # LLM deployment on ROCm
│   │   ├── fine-tune/      # LLM fine-tuning on ROCm
│   │   ├── infra/          # Operator optimization / GPU programming on ROCm
│   │   ├── amd-yes/        # Community AMD project showcases
│   │   └── references.md   # Curated ROCm references
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
      • Ubuntu 24.04: uv-based install and optional unified installer script<br>
      • Verification, uninstall, and switching GPU targets
    </td>
  </tr>
</table>

### 01. Deploy — LLM deployment on ROCm

<p align="center">
  <strong>🚀 ROCm LLM deployment</strong><br>
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
  <strong>🔧 ROCm LLM fine-tuning</strong><br>
  <em>Efficient fine-tuning on AMD GPUs</em><br>
  📖 <strong><a href="./docs/en/02-fine-tune/index.md">Getting Started with ROCm Fine-tune</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Fine-tuning tutorials from scratch<br>
      • Single-machine fine-tuning scripts<br>
      • Multi-node multi-GPU fine-tuning
    </td>
  </tr>
</table>

### 03. Infra — operator optimization & GPU programming

<p align="center">
  <strong>⚙️ ROCm Operator Optimization & GPU Programming</strong><br>
  <em>From AMD AI hardware panorama to HIP operators & performance profiling</em><br>
  📖 <strong><a href="./docs/en/03-infra/index.md">Getting Started with ROCm Operator Optimization</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • AMD AI hardware panorama & ROCm ecosystem<br>
      • GPU software stack & hardware architecture deep dive<br>
      • HIP programming & hand-written Kernel practice<br>
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
      • <a href="https://rocm.docs.amd.com/">ROCm official documentation</a><br>
      • <a href="https://github.com/amd">AMD on GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCm release notes</a><br>
      • <a href="./docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">AMD GPU architecture whitepapers (CDNA / RDNA)</a><br>
      • <a href="./docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">Framework and inference service ROCm quick install links</a><br>
      • Related news
    </td>
  </tr>
</table>

### 05. AMD-YES — community showcases

<p align="center">
  <strong>✨ AMD project showcases</strong><br>
  <em>Community-driven examples on AMD GPUs</em><br>
  📖 <strong><a href="./docs/en/05-amd-yes/index.md">Getting Started with ROCm AMD-YES</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • toy-cli — lightweight terminal LLM assistant<br>
      • WeChat “Jump Jump” with YOLOv10 — Game AI in Action (Train and use YOLOv10 under ROCm)<br>
      • Chat-甄嬛 — period-style dialogue model<br>
      • Travel planner — HelloAgents agent demo<br>
      • happy-llm — distributed LLM training
    </td>
  </tr>
</table>

## Contributing

&emsp;&emsp;We welcome contributions of all kinds:

* Improve or add tutorials
* Fix errors and bugs
* Share your AMD projects
* Suggest ideas and directions

&emsp;&emsp;Please read **[规范指南](./规范指南.md)** (structure, naming, images—aligned with tutorials such as Qwen3), then **[CONTRIBUTING.md](./CONTRIBUTING.md)** (issues, PRs, and per-model directory conventions).

&emsp;&emsp;If you run into troubleshooting or FAQ issues while using ROCm, deploying models, or reading the tutorials, you are also welcome to join the **[community discussion](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** to share experience, report problems, and help improve the tutorials with the community.

&emsp;&emsp;If you want to help maintain the repo long term, reach out—we can add you as a maintainer.

## Acknowledgments

### Core contributors

- [Zhixue Song (不要葱姜蒜) — project lead](https://github.com/KMnO4-zx) (Datawhale member; self-llm and happy-llm project lead)
- [Yu Chen — project lead](https://github.com/lucachen) (content creator; Google Developer Expert in Machine Learning)
- [Sizhou Chen — contributor](https://github.com/jjyaoao) (Datawhale member; hello-agents project lead)
- [Jiahang Pan — contributor](https://github.com/amdjiahangpan) (content creator; AMD software engineer)
- [Weihong Liu — contributor](https://github.com/Weihong-Liu) (Datawhale member)
- [Dongbo Hao — contributor](https://github.com/wlkq151172) (content creator; Jiangnan University graduate student)
- [Muling Ke — contributor](https://github.com/1985312383) (Datawhale member)

> More contributors are always welcome.

### Others

- Ideas and feedback are welcome—please open issues.
- Thanks to everyone who has contributed tutorials.
- Thanks to **AMD University Program** for supporting this project.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>


## License

[MIT License](./LICENSE)

---

<div align="center">

**Let’s build the future of AMD AI together.** 💪

Made with ❤️ by the hello-rocm community

</div>
