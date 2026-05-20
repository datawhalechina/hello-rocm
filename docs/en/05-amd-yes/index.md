<div align=center>
  <h1>AMD-YES 🎮</h1>
  <strong>A Curated Collection of Cool Projects on AMD GPUs</strong>
</div>

<div align="center">

*Making AI more fun, unleashing creativity* ✨

[Back to Home](/)

</div>

## 🚀 Quick Navigation

| Local Run (Single GPU) | Cluster Deployment (Multi-GPU) |
|---------|---------|
| [🧸 toy-cli](/05-amd-yes/toy-cli) ✅️ | [🚀 happy-llm](/05-amd-yes/happy-llm/) ✅️ |
| [🎮 WeChat Jump](/05-amd-yes/wechat-jump) ✅️| |
| [🎭 Chat-Huanhuan](/05-amd-yes/huanhuan-chat) ✅️| |
| [✈️ Smart Travel Planner](/05-amd-yes/hello-agents/) ✅️| |
| [🦞 OpenClaw Private Assistant](/05-amd-yes/openclaw) ✅️| |
| [📚 Torch-RecHub Recommender Practice](/05-amd-yes/torch-rechub/) ✅️| |

> ✅ Supported | 🚧 In progress 

## Introduction

AMD-YES brings together practical project examples built on AMD GPU and ROCm platform. Whether you want to quickly experience LLM applications, learn computer vision, or dive deep into distributed training, you can find suitable example projects here.

This module is divided into two stages: First, quickly get started with various applications on a single local GPU, then advance to cluster deployment and distributed training. Each project includes complete code and detailed tutorial guidance.

### AMD-YES Learning Path

<div align='center'>
    <img src="../../../public/images/05-amd-yes/overview/amd_yes_project_overview_en.png" alt="Figure 5.1 AMD-YES project learning path overview" width="95%">
</div>

## Project List

### Local Run (Single GPU)

Suitable for individual developers to quickly get started – you can do it all with just one AMD GPU!

---

#### 🧸 toy-cli - Lightweight LLM Terminal Assistant

[toy-cli](https://github.com/KMnO4-zx/toy-cli) is a simplified code Agent that provides a minimal command-line interface for calling LLM APIs.

- **Target Audience**: Beginners, users wanting to quickly learn API calls
- **Difficulty Level**: ⭐
- **Estimated Time**: 30 minutes

📖 [Start Learning toy-cli](/05-amd-yes/toy-cli)

---

#### 🎮 YOLOv10 WeChat Jump

[YOLOv10 WeChat Jump](https://github.com/KMnO4-zx/wechat-jump) is an automated game AI based on YOLOv10 object detection, which automatically recognizes jump targets in real-time and calculates distances accurately.

- **Target Audience**: Developers interested in computer vision and game AI
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1 hour

📖 [Start Learning WeChat Jump](/05-amd-yes/wechat-jump)

---

#### 🎭 Chat-Huanhuan - Palace Language Model

[Chat-Huanhuan](https://github.com/KMnO4-zx/huanhuan-chat) is a LoRA fine-tuned model trained on dialogue excerpts from the TV series, perfectly imitating Huanhuan's tone and speaking style.

- **Target Audience**: Developers wanting to learn model fine-tuning and LoRA techniques
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1.5 hours

📖 [Start Learning Chat-Huanhuan](/05-amd-yes/huanhuan-chat)

---

#### ✈️ Smart Travel Planner

[Smart Travel Planner](/05-amd-yes/hello-agents/) is an intelligent Agent application based on the HelloAgents framework, integrating MCP protocol to call Amap API, running large models locally on AMD GPU.

- **Target Audience**: Developers wanting to learn Agent frameworks and MCP protocol
- **Difficulty Level**: ⭐⭐⭐
- **Estimated Time**: 2 hours

📖 [Start Learning Smart Travel Planner](/05-amd-yes/hello-agents/smart-travel-planner/amd395-helloagents-smart-travel-planner)

---

#### 🦞 OpenClaw - Fully Private Local AI Agent Platform

[OpenClaw](https://github.com/ValueCell-ai/ClawX) is a unified message processing and AI agent platform. Deploy a large model locally on AMD 395 Max for a fully private personal AI assistant.

- **Target Audience**: Developers wanting to learn AI Agent platforms and local privacy deployment
- **Difficulty Level**: ⭐⭐⭐
- **Estimated Time**: 1.5 hours

📖 [Start Learning OpenClaw](/05-amd-yes/openclaw)

---

#### 📚 Torch-RecHub - Recommender Systems Practice

[Torch-RecHub](https://github.com/datawhalechina/torch-rechub) recommender systems practice covers CTR ranking, sequential interest modeling, matching, multi-task learning, experiment tracking, and model export.

- **Target Audience**: Developers who want to learn recommender system modeling and training on AMD GPU / ROCm environments
- **Difficulty Level**: ⭐⭐⭐
- **Estimated Time**: 3+ hours

📖 [Start Learning Torch-RecHub Recommender Practice](/05-amd-yes/torch-rechub/)

---

### Cluster Deployment (Multi-GPU)

Advanced usage – distributed training and deployment!

---

#### 🚀 Happy-LLM - Train Large Models from Scratch

[Happy-LLM](https://github.com/datawhalechina/happy-llm) provides comprehensive tutorials for distributed multi-machine multi-GPU training, including core principles of large models and hands-on training implementation.

- **Target Audience**: Advanced developers wanting to gain deep knowledge of large model training
- **Difficulty Level**: ⭐⭐⭐
- **Estimated Time**: 3+ hours

📖 [Start Learning Happy-LLM](/05-amd-yes/happy-llm/)

---

## Environment Requirements

### Hardware Requirements

- AMD GPU (ROCm-compatible GPUs, such as RX 7000 series, MI series, etc.)
- Recommended VRAM: 8 GB or more

### Software Requirements

- Operating System: Linux (Ubuntu 22.04+) or Windows 11
- ROCm 7.12.0/7.2.0 or higher version
- Python 3.10~3.12

## Frequently Asked Questions

<details>
<summary>Q: How do I verify if my AMD GPU supports ROCm?</summary>

Please refer to the [ROCm Official Support List](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) to check supported GPU models.

</details>

<details>
<summary>Q: What is the recommended learning order for these projects?</summary>

It is recommended to learn in the following order:
1. Start with `toy-cli` first to get familiar with basic LLM API calls
2. Choose projects of interest to dive deeper (game AI, dialogue models, Agent applications)
3. After completing basic projects, challenge yourself with `Happy-LLM` for advanced distributed training

</details>

## Reference Resources

- [self-llm Complete Tutorial](https://github.com/datawhalechina/self-llm)
- [HelloAgents AI Agent Framework](https://github.com/datawhalechina/hello-agents)
- [ROCm Official Documentation](https://rocm.docs.amd.com/)
- [hello-rocm Main Project](../)

---

<div align="center">

**Contributions of more project examples are welcome!** 🎉

[Submit Issue](https://github.com/datawhalechina/hello-rocm/issues) | [Submit PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
