---
layout: home

hero:
  name: hello-rocm
  text: ROCm tutorials and practice cases for AMD GPUs
  tagline: A structured learning path for AMD ROCm in LLM workflows, from environment setup and deployment to fine-tuning and operator optimization.
  image:
    src: /images/head.png
    alt: hello-rocm
  actions:
    - theme: brand
      text: Start with environment setup
      link: /environment/
    - theme: alt
      text: View practice cases
      link: /amd-yes/
    - theme: alt
      text: 简体中文
      link: /zh/

features:
  - title: Unified environment baseline
    details: Set up ROCm, PyTorch, and uv first so deployment, fine-tuning, and Infra tutorials share the same foundation.
    link: /environment/
  - title: LLM deployment
    details: Deploy models on AMD GPUs with LM Studio, vLLM, Ollama, llama.cpp, and related ROCm workflows.
    link: /deploy/
  - title: LLM fine-tuning
    details: Follow ROCm fine-tuning examples for models such as Qwen3 and Gemma4.
    link: /fine-tune/
  - title: Operator optimization
    details: Learn AMD AI hardware, the ROCm software stack, HIP operators, PyTorch custom operators, and performance-oriented practices.
    link: /infra/
  - title: References
    details: Curated ROCm, AMD GPU, and AI development resources for deeper reading.
    link: /references
  - title: AMD practice cases
    details: Community examples covering local agents, model training, vision applications, and developer tools on AMD platforms.
    link: /amd-yes/
---

## Recommended Path

1. Read [Environment](/environment/) first and finish the ROCm, PyTorch, and Python toolchain setup.
2. Move to [Deploy](/deploy/) and validate your environment with LM Studio, vLLM, or another inference workflow.
3. Continue with [Fine-tune](/fine-tune/) to understand LoRA fine-tuning on ROCm.
4. Explore [Infra](/infra/) if you want to go deeper into low-level performance and operator development.
5. Check [AMD Practice](/amd-yes/) when you need complete project examples.

## Content Map

| Section | Content |
| --- | --- |
| [Environment](/environment/) | ROCm installation, configuration, validation, and GPU architecture references |
| [Deploy](/deploy/) | Multi-framework local deployment for models such as Qwen3 and Gemma4 |
| [Fine-tune](/fine-tune/) | LLM fine-tuning notes and examples on ROCm |
| [Infra](/infra/) | AMD AI hardware, ROCm software stack, HIP operators, and PyTorch custom operators |
| [References](/references) | ROCm and AMD AI ecosystem resources |
| [AMD Practice](/amd-yes/) | Application cases and community projects for AMD platforms |

## Who This Is For

- Learners who have an AMD GPU and want to run or fine-tune LLMs locally.
- Developers who want a structured ROCm workflow instead of scattered setup notes.
- Practitioners who want to move from inference deployment into operator development and performance analysis.
