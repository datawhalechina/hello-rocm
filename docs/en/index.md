---
layout: home

hero:
  name: hello-rocm
  text: Your First Hands-On Station for AI & Agentic AI on AMD.
  tagline: Turn your AMD PC into an everyday AI assistant for study and work — from local deployment (Gemma 4 E4B/12B, Qwen3.5-9B, Qwen3.6-35B-A3B) to WorkBuddy office automation, shareable Knowledge MCP, Chinese LoRA fine-tuning, and AMD Quark / Hyperloom acceleration.
  image:
    src: /images/logo.png
    alt: hello-rocm
  actions:
    - theme: brand
      text: 🚀 Read Full 2.0 Guide (简体中文)
      link: /zh/
    - theme: alt
      text: 📖 Start Chapter 0 (中文导读)
      link: /zh/00-guide/
    - theme: alt
      text: 🗺️ ROCm Atlas
      link: /atlas/rocm/index.html
      target: _blank
      rel: noreferrer

features:
  - title: 🚀 4 Tiered Flagship Models × 3 Standard Engines
    details: Standardized on Gemma 4 E4B (8G entry), Qwen3.5-9B & Gemma 4 12B (16G+ sweet-spot), and Qwen3.6-35B-A3B (64G+ unified memory flagship) served via llama-server, lemonade-server, and vLLM.
    link: /zh/01-first-model/
  - title: 💼 Zero-Code Office & Coding First
    details: Connect local models directly to WorkBuddy (editable .pptx via dashi-ppt-skill, auto Excel filling, RRULE scheduled tasks), Qoder IDE, and Codex CLI.
    link: /zh/02-connect-tools/
  - title: 🧠 Personal RAG & Shareable Knowledge MCP
    details: Build a bi-directional Obsidian knowledge base and wrap it with FastMCP into a reusable Knowledge Management MCP Server alongside progressive Agent Skills.
    link: /zh/04-mcp-and-skills/
  - title: ⚡ LoRA Fine-Tuning, AMD Quark & Hyperloom
    details: Fine-tune Chinese notification extraction & digital personas, benchmark before/after quantization with AMD Quark, unlock AITER kernels, and auto-optimize via AMD-AGI/Hyperloom.
    link: /zh/06-optimize-speed/
---

## 🌐 Language Note

**hello-rocm 2.0** has been completely redesigned into a clean 10-chapter (**Chapters 0–9**) hands-on curriculum. The complete edition is currently published in Simplified Chinese at **[/zh/](/zh/)**, with verified client configurations available under `src/configs/` and runnable MCP / evaluation scripts under `src/main-assistant/`.
