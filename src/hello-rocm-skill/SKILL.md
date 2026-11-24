---
name: hello-rocm
description: >-
  hello-rocm 2.0 伴学助教技能：基于 AMD 平台的 AI 与 Agentic AI 入门实践手册（全栈总览先导课）。
  当用户询问 AMD GPU/APU 选型、ROCm 10 环境安装、四大主体模型（Gemma 4 E4B/12B、Qwen3.5-9B、Qwen3.6-35B-A3B）、
  三大推理引擎（llama-server / lemonade-server / vLLM）、WorkBuddy 办公三件套（dashi-ppt-skill / 自动填表 / 定时任务）、
  Qoder / Codex 接入、Obsidian 知识库与可共享「知识管理 MCP」、中文 LoRA 微调或 AMD Quark / Hyperloom 量化加速时自动激活。
---

# hello-rocm 2.0 伴学助教 Skill

本 Skill 索引 **hello-rocm 2.0** 纯净架构的完整章节地图、硬件选型与排障入口（所有路径均相对于仓库根目录）。

## 一、核心技术基线

1. **四大分档主体模型**：
   - 🟢 **入门档（8G–12G 独显 / 核显 & XDNA2 NPU）**：`Gemma 4 E4B`（`google/gemma-4-E4B-it`）
   - 🔵 **甜点档（16G–24G 独显）**：`Qwen3.5-9B`（中文办公/微调主力）与 `Gemma 4 12B`（推理/`Qoder` 编程主力）
   - 🟣 **旗舰档（64G–128G 统一内存 AI PC / 云）**：`Qwen3.6-35B-A3B`（35B 总参 / 3B 激活 MoE 旗舰主力）
2. **三大标准推理服务引擎**：仅使用 `llama-server`（端口 `8080`）、`lemonade-server`（端口 `13305`）、`vLLM`（端口 `8000`），接入 Agent 时强制开启 `32768` 上下文与工具调用参数。

## 二、第 0～9 章精准导航索引

| 用户想解决的问题 | 引导阅读的仓库路径 | 进阶导流专精课 |
|:---|:---|:---|
| 查硬件档位、领免费云算力、看全局学习旅程 | `docs/zh/00-guide/` & `docs/zh/cloud/amd-radeon-cloud.md` | 总览先导 |
| 算显存账、装 ROCm 10 + `uv` + PyTorch 2.13、拉起三大服务引擎 | `docs/zh/01-first-model/` | `datawhalechina/self-llm` |
| 接入 WorkBuddy（`dashi-ppt-skill` 出 PPT / 自动填表 / 定时办公）、`Qoder`、`Codex` | `docs/zh/02-connect-tools/` & `src/configs/` | Datawhale WorkBuddy 教程 |
| 练 6 关提示词、菜谱切分实验、搭 `WorkBuddy + Obsidian` 双链笔记库 | `docs/zh/03-prompt-and-rag/` | `datawhalechina/all-in-rag` |
| 把个人知识库封装成可共享给别人调用的「知识管理 MCP」与自定义 `SKILL.md` | `docs/zh/04-mcp-and-skills/` & `src/main-assistant/knowledge_mcp/server.py` | `datawhalechina/hello-agents` |
| 做第一次中文 LoRA 微调（弱智吧 / 通知抽取 / `WeClone` 分身）与测试集评测 | `docs/zh/05-fine-tune/` & `src/main-assistant/eval/eval_before_after_zh.py` | `datawhalechina/happy-llm` |
| 用 `AMD Quark` 对比单模型量化前后速度体积、开启 `AITER`/Prefix Cache 与 `Hyperloom` | `docs/zh/06-optimize-speed/` & `src/main-assistant/quant_bench/quark_before_after_bench.py` | `datawhalechina/hello-gpu` |
| 打包个人作品集五件套、打磨 STAR 简历项目与准备面试 | `docs/zh/07-portfolio-resume/` | 毕业深造驿站 |
| 选做 8 大场景 15 个趣味实战案例（`case-01` ～ `case-15`） | `docs/zh/08-fun-cases/` | 全场景实战 |
| 查 AMD 官方明星开源项目、高阶行业案例与全系 GPU `gfx` 架构速查表 | `docs/zh/09-amd-ecosystem/` | 生态百宝箱 |
