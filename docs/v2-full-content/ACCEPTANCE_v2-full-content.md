# ACCEPTANCE — `hello-rocm 2.0` 全书教科书级独立教程执行与整体验收台账

> **执行阶段**：6A 工作流 — 阶段 5（Automate） & 阶段 6（Assess 终验通过）  
> **工作分支**：`dev_1.0`（`master` 分支严格保持 `78a6f19` 未动）  
> **终验状态**：✅ **全部 12 项原子任务（共 57 篇中文核心页面 + 配套源码与单测门控）100% 验收通过**

---

## 一、三批次 12 项原子任务完成总表

| 批次 | 任务编号与名称 | 交付物清单 | 自动化单测与门控状态 | 验收状态 |
|:---|:---|:---|:---|:---|
| **Batch 1** | **Task 1**：彻底清理中英文 `00~05` 旧归档并升级测试门控 | 删除 `docs/zh/00~05`、`docs/en/00~05`、`src/infra`、`src/amd-yes`（通用案例迁入 `src/cases/`）；更新 `docs/en/index.md`、`CONTENT_GUIDE.md`、`src/hello-rocm-skill/*`、`tests/test_course_v2.py` | 零旧目录与零旧链接断言 `OK` | ✅ 已完成 |
| **Batch 1** | **Task 2**：第 0 章（导读）3 篇教科书级独立教程 | `docs/zh/00-guide/index.md`、`03-hardware-and-cloud.md`、`04-roadmap-and-skill.md` | 路由与内链 100% 通过 | ✅ 已完成 |
| **Batch 1** | **Task 3**：第 1 章（环境与三大引擎）6 篇教科书级独立教程 | `docs/zh/01-first-model/index.md`、`01-model-intuition.md`、`02-rocm-env-setup.md`、`03-llama-server.md`、`04-lemonade-server.md`、`05-vllm-server.md` | 路由与内链 100% 通过 | ✅ 已完成 |
| **Batch 1** | **Task 4**：第 2 章（WorkBuddy 三件套与编程工具）6 篇教科书级独立教程 | `docs/zh/02-connect-tools/index.md`、`02-workbuddy-setup.md`、`03-workbuddy-dashi-ppt.md`、`04-workbuddy-excel-and-cron.md`、`05-qoder-ide.md`、`06-codex-and-toy-cli.md` | 配置契约与内链 100% 通过 | ✅ 已完成 |
| **Batch 2** | **Task 5**：第 3 章（6 级提示词与私人笔记 RAG）4 篇独立教程 | `docs/zh/03-prompt-and-rag/index.md`、`01-prompt-6-levels.md`、`02-chunking-intuition.md`、`03-workbuddy-obsidian-rag.md` | 路由与内链 100% 通过 | ✅ 已完成 |
| **Batch 2** | **Task 6**：第 4 章（可共享知识管理 MCP 与 SKILL）4 篇独立教程 | `docs/zh/04-mcp-and-skills/index.md`、`02-knowledge-mcp-server.md`、`03-custom-office-skill.md`、`04-safety-guardrails.md` | MCP 安全单测与内链 100% 通过 | ✅ 已完成 |
| **Batch 2** | **Task 7**：第 5 章（中文 LoRA 微调与人设分身）5 篇独立教程 | `docs/zh/05-fine-tune/index.md`、`02-lora-and-ruozhiba.md`、`03-office-json-sft.md`、`04-eval-before-after.md`、`05-weclone-and-xhs.md` | 中文评测单测与内链 100% 通过 | ✅ 已完成 |
| **Batch 2** | **Task 8**：第 6 章（Quark 量化对比与零 C++ 算子加速）5 篇独立教程 | `docs/zh/06-optimize-speed/index.md`、`02-quark-before-after.md`、`03-zero-cpp-operators.md`、`04-agent-prefix-caching.md`、`05-hyperloom-auto-opt.md` | Quark 测速单测与内链 100% 通过 | ✅ 已完成 |
| **Batch 3** | **Task 9**：第 7 章（成功故事、五件套作品集与 STAR 简历）3 篇独立教程 | `docs/zh/07-portfolio-resume/index.md`、`02-portfolio-5-pieces.md`、`03-star-resume-interview.md` | 路由与内链 100% 通过 | ✅ 已完成 |
| **Batch 3** | **Task 10 & 11**：第 8 章（15 篇独立五要素应用案例）17 篇独立文档 | `docs/zh/08-fun-cases/index.md`、`_CASE_TEMPLATE.md`、`case-01-*.md` ～ `case-15-*.md` | 15 篇五要素章节+FAQ断言 `OK` | ✅ 已完成 |
| **Batch 3** | **Task 9 续**：第 9 章（AMD 官方开源生态、科研案例与自包含架构字典）3 篇独立教程 | `docs/zh/09-amd-ecosystem/index.md`、`02-research-and-industry-cases.md`、`03-hardware-dictionary-and-cloud.md` | 路由与图片静态资源校验 `OK` | ✅ 已完成 |
| **Batch 3** | **Task 12**：全站配置同步与阶段 6 整体验收交付 | `docs/.vitepress/config.mts`、`README.md`、`FINAL_v2-full-content.md`、`TODO_v2-full-content.md` | **`Ran 9 tests in 0.067s — OK`** | ✅ 已完成 |

---

## 二、阶段 6（Assess）整体验收核查清单

- [x] **所有需求已实现**：
  - 旧版 `00-environment ~ 05-amd-yes` 中英文目录及 `README.md` 折叠旧索引 100% 删除，全站无任何遗留断链；
  - 四大主体分档模型（🟢 `Gemma 4 E4B`、🔵 `Qwen3.5-9B` & `Gemma 4 12B`、🟣 `Qwen3.6-35B-A3B`）贯穿全书；
  - 推理引擎严格收敛为 **`llama-server`、`lemonade-server`、`vLLM`** 三大标准服务；
  - 第 1 章完整配备跨平台（Win11 / Ubuntu 24.04 / 云算力）ROCm 10 环境搭建与 4 步矩阵乘体检；
  - 第 2 章首发重点推出 WorkBuddy（唯一使用 `dashi-ppt-skill` 大师 PPT + 自动填 Excel + 定时任务，对齐 Datawhale WorkBuddy 教程）以及 `Qoder`、`Codex CLI` 与 `toy-cli`；
  - 第 3～4 章拆分为独立两章，实现「个人知识库封装为可给他人调用的知识管理 MCP Server」；
  - 第 5 章选修仅保留配方 A（`WeClone` 微信数字分身 / 小红书种草体），一对一导流 `happy-llm`；
  - 第 6 章实现单模型 `AMD Quark` 量化前后体积/显存/速度对比、对称 KV Cache 避坑、零 C++ 算子唤醒与 `Hyperloom` 自动寻优，一对一导流 `hello-gpu`；
  - 第 8 章 15 个趣味应用拆分为 `case-01` ～ `case-15` 共 15 篇独立五要素标准教程；
  - 第 9 章完整收录 AMD 官方开源明星项目、前沿行业案例与自包含全系 `gfx` 硬件字典。
- [x] **所有自动化测试通过**：`PYTHONPATH=. /opt/anaconda3/bin/python3 tests/test_course_v2.py -v` 共 9 项核心与门控测试全部 `OK`。
- [x] **安全规范完全满足**：所有涉及第三方 API Key（如高德地图、飞书机器人）均强制使用本地 `.env` 管理且通过 `.gitignore` 屏蔽，仓库零明文敏感凭据。
