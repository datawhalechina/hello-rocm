# ACCEPTANCE — `hello-rocm 2.0` 全书教科书级独立教程执行与验收台账

> **执行阶段**：6A 工作流 — 阶段 5（Automate） & 阶段 6（Assess）  
> **工作分支**：`dev_1.0`（严格保护 `master` 分支不变）  
> **批次节奏**：按阶段 4 用户确认的 **3 个大批次（Batch 1 → Batch 2 → Batch 3）** 逐步推进、逐批自测提交汇报。

---

## 一、三批次原子任务执行进度表

| 批次 | 任务编号与名称 | 交付物清单 | 自动化单测与门控状态 | 当前状态 |
|:---|:---|:---|:---|:---|
| **Batch 1** | **Task 1**：彻底清理中英文 `00~05` 旧归档并升级测试门控 | 删除 `docs/zh/00~05`、`docs/en/00~05`、`src/infra`、`src/amd-yes`（通用案例迁入 `src/cases/`）；更新 `docs/en/index.md`、`CONTENT_GUIDE.md`、`src/hello-rocm-skill/*`、`tests/test_course_v2.py` | 8 项单测与全站零旧链接门控 `OK` | ✅ 已完成 |
| **Batch 1** | **Task 2**：第 0 章（导读）3 篇教科书级独立教程 | `docs/zh/00-guide/index.md`、`03-hardware-and-cloud.md`、`04-roadmap-and-skill.md` | 路由与内链 100% 通过 | ✅ 已完成 |
| **Batch 1** | **Task 3**：第 1 章（环境与三大引擎）6 篇教科书级独立教程 | `docs/zh/01-first-model/index.md`、`01-model-intuition.md`、`02-rocm-env-setup.md`、`03-llama-server.md`、`04-lemonade-server.md`、`05-vllm-server.md` | 路由与内链 100% 通过 | ✅ 已完成 |
| **Batch 1** | **Task 4**：第 2 章（WorkBuddy 三件套与编程工具）6 篇教科书级独立教程 | `docs/zh/02-connect-tools/index.md`、`02-workbuddy-setup.md`、`03-workbuddy-dashi-ppt.md`、`04-workbuddy-excel-and-cron.md`、`05-qoder-ide.md`、`06-codex-and-toy-cli.md` | 路由、配置契约与内链 100% 通过 | ✅ 已完成 |
| **Batch 2** | **Task 5**：第 3 章（6 级提示词与私人笔记 RAG）4 篇独立教程 | `docs/zh/03-prompt-and-rag/index.md`、`01-prompt-6-levels.md`、`02-chunking-intuition.md`、`03-workbuddy-obsidian-rag.md` | 路由与内链 100% 通过 | ✅ 已完成 |
| **Batch 2** | **Task 6**：第 4 章（可共享知识管理 MCP 与 SKILL）4 篇独立教程 | `docs/zh/04-mcp-and-skills/index.md`、`02-knowledge-mcp-server.md`、`03-custom-office-skill.md`、`04-safety-guardrails.md` | MCP 单测与全站内链 100% 通过 | ✅ 已完成 |
| **Batch 2** | **Task 7**：第 5 章（中文 LoRA 微调与人设分身）5 篇独立教程 | `docs/zh/05-fine-tune/index.md`、`02-lora-and-ruozhiba.md`、`03-office-json-sft.md`、`04-eval-before-after.md`、`05-weclone-and-xhs.md` | 中文评测单测与内链 100% 通过 | ✅ 已完成 |
| **Batch 2** | **Task 8**：第 6 章（Quark 量化对比与零 C++ 算子加速）5 篇独立教程 | `docs/zh/06-optimize-speed/index.md`、`02-quark-before-after.md`、`03-zero-cpp-operators.md`、`04-agent-prefix-caching.md`、`05-hyperloom-auto-opt.md` | Quark 报告单测与内链 100% 通过 | ✅ 已完成 |
| **Batch 3** | **Task 9**：第 7 章（成功故事、五件套作品集与 STAR 简历）3 篇独立教程 | `docs/zh/07-portfolio-resume/index.md`、`02-portfolio-5-pieces.md`、`03-star-resume-interview.md` | 待执行 | ⏳ 待执行 |
| **Batch 3** | **Task 10**：第 8 章（15 篇独立五要素应用案例）17 篇独立教程 | `docs/zh/08-fun-cases/index.md`、`_CASE_TEMPLATE.md`、`case-01-*.md` ～ `case-15-*.md` | 待执行 | ⏳ 待执行 |
| **Batch 3** | **Task 11**：第 9 章（AMD 官方开源生态、科研案例与自包含架构字典）3 篇独立教程 | `docs/zh/09-amd-ecosystem/index.md`、`02-research-and-industry-cases.md`、`03-hardware-dictionary-and-cloud.md` | 待执行 | ⏳ 待执行 |
| **Batch 3** | **Task 12**：全站配置同步与阶段 6 整体验收交付 | `docs/.vitepress/config.mts`、`README.md`、`FINAL_v2-full-content.md`、`TODO_v2-full-content.md` | 待执行 | ⏳ 待执行 |
