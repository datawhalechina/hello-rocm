# FINAL：hello-rocm 2.0 全栈总览先导课重构总结报告

> **项目分支**：`dev_1.0`（已完整同步至远程 `origin/dev_1.0`，`master` 保持原始干净状态）

---

## 一、交付成果全景摘要

本次重构按照 **6A 工作流规范（Align → Architect → Atomize → Approve → Automate → Assess）**，完成了 `hello-rocm` 从“底层技术栈字典”向 **“零基础小白 AI 与 Agentic AI 全栈总览先导课”** 的体系升级与全文落地：

1. **顶层课程总纲与排版 PDF**
   - 完整课程总大纲 `docs/course-refactor/CURRICULUM_OUTLINE_v2.md` 及 A4 精排 8 页 PDF `docs/course-refactor/hello-rocm-2.0-课程总大纲.pdf`。
   - 明确阐述了 **6 站用户学习体验旅程**、**一章一专精课双向引用与导流矩阵** 及 **第 0～9 章完整章节体系**。
2. **第 0～9 章全量中文主线教程与 VitePress 导航（共 17 篇核心教程文档）**
   - `00-guide/`：导读、体验旅程、绿/蓝/紫/云四档硬件 3 分钟自检与课程引用全景图；
   - `01-first-model/`：数据类型与显存“三本明白账” + `llama-server` / `lemonade-server` / `vLLM` 三大标准引擎极速部署（→ `self-llm`）；
   - `02-connect-tools/`：**第一步重点推出** WorkBuddy 办公三件套（`dashi-ppt-skill` 导出可编辑 `.pptx` + AI 自动填表 + 定时办公自动化）+ `Qoder` IDE + `Codex CLI`（→ Datawhale WorkBuddy 教程）；
   - `03-prompt-and-rag/`：6 关本地提示词闯关 + 黑暗料理菜谱切分实验 + `WorkBuddy + Obsidian` 双链笔记库（→ `all-in-rag`）；
   - `04-mcp-and-skills/`：把个人知识库制作成可给别人调用的 **「知识管理 MCP」** + 自定义省显存办公 `SKILL.md`（→ `hello-agents`）；
   - `05-fine-tune/`：弱智吧 5 分钟破冰 + 中文通知抽取主线微调 + `WeClone`/小红书网感分身 + 自建测试集评测（→ `happy-llm`）；
   - `06-optimize-speed/`：`AMD Quark` 单模型量化前后对比 + 零 C++ 唤醒融合算子 + Prefix Caching + `AMD-AGI/Hyperloom` 自动寻优（→ `hello-gpu`）；
   - `07-portfolio-resume/`：成功故事模板、作品集五件套打包清单、STAR 简历项目经历与面试追问指南；
   - `08-fun-cases/`：覆盖 8 大场景的 **15 个趣味实战案例库**（办公置顶）、统一五要素模板 `_CASE_TEMPLATE.md` 与新手试学反馈模板；
   - `09-amd-ecosystem/`：汇聚 AMD 官方明星开源项目、仓库前沿科研行业案例（无损兼容全部历史路由）与算力架构百宝箱。
3. **配套已验证配置库、可运行源码与自动化测试套件**
   - `src/configs/`：`workbuddy-models.json`、`qoder-openai-compatible.json`、`codex-config.toml`；
   - `src/main-assistant/`：可共享 `knowledge_mcp/server.py`、自定义办公 `SKILL.md`、中文评测脚本 `eval_before_after_zh.py`、`AMD Quark` 前后测速报告生成器 `quark_before_after_bench.py`；
   - `tests/test_course_v2.py`：单元测试与全站路由/死链自动门控脚本（7/7 全部通过）。
