# ALIGNMENT：彻底移除旧版 00~05 目录并以 6A 工作法全量深写 2.0 教程对齐文档

> **当前阶段**：6A 工作流 - 阶段 1：Align（需求对齐与关键决策澄清）  
> **任务名称**：`v2-full-content`  
> **执行分支**：`dev_1.0`（本地与云端 `master` 保持 `78a6f19` 零污染）

---

## 一、原始需求与背景

- **用户核心指令**：
  > `00-environment ~ 05-amd-yes/  # 专项拓展档案库（见下方折叠索引）` 相关内容就直接删除了，然后用 6A 工作法逐步去写所有的内容。
- **已锁定的前置课程基线**：
  - **课程定位**：`hello-rocm` —— 每个 AI 小白，从零开始，动手实践的第一站。基于 AMD 平台的 AI 与 Agentic AI 入门实践手册（全栈总览先导课）。
  - **四大分档主体模型**：
    - 🟢 入门档（8G–12G 独显 / 核显 & NPU）：**`Gemma 4 E4B`**（`google/gemma-4-E4B-it`）
    - 🔵 甜点档（16G–24G 独显）：**`Qwen3.5-9B`**（`Qwen/Qwen3.5-9B`）& **`Gemma 4 12B`**（`google/gemma-4-12B-it`）
    - 🟣 旗舰档（64G–128G 统一内存 AI PC / 云）：**`Qwen3.6-35B-A3B`**（`Qwen/Qwen3.6-35B-A3B`）
  - **三大标准推理服务引擎**：仅使用 **`llama-server`（`llama.cpp`）**、**`lemonade-server`**、**`vLLM`**（剔除 LM Studio / Ollama 等分支）。
  - **0～9 章完整结构与一章一课导流**：
    - 第 0 章：导读、6 站用户学习体验旅程、四档硬件自检、课程引用导航
    - 第 1 章：数据类型三本账、ROCm 10 环境完整配置、三大引擎部署（→ `self-llm`）
    - 第 2 章：WorkBuddy 办公三件套（唯一锁定 `dashi-ppt-skill` 出可编辑 `.pptx` + AI 自动填表 + 定时任务）、`Qoder`、`Codex`（→ Datawhale WorkBuddy 教程）
    - 第 3 章：提示词 6 关闯关、菜谱切分实验、`WorkBuddy + Obsidian` 双链笔记库（→ `all-in-rag`）
    - 第 4 章：可共享给别人调用的「知识管理 MCP」、Agent Skills（`SKILL.md`）与安全护栏（→ `hello-agents`）
    - 第 5 章：中文模型微调（弱智吧破冰 + 中文通知抽取 + `WeClone`/小红书网感分身 + 自建测试集评测）（→ `happy-llm`）
    - 第 6 章：`AMD Quark` 单模型量化前后对比、零 C++ 唤醒融合算子、Prefix Caching 与 `Hyperloom` 自动优化（→ `hello-gpu`）
    - 第 7 章：成功故事、个人作品集五件套与简历面试指南
    - 第 8 章：15 个趣味实战案例库（办公置顶 · 8 大场景 · 统一五要素模板 + 新手试学 FAQ）
    - 第 9 章：AMD 优秀开源项目、前沿行业案例与开发者资源百宝箱

---

## 二、现有仓库上下文与删除影响面分析

通过对当前 `dev_1.0` 仓库执行全量依赖扫描，发现彻底删除旧版 `00-environment ~ 05-amd-yes/` 涉及以下 5 类关联影响，必须在架构与执行阶段一并闭环处理：

1. **目录与导航净空**：
   - `docs/zh/` 下目前新旧目录交错并存（如 `01-deploy` 与 `01-first-model`、`02-connect-tools` 与 `02-fine-tune`、`05-amd-yes` 与 `05-fine-tune`），删除 `docs/zh/00-environment` ～ `docs/zh/05-amd-yes` 后，`docs/zh/` 将只保留清爽纯粹的 `00-guide` ～ `09-amd-ecosystem`（外加公共支撑页 `cloud/` 与 `learning/`）。
   - `README.md` 底部折叠索引、`docs/.vitepress/config.mts` 顶部“专项档案库”下拉菜单与旧侧边栏需同步移除。
2. **有价值技术细节的内化吸收（零外链悬空）**：
   - 旧目录中与 2.0 主线强相关的精华内容（例如：`Gemma 4` / `Qwen3.6-35B-A3B` 的 Lemonade/FastFlowLM 实测细节、GPU `gfx` 架构速查表、`toy-cli`、`minesweeper-agent`、高德 MCP 旅行规划、`openclaw` 等源码级步骤），必须在删除旧目录前**完整重构内化**到新版第 1、8、9 章独立文档中，彻底杜绝指向已删除旧路径的死链。
3. **关联工具与辅助页面同步修复**：
   - `src/hello-rocm-skill/`（内置伴学助教 Skill）目前内部索引仍指向 `00-environment` ～ `05-amd-yes`，需同步升级指向新版 `00-guide` ～ `09-amd-ecosystem`。
   - `docs/zh/cloud/amd-radeon-cloud.md` 与 `docs/zh/learning/` 中指向旧目录的链接需同步更新为新章节路径。
4. **公共图片资源保留与清理**：
   - `docs/public/images/` 中被新版第 0～9 章复用的架构图、终端回显截图保留并按新目录引用。
5. **从“骨架速览稿”升级为“全量深度实战教材”**：
   - 上一轮快速搭建的 17 篇文档多为提纲式合辑（如 `1.3~1.5` 三大引擎合为一篇、15 个案例压缩在两篇合辑中）。按 6A 工作法全量重写时，需将每一小节与每一个趣味案例拆解为**独立、详实、零跳步的正式教程文档**。

---

## 三、智能决策清单与待确认关键点

### 已基于工程最佳实践自主决策项
- **图片与有用实验源码处理**：保留 `src/configs/`、`src/main-assistant/`、`src/hello-rocm-skill/`（升级路径引用）及第 8 章案例实际用到的 Python/Notebook 源码（统一归整至 `src/cases/`），移除无关冗余引用。
- **质量门控升级**：升级 `tests/test_course_v2.py`，强制断言 `docs/zh/` 下不存在 `00-environment ~ 05-amd-yes` 旧目录及任何指向旧路径的残留字符串。

### 需与项目负责人对齐确认的 3 个关键决策点（中断询问）
1. **英文旧目录（`docs/en/00-environment ~ 05-amd-yes`）与旧底层源码（`src/infra` 等）的清理边界**；
2. **全量教程的文件拆分颗粒度**（每小节独立成篇 + 第 8 章 15 个案例独立拆分为 15 篇完整五要素文档）；
3. **阶段 5（Automate）逐步编写的人工检视节奏**（逐章分批确认 vs 按原子任务连续分章深写提交）。
