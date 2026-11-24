# CONSENSUS：hello-rocm 2.0 课程定位重构与大纲规划共识文档

> **当前阶段**：6A 工作流 - 阶段 1：Align（最终共识 · 0～9 章定稿版）  
> **任务名称**：`course-refactor`  
> **状态**：所有关键边界已锁定（三大引擎、办公三件套、Qoder、拆章为 0～9 章、课程双向引用与导流、收官第 9 章 AMD 生态百宝箱）

---

## 一、明确的需求描述与核心定位共识

### 1.1 课程品牌重塑与生态角色

- **项目名称**：`hello-rocm`
- **主标语（Slogan）**：**每个 AI 小白，从零开始，动手实践的第一站。**
- **副标题（手册定位）**：**基于 AMD 平台的 AI 与 Agentic AI 入门实践手册（全栈总览先导课）。**
- **贯穿全书的成功故事**：「我把自己的 AMD 电脑，变成了学习和工作的 AI 助手。」
- **大纲三大必含顶层模块**：
  1. **用户学习体验旅程（6 站心智跃迁与获得感曲线）**；
  2. **本课程与其他课程的双向引用与导流关系（一章一专精课映射）**；
  3. **完整课程章节大纲（第 0 章 ～ 第 9 章）**。

### 1.2 核心章节拆分与一章一课映射规范（第 0～9 章）

- **底层推理服务收敛**：全书仅使用 **`llama-server`（`llama.cpp`）**、**`vLLM`** 与 **`lemonade-server`** 拉起本地 `/v1` 服务。
- **0～9 章结构与外部课程引用/导流映射**：
  - **第 0 章｜导读**：痛点价值、成功故事、四档硬件自检、6 站体验旅程与课程引用全景图。
  - **第 1 章｜数据类型直觉与三大引擎部署** ⇄ 链接 **[`self-llm`](https://github.com/datawhalechina/self-llm)**。
  - **第 2 章｜零代码办公先行（WorkBuddy 三件套：`dashi-ppt-skill` 出 PPT + AI 自动填表 + AI 定时任务）+ `Qoder` + `Codex`** ⇄ 链接 **[Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**。
  - **第 3 章｜让助手读懂你：提示词闯关 + `WorkBuddy + Obsidian` 个人知识库（原第 3 章前半独立成章）** ⇄ 引用菜谱切分实验并导流 **[`all-in-rag`](https://github.com/datawhalechina/all-in-rag)**。
  - **第 4 章｜让助手长出手脚：可共享「知识管理 MCP」+ Agent Skills + 智能体实战（原第 3 章后半独立成章，交付「个人学习助手 v2.0」）** ⇄ 引用核心循环并导流 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)**。
  - **第 5 章｜能力不够做中文微调（弱智吧破冰 + 中文通知抽取 + `WeClone`/小红书网感分身 + 自建测试集评测）** ⇄ **链接 [`happy-llm`](https://github.com/datawhalechina/happy-llm)**（深入大模型底层原理、预训练与微调对齐；多模型食谱兼顾 `self-llm`）。
  - **第 6 章｜速度不够做量化与加速（`AMD Quark` 单模型量化前后对比 + 零 C++ 唤醒融合算子 + `Hyperloom` 自动寻优）** ⇄ 链接 **[`hello-gpu`](https://github.com/datawhalechina/hello-gpu)**。
  - **第 7 章｜收获时刻：成功故事、个人作品集五件套与简历成果锻造**。
  - **第 8 章｜学以致用：15 个趣味实战案例库**（办公三件套置顶，覆盖 8 大场景，统一五要素 + 新手试学 FAQ）。
  - **第 9 章｜拓展视野：AMD 优秀开源项目、前沿行业案例与开发者资源百宝箱（新增收官章）**：汇聚 `amd/gaia`、`Hyperloom`、`Quark`、`AITER`、`Primus`、`TheRock` 等官方项目，无损沉淀仓库 `Every Embodied`、`Torch-RecHub`、`Happy-LLM`、Strix Halo 极限部署与原 `03-infra` 算子实验室，附算力福利与硬件架构字典。

---

## 二、技术约束与分支保护

1. **分支隔离**：全部规划与重构工作在 **`dev_1.0`** 分支推进并推送到 `origin/dev_1.0`，保持本地 `master` 与 `origin/master` 100% 一致不污染。
2. **双平台四档硬件适配**：Windows 11 & Ubuntu 本地优先（绿/蓝/紫三档硬件 + 云端免费 AMD Radeon Cloud 兜底）。
3. **历史外链零 404**：原仓库硬核内容归入第 9 章进阶案例与实验室，保持原有路由兼容。
