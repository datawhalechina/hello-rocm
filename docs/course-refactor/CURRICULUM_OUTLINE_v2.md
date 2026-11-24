# hello-rocm 2.0 完整课程总大纲（总览入口与生态导流定稿版）

---

## 一、课程核心定位、成果闭环与生态导流矩阵

- **课程名称**：`hello-rocm`
- **核心标语（Slogan）**：**每个 AI 小白，从零开始，动手实践的第一站。**
- **课程副标题（定位）**：**基于 AMD 平台的 AI 与 Agentic AI 入门实践手册（全栈总览先导课）。**
- **生态角色（先总览跑通闭环，再分流深造）**：
  - **`hello-rocm` 是面向零基础学习者的「AI 全景总览入口课」** —— 先带小白在一台 AMD 电脑上低成本体验「模型部署 → AI 办公 → 知识库与 Agent → 中文微调 → 量化与算子加速」全流程获得感；**每学完一个模块或全书结业后，精准导流至 Datawhale 各垂直领域王牌开源课程继续深度进修**：

| 在 `hello-rocm` 体验的入门模块 | 学完后进阶导流目标课程 | 进阶深造核心内容 |
|:---|:---|:---|
| **第 1 章（本地部署）& 第 4 章（模型微调）** | 🔗 **[`self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)** | 拓展百余款主流开源大模型的部署方案、多框架全参/LoRA 微调与进阶实操 |
| **第 2 章（WorkBuddy 零代码办公与自动化）** | 🔗 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)** | 深入解锁更多桌面办公工作流、复杂多表处理、自定义技能与自动化实战 |
| **第 3 章 & 第 7 章（个人知识库 RAG 方向）** | 🔗 **[`all-in-rag`（RAG 技术全栈指南）](https://github.com/datawhalechina/all-in-rag)** | 从个人笔记问答进阶到生产级文档解析、混合检索、重排（Rerank）与 GraphRAG |
| **第 3 章 & 第 7 章（MCP / Skills / 智能体方向）** | 🔗 **[`hello-agents`（从零开始构建智能体）](https://github.com/datawhalechina/hello-agents)** | 从调用工具进阶到手写 ReAct / Plan-and-Solve 框架、记忆机制与多智能体协同 |
| **第 5 章 & 进阶附录（GPU 架构与算子优化方向）** | 🔗 **[`hello-gpu`（GPU 架构与算子优化实战）](https://github.com/datawhalechina/hello-gpu)** | 从零 C++ 开关加速迈入硬核深水区：AMD GPU 硬件架构、HIP / Triton 算子编写与性能剖析 |

- **底层推理服务统一标准（全书收敛为三大标准引擎，拒绝选择困难）**：
  - 统一仅使用 **`llama-server`（`llama.cpp`）**、**`vLLM`** 与 **`lemonade-server`** 拉起本地 OpenAI 兼容服务（覆盖 AMD Radeon 独显、Ryzen AI 核显/NPU 与云算力）。
- **贯穿全书的成功故事**  
  > **“我把自己的 AMD 电脑，变成了学习和工作的 AI 助手。”**  
  > 完整记录一名普通学习者：从最低成本用 `llama-server` / `lemonade-server` / `vLLM` 跑起本地模型 ➡️ 接入 **WorkBuddy** 完成 AI PPT 生成（`dashi-ppt-skill`）、AI 自动填表与定时处理日常办公 ➡️ 接入 **Qoder** 与 **Codex** 辅助写代码 ➡️ 把个人笔记库封装成可给别人调用的 **知识管理 MCP** ➡️ 完成第一次中文微调（群通知提取 & 微信分身/小红书网感） ➡️ 用 **AMD Quark** 对比量化加速并用 **Hyperloom** 体验自动寻优 ➡️ 最终沉淀个人 AI 应用与简历成果，并找到自己在 Datawhale 课程体系中的下一站深造方向。
- **作品集与简历成果转化**
  - **沉淀五件套**：应用代码、交互演示、训练或微调记录（SwanLab）、效果评估报告（自建测试集前后对比 + 量化前后测速对比）、个人贡献说明。
  - **证明四大能力**：模型部署、数据处理、应用集成、科学评测。
  - **标准简历项目示例**：
    > *基于 AMD 平台完成小模型微调并构建本地学习助手，负责数据整理、模型服务与知识检索集成，通过自建测试集评估微调前后及应用效果。*
- **四档零焦虑硬件适配（Windows 11 & Ubuntu 本地优先 + 免费云算力兜底）**
  - 🟢 **入门档（8G–12G 独显 / 16G–32G 核显本）**：主跑 `0.6B ~ 4B (Q4_K_M)`
  - 🔵 **甜点档（16G–24G 独显，如 RX 7800 XT / 7900 / 9070 系列）**：主跑 `4B ~ 14B (Q4_K_M / Q8_0)`
  - 🟣 **旗舰档（64G–128G 统一内存 AI PC，如 Ryzen AI Max+ 395）**：主跑 `14B ~ 35B+ MoE` 超长上下文
  - ☁️ **云端兜底档（手头暂无 AMD 显卡）**：免费接入 **AMD Radeon Cloud** 全程实操

---

## 二、🚀 第一步重点推出内容：本地模型 + WorkBuddy 办公自动化三件套

打破“ROCm 教程太技术、不好学”的印象，**第一步重点首发内容**直接联动并链接至 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**，用本地免费 AMD 算力跑通人人每天刚需的三大办公场景：

1. **AI PPT 一键生成（唯一锁定开源王牌 `dashi-ppt-skill` 大师 PPT Skill）**
   - **开源仓库**：`github.com/chuspeeism/dashi-ppt-skill`（一行 `npx dashi-ppt-skill@latest` 装入 WorkBuddy）。
   - **核心亮点**：内置 12 套视觉主题 + 1,020 个商业/学术版式（SWOT、漏斗图、雷达图等），自带浏览器局部点选微调台，**支持一键导出在 PowerPoint / WPS 中真实可二次编辑的 `.pptx` 文件**（以及 PDF / 离线 HTML）。
2. **AI 自动填表（非结构化单据/资料 ➡️ 标准 Excel 报表）**
   - 扫描本地文件夹中的多份报销单、简历或原始明细表，严格按字段契约提取关键信息并自动追加填入目标 Excel 表格（缺失字段自动标“待确认”）。
3. **用 AI 定时任务自动处理日常办公事项（到点自动干活）**
   - 遵循「先在对话框手动跑通 Prompt ➡️ 绑定独立沙盒工作目录 ➡️ 配置定时周期（RRULE / 每日每周规则）并激活」标准流程，实现每日早报汇总、定时表单统计与周五周报自动生成。

---

## 三、完整课程章节内容大纲（第 0 章 ～ 第 7 章 + 生态进阶导流）

### 📖 第 0 章｜导读：把你的 AMD 电脑变成学习和工作的 AI 助手
- **0.1 为什么从本地 AI 开始？**：零 Token 费用焦虑、办公文件与个人隐私 100% 不出本机、亲手定制专属助手。
- **0.2 学习者成功故事全景**：从零基础小白到打造全天候本地 AI 办公与学习助手的成长路径。
- **0.3 3 分钟硬件对号入座与云算力领用**：速查显卡与显存，对照「绿/蓝/紫/云」四档标签选定配置。
- **0.4 全局总览地图与下一站深造导航**：看懂 `hello-rocm` 如何作为总览第一站串联 [`self-llm`](https://github.com/datawhalechina/self-llm)、[WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)、[`all-in-rag`](https://github.com/datawhalechina/all-in-rag)、[`hello-agents`](https://github.com/datawhalechina/hello-agents) 与 [`hello-gpu`](https://github.com/datawhalechina/hello-gpu)，并安装伴学助教 `hello-rocm Skill`。

---

### 🚀 第 1 章｜第一站：直观认识数据类型与三大引擎最低成本部署
> **本章目标**：以直观认识搞懂参数量、文件大小、运行内存与精度的关系，并用 **三大标准服务引擎（`llama-server`、`lemonade-server`、`vLLM`）** 跑起本地模型服务。

- **1.1 【直觉必修课】数据类型与模型资源的“三本明白账”**
  - **第一本账：参数量（0.6B / 4B / 8B / 32B）是什么？** —— 直观认识模型“脑容量”与不同日常任务的匹配关系。
  - **第二本账：参数量、权重文件大小与运行内存/显存有什么区别？** —— 为什么 4.9GB 的模型权重文件跑起来会占 8GB 显存？图解 **「字典净重（权重文件）+ 答题草稿纸（随上下文增长的 KV Cache）+ 桌面运转空间（运行时开销）」**。
  - **第三本账：为什么同一个模型有不同精度版本（BF16 / FP16 / FP8 / Q8_0 / Q4_K_M），以及如何选择？** —— 用「无损原片 vs 高清压缩图 vs 缩略图」建立数据类型直觉，附 **《AMD 显存档位 × 精度版本选型速查表》**。
- **1.2 极简环境准备（Windows 11 & Ubuntu 双平台基线）**
  - 显卡驱动核对、`uv` Python 环境准备与免费云算力连通。
- **1.3 轻量极速引擎：用 `llama-server`（`llama.cpp`）拉起本地 GGUF 服务**
  - 免复杂编译直接启动 ROCm 版 `llama-server`（默认暴露 OpenAI 兼容接口 `http://127.0.0.1:8080/v1`）。
- **1.4 端侧一体与异构引擎：用 `lemonade-server` 拉起本地 GPU / NPU 服务**
  - 一条命令启动 AMD 生态原生 `lemonade-server`（默认暴露 `http://127.0.0.1:13305/v1` 并自带管理界面，支持 Radeon GPU 与 Ryzen AI NPU）。
- **1.5 高吞吐并发引擎：用 `vLLM` 拉起标准推理服务**
  - 启动 ROCm 版 `vLLM` 服务（`http://127.0.0.1:8000/v1`），一张表看清三大引擎（`llama-server` / `lemonade-server` / `vLLM`）各自的最佳适用场景。
- 🎯 **本章实战产出**：使用三大引擎任一拉起本地 `/v1` 模型服务，完成《本机显存水位与速度初测卡》。
- 🔗 **学完本章想玩转更多开源模型部署？进阶导流** ➡️ **[Datawhale `self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)**（涵盖百余款国内外主流开源大模型的完整部署方案与参数配置库）。

---

### 🔌 第 2 章｜即刻用起来：零代码办公先行 —— 接入 WorkBuddy、Qoder 与 Codex
> **本章目标（首发核心章节）**：发布经过真机验证的版本号与配置文件，将本地模型服务无缝接入 **WorkBuddy、Qoder 与 Codex**，并联动 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)** 打通日常办公自动化！

- **2.1 读懂“万能插座”（OpenAI 兼容接口）与本地 Agent 三大安全铁律**
  - 白话讲清 `Base URL`、`API Key`、`Model ID`、`supportsToolCall`；牢记**独立英文沙盒文件夹、非 C 盘缓存、批量操作必加 `dry-run` 预览确认**三大铁律。
- **2.2 🌟【第一步重点推出】接入 WorkBuddy：AI PPT 生成、AI 自动填表与定时办公自动化**
  - **发布官方已验证 `models.json` 与避坑清单**：对接 `llama-server` / `vLLM` / `lemonade-server`，讲清 UI `/v1` 与 `models.json` `/v1/chat/completions` 差异、非空 `apiKey` 校验、以及必开 `32768` 上下文防止内置工具说明被截断导致“回复空白”。
  - **实战一 · AI PPT 生成（唯一采用 `dashi-ppt-skill` 大师 PPT Skill）**：安装 `dashi-ppt-skill`，根据本地文档一键生成浏览器可微调、导出可在 WPS/PowerPoint 二次编辑的 `.pptx` 演示文稿。
  - **实战二 · AI 自动填表**：参考并链接 [Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)，批量解析本地单据/简历文件夹，按严格字段契约自动填入汇总 Excel 表格。
  - **实战三 · 用 AI 定时任务自动处理日常办公事项**：参考 [Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) 配置 WorkBuddy 定时任务（RRULE 每日/每周规则），到点自动读取目录数据、填表统计并生成日常汇报。
- **2.3 接入 `Qoder`：打造全离线零成本 AI 编程 IDE**
  - 在 `Qoder`（设置 → 模型 → 添加自定义模型 `OpenAI Compatible` / `Chat Completions API`）中接入本地 `vLLM` / `llama-server` / `lemonade-server`，配置上下文窗口并通过一键校验，体验本地代码补全、对话重构与 `.agents/skills/` 技能加载。
- **2.4 接入 `Codex CLI` 与终端轻量智能体（含仓库 `toy-cli`）**
  - 配置 `~/.codex/config.toml` 直连本地服务，在命令行自动查报错、写脚本、跑测试。
- 🎯 **本章实战产出**：零 API 费用跑通 WorkBuddy「AI PPT + 自动填表 + 定时办公任务」三件套，并完成 `Qoder` / `Codex` 本地编程联调。
- 🔗 **学完本章想深入解锁全套 AI 办公进阶玩法？进阶导流** ➡️ **[Datawhale WorkBuddy 完整教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**。

---

### 🧠 第 3 章｜玩转 Agentic AI：提示词闯关、个人知识库、知识管理 MCP 与 Skills
> **本章目标**：掌握提示词、RAG、MCP 与 Skills 四大核心能力，亲手**把个人知识库封装成标准 MCP 服务供他人调用**，并衔接 Datawhale RAG 与智能体进阶课程！

- **3.1 提示词闯关课：6 关本地交互式 Notebook 驯服中小模型**
  - 借鉴 Anthropic 交互式提示词教程与 Datawhale `smart-prompt`，将 Jupyter 闯关场直连本机 API，边改边通关（角色边界、分隔符隔离、Few-Shot 示例、JSON/表格强约束、防幻觉认怂）。
- **3.2 个人知识库实战（RAG）：从「黑暗料理菜谱切分实验」到 `WorkBuddy + Obsidian` 私人笔记库**
  - 联动 [Datawhale `all-in-rag`](https://github.com/datawhalechina/all-in-rag)「今天吃什么」菜谱实验，直观对比“硬切 150 字教出黑暗料理 vs 按标题结构切分变大厨”；并在 **`WorkBuddy + Obsidian`**（配合官方 `obsidian-skills`）中零代码搭好带 `[[双向链接]]` 与出处角标的私人课件问答库。
- **3.3 工具与万能插座 MCP：体验趣味 MCP + 手把手将「个人知识库」封装成可共享的 `知识管理 MCP`**
  - 联动 [Datawhale `hello-agents`](https://github.com/datawhalechina/hello-agents) 看清 `观察→思考→行动` 循环，先体验 B站字幕/高德地图等趣味 MCP；
  - **核心实战**：使用 `FastMCP` (`@mcp.tool()`) 把 3.2 节建好的个人知识库封装为标准的 **`知识管理 MCP Server`**（提供知识检索、新增笔记卡片、按标签盘点等标准工具接口），不仅能在自己的 WorkBuddy、`Qoder`、Codex 中跨软件随调随用，还能直接打包分享给队友调用！
- **3.4 模块化技能包 Agent Skills（`SKILL.md`）：渐进式省显存秘籍与自定义办公技能**
  - 讲透为什么 `SKILL.md` 平时只加载两行描述（拯救本地小模型宝贵显存）；解剖本仓库 `hello-rocm-skill` 与 `dashi-ppt-skill` 结构，亲手为 WorkBuddy / `Qoder` / Codex 写出专属《日常办公检查/周报排版 `SKILL.md`》。
- **3.5 安全护栏与进阶痛点复盘（自然引出第 4、5 章）**
- 🎯 **本章实战产出**：**「个人学习助手 v2.0」（集成私人笔记 RAG + 知识管理 MCP + 自定义办公 `SKILL.md`）**。
- 🔗 **学完本章想系统深造 RAG 与 Agent 底层架构？双通道进阶导流** ➡️
  - **知识库深造**：**[Datawhale `all-in-rag`（RAG 技术全栈指南）](https://github.com/datawhalechina/all-in-rag)**（进阶掌握混合检索、重排 Rerank、多模态解析与 GraphRAG）；
  - **智能体深造**：**[Datawhale `hello-agents`（从零开始构建智能体）](https://github.com/datawhalechina/hello-agents)**（从零手写智能体核心框架、记忆系统与多 Agent 协作）。

---

### 🔧 第 4 章｜能力不够怎么办？动手做第一次中文模型微调（魔搭纯中文生态）
> **本章目标**：基于 ModelScope 魔搭社区中文数据集，提供「`LLaMA-Factory WebUI` 零代码点鼠标 + 60 行透明代码」双通道，完成办公指令微调与个性化网感分身微调！

- **4.1 微调决策树：什么时候改提示词、加知识库，什么时候必须微调？**
- **4.2 直觉算清 LoRA 显存账 + 5 分钟「弱智吧（`COIG-CQIA`）」中文逻辑破冰实验**
  - 拉取中科院开源 `m-a-p/COIG-CQIA`（弱智吧中文子集，240–500 条），在 `LLaMA-Factory WebUI` + `SwanLab` 看板下，8G 显卡 **5 分钟** 见证小模型识破中文双关逻辑陷阱！
- **4.3 中文数据清洗与脱敏构造实战**
- **4.4 主线实战：基于 `MSAgent-Bench` 与 `InstructIE` 微调中文通知抽取与工具调用**
  - 让本地小模型从口语化中文办公/课程群通知中精准提取结构化待办与填表 JSON，解决小模型调工具格式不稳痛点。
- **4.5 自建中文测试集评估微调前后效果 & 导出上线**
  - 用 20 行评测脚本跑 50 条独立测试题，定量对比微调前后准确率，合并导出权重重新接入 `llama-server` / `vLLM` / `lemonade-server`。
- **4.6 趣味进阶实战（网感与分身）：`WeClone` 微信聊天记录数字分身 / 小红书爆款种草体**
  - 使用个人脱敏微信聊天记录（参考 `WeClone`）或魔搭小红书高赞文案集进行风格 LoRA 微调，彻底消除机械“AI 客服腔”，通过破功词下降率与双盲测试验证效果。
- 🎯 **本章实战产出**：微调权重 + SwanLab 训练记录 + 《自建中文测试集微调前后效果评估报告》。
- 🔗 **学完本章想挑战更多模型家族与高阶微调范式？进阶导流** ➡️ **[Datawhale `self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)**（深入解锁 Qwen / DeepSeek / GLM / Gemma 等全系模型微调与部署案例）。

---

### ⚡ 第 5 章｜速度不够怎么办？AMD 量化对比、零 C++ 算子加速与 `Hyperloom` 自动优化
> **本章目标**：全程零 C++ 代码！直观对比 `AMD Quark` 量化前后差异，一键唤醒 AMD 融合算子，体验用 `AMD-AGI/Hyperloom` 全自动寻优，并为渴望深入底层的同学打通前往 `hello-gpu` 的桥梁！

- **5.1 给慢助手做 AI 体检：看懂 TTFT / TPS 指标 + 用 `AMD-AGI/TraceLens` 自动定位瓶颈**
- **5.2 AMD 官方量化直观实战：用 `AMD Quark` 跑通同一模型量化前后对比**
  - 选定一个基准开源模型，使用 AMD 官方 `AMD Quark` (`pip install amd-quark`) 跑通量化全流程，**直观实测并记录量化前 vs 量化后的文件大小（GB）、显存占用、生成速度（tok/s）与回答效果对比表**。
- **5.3 零 C++ 算子加速：`@torch.compile` 自动融合 + 一键唤醒 `hipBLASLt` / `rocWMMA FA` / `AITER`**
  - 15 行纯 Python 加 `@torch.compile` 体验 ROCm Triton 自动算子融合；在 `llama-server` 开启 `ROCBLAS_USE_HIPBLASLT=1 -fa on`，在 `vLLM` 开启 **`VLLM_ROCM_USE_AITER=1`**（启用 `ROCm/aiter` 高性能融合算子库），实测长文首字与吞吐翻倍！
- **5.4 让 WorkBuddy、`Qoder` 与 Codex 秒回：Prefix Caching 前缀缓存与投机采样**
  - 开启前缀缓存化解 Agent 数万字系统提示词重复计算（首字等待从 10s 压至 <1s），配合小模型投机采样与 `lemonade-server` NPU+GPU 分流。
- **5.5 【Agentic 自动寻优压轴】结合 [`AMD-AGI/Hyperloom`](https://github.com/AMD-AGI/Hyperloom) 与 `AMD Skills` 实现全自动优化**
  - 在 `Qoder` / Codex 中通过 `@hyperloom/inference_optimizer/SKILL.md` 一句话驱动 `Magpie → TraceLens → Arbor + GEAK` 全自动调参与重写热点算子闭环。
- 🎯 **本章实战产出**：《模型量化前后与算子加速实测对比表（体积 / 显存 / 速度 / 效果）》。
- 🔗 **学完本章对底层 GPU 硬件原理与手写高性能算子产生浓厚兴趣？进阶导流** ➡️ **[Datawhale `hello-gpu`（GPU 架构与算子优化实战）](https://github.com/datawhalechina/hello-gpu)**（系统学习 AMD GPU 硬件架构、ROCm 工具链、手写 HIP / Triton 算子优化与单卡深度性能分析）。

---

### 🏆 第 6 章｜收获时刻：成功故事、简历成果锻造与下一站课程深造导航
- **6.1 撰写你的成功故事**：「我把自己的 AMD 电脑，变成了学习和工作的 AI 助手」成长纪实模板。
- **6.2 个人 AI 作品集五件套打包清单**：应用代码 + 演示实录 + 训练/微调记录 + 量化与测试集效果评估报告 + 个人贡献说明。
- **6.3 简历项目打磨与面试追问指南**：用实测数据证明**模型部署、数据处理、应用集成与评测能力**。
- **6.4 🎓 毕业导流驿站：从 `hello-rocm` 出发，按职业与兴趣选修你的下一门专精课**
  - **想走大模型算法与微调部署工程师方向** ➡️ 直通 **[`self-llm`](https://github.com/datawhalechina/self-llm)**
  - **想走 AI 超级个体与数字化办公专家方向** ➡️ 直通 **[Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**
  - **想走企业级知识库与 AI Agent 应用架构师方向** ➡️ 直通 **[`all-in-rag`](https://github.com/datawhalechina/all-in-rag)** + **[`hello-agents`](https://github.com/datawhalechina/hello-agents)**
  - **想走高性能计算（HPC）与 AI Infra 推理算子工程师方向** ➡️ 直通 **[`hello-gpu`](https://github.com/datawhalechina/hello-gpu)**

---

## 四、第 7 章：15 个趣味实战案例库（首发办公置顶 · 绑定进阶课程导流）

> 所有案例底层统一仅使用 **`llama-server`、`vLLM` 或 `lemonade-server`** 提供模型服务；每个案例强制包含 **「🎬 演示 + ✅ 支持配置 + 💻 代码与步骤 + 🔍 验证方法 + 🚀 扩展任务（含对应专精课程导流链接）」** 五要素，并执行**新手试学机制**持续完善常见问题（FAQ）。

| 编号 | 所属场景 | 案例名称 | 核心玩法与交付成果（及进阶导流衔接） | 最低硬件档位 |
|:---:|:---|:---|:---|:---:|
| **Case 01** | 📊 报告生成 | **🌟【首发王牌】一句话文档变可编辑汇报 PPT（WorkBuddy + `dashi-ppt-skill`）** | 拖入总结文档套用 1020 套版式并浏览器微调，导出 WPS/Office 原生可编辑 `.pptx`（🔗 进阶：[WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)） | 🟢 8G / 🔵 16G |
| **Case 02** | 🗂️ 文件整理 | **【办公必装】AI 自动填表与杂乱文件夹智能归档** | 批量解析报销单/简历 PDF 自动填入标准 Excel 表，并分类归档杂乱目录（🔗 进阶：[WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)） | 🟢 8G |
| **Case 03** | 🗺️ 任务规划 | **【到点自动干活】用 WorkBuddy AI 定时任务自动处理日常办公事项** | 配置 RRULE 每日/每周定时任务，自动完成统计填表与日报周报生成（🔗 进阶：[WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)） | 🟢 8G |
| **Case 04** | 📊 报告生成 | **全本地万字深度调研报告专家（`Local Deep Research`）** | 给一个课题自动多轮检索网页、论文与本地资料，输出带引用角标研报（🔗 进阶：[`all-in-rag`](https://github.com/datawhalechina/all-in-rag) / [`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🔵 16G |
| **Case 05** | 📄 论文阅读 | **多篇英文文献横评对比表 & 组会汇报 PPT 一键生成（WorkBuddy + `dashi-ppt-skill`）** | `@` 多篇论文 PDF 先出方法局限性横评表，再直出组会 `.pptx`（🔗 进阶：[WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) / [`all-in-rag`](https://github.com/datawhalechina/all-in-rag)） | 🔵 16G |
| **Case 06** | 📄 论文阅读 | **公式图表纹丝不乱：双语学术论文精翻神器（`PDFMathTranslate` + `Zotero`）** | 调用本地 `vLLM` 并发服务整本精翻英文 arXiv 论文 PDF，公式排版零损保留（🔗 进阶：[`self-llm`](https://github.com/datawhalechina/self-llm)） | 🟢 8G |
| **Case 07** | 📚 个人知识库 | **`WorkBuddy + Obsidian` 双链第二大脑 & 可共享「知识管理 MCP」** | 本地笔记双链问答，并用 `FastMCP` 将知识库打包成 MCP 给别人调用（🔗 进阶：[`all-in-rag`](https://github.com/datawhalechina/all-in-rag) + [`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🟢 8G |
| **Case 08** | 📚 个人知识库 | **复习讲义秒变双人播客 & 微信/小红书网感分身（`Open Notebook` + `WeClone`）** | PDF 讲义一键转双人播客 MP3 路上听，并用聊天记录微调专属网感分身（🔗 进阶：[`self-llm`](https://github.com/datawhalechina/self-llm)） | 🟢 8G / 🔵 16G |
| **Case 09** | 🎙️ 语音交互 | **全离线保密会议与网课纪要神器（`Meetily` + `Faster-Whisper`）** | 录音不上云，实时离线转写、区分发言人并提炼待办事项清单（🔗 进阶：[`self-llm`](https://github.com/datawhalechina/self-llm)） | 🟢 8G |
| **Case 10** | 🎙️ 语音交互 | **桌面 Live2D 实时语音伴侣与英语口语陪练（`Open-LLM-VTuber` / 仓库 `MiniCPM-o`）** | 与桌面有表情、可随时打断的本地虚拟伙伴全离线实时练口语（🔗 进阶：[`self-llm`](https://github.com/datawhalechina/self-llm)） | 🟢 8G / 🔵 16G |
| **Case 11** | 💻 编程助手 | **`Qoder` 全离线 AI 编程搭档实战** | 将本地三大引擎接入 `Qoder` IDE，零成本实现代码补全、重构与单测生成（🔗 进阶：[`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🟢 8G / 🔵 16G |
| **Case 12** | 💻 编程助手 | **终端自主修 Bug 智能体与百行源码解剖（`OpenAI Codex CLI` & 仓库 `toy-cli`）** | 终端自动读报错改代码跑测试，30 分钟读懂极简 Coding Agent 循环（🔗 进阶：[`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🟢 8G / 🔵 16G |
| **Case 13** | 🗺️ 任务规划 | **网页自动化操控与智能旅行规划师（`Browser-Use` + 仓库高德 MCP `smart-travel-planner`）** | 本地 AI 操控浏览器比价查资料，结合 `HelloAgents` 调高德 MCP 排日程（🔗 进阶：[`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🔵 16G |
| **Case 14** | 🗺️ 任务规划 | **24 小时全隐私多端个人管家中枢（仓库 `OpenClaw` + Prefix Caching 加速）** | 手机飞书发消息，家里 AMD 电脑自动拆解复杂任务并利用前缀缓存秒回（🔗 进阶：[`hello-agents`](https://github.com/datawhalechina/hello-agents) / [`hello-gpu`](https://github.com/datawhalechina/hello-gpu)） | 🟣 旗舰 / ☁️ 云 |
| **Case 15** | 🎮 游戏互动 | **从扫雷挑战赛、《我的世界》AI 搭档到文字跑团剧场（仓库 `Minesweeper Agent` & `Mindcraft`）** | 在扫雷看清 Agent 推理闭环，再放进 Minecraft 自动生存或驱动跑团剧场（🔗 进阶：[`hello-agents`](https://github.com/datawhalechina/hello-agents) / [`all-in-rag`](https://github.com/datawhalechina/all-in-rag)） | 🟢 8G / 🔵 16G |

---

## 五、附录与进阶选修导流区（无损保护硬核资产 · 直连专精课程）

- **附录 A**：已验证软硬件、三大服务引擎（`llama-server` / `lemonade-server` / `vLLM`）与客户端（`WorkBuddy` / `Qoder` / `Codex`）配置文件速查总库
- **附录 B**：新手试学全局排错急救手册
- **进阶选修与外部课程直通专区**：
  - **多模型部署与微调百科**（原 `01-deploy` 多模型矩阵）：无损保留路由，并统一链接至 **[`self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)**；
  - **AMD GPU 架构与底层算子编程**（原 `03-infra` 硬件解密、HIP 算子、自定义 PyTorch 算子）：无损保留路由，并无缝导流至 **[`hello-gpu`（GPU 架构与算子优化实战）](https://github.com/datawhalechina/hello-gpu)**；
  - **RAG 与 Agent 全栈开发**（原 `hello-agents` 旅行助手等）：统一导流至 **[`all-in-rag`](https://github.com/datawhalechina/all-in-rag)** 与 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)**；
  - **科研与行业拓展归档**：`Every Embodied`、`Torch-RecHub`、`Happy-LLM`、`Strix Halo 原生部署` 保持旧 URL 路由兼容零 404。
