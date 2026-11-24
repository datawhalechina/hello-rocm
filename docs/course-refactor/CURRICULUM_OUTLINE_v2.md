# hello-rocm 2.0 完整课程总大纲（定稿版）

---

## 一、课程核心定位与成果闭环

- **课程名称**：`hello-rocm`
- **核心标语（Slogan）**：**每个 AI 小白，从零开始，动手实践的第一站。**
- **课程副标题（定位）**：**基于 AMD 平台的 AI 与 Agentic AI 入门实践手册（全栈总览先导课）。**
- **核心生态角色**：`hello-rocm` 是面向零基础学习者的 **「AI 全景总览入口课」** —— 先在一台 AMD 电脑（或免费云算力）上低成本跑通「本地部署 → AI 办公 → 知识库 → 智能体与 MCP → 中文微调 → 量化与算子加速 → 趣味案例与简历沉淀」的完整体验闭环，再按兴趣精准导流至 Datawhale 各垂直领域专精课程继续深造。
- **课程主体模型统一标准（锁定三大高性价比黄金主力）**：
  - **`Gemma 4 E4B`（`google/gemma-4-E4B-it`）**：8G 端侧全模态（文本/图像/音频 + 原生工具调用）轻量王牌；
  - **`Qwen3.5-9B`（`Qwen/Qwen3.5-9B`）**：中文语义理解、办公填表与纯中文 LoRA 微调 / `AMD Quark` 量化对比主力；
  - **`Gemma 4 12B`（`google/gemma-4-12B-it`）**：强逻辑推理、`Qoder` / `Codex` 跨文件编程与复杂多步 Agent 进阶主力。
- **底层推理服务统一标准（全书收敛为三大标准引擎，拒绝选择困难）**：
  - 全书统一仅使用 **`llama-server`（`llama.cpp`）**、**`vLLM`** 与 **`lemonade-server`** 拉起本地 OpenAI 兼容服务（全面覆盖 AMD Radeon 独显、Ryzen AI 核显/NPU 与云算力）。
- **贯穿全书的成功故事**  
  > **“我把自己的 AMD 电脑，变成了学习和工作的 AI 助手。”**
- **作品集与简历成果转化**
  - **沉淀五件套**：应用代码、交互演示、训练或微调记录（SwanLab）、效果评估报告（自建测试集前后对比 + 量化前后测速对比）、个人贡献说明。
  - **证明四大能力**：模型部署、数据处理、应用集成、科学评测。
  - **标准简历项目示例**：
    > *基于 AMD 平台完成小模型微调并构建本地学习助手，负责数据整理、模型服务与知识检索集成，通过自建测试集评估微调前后及应用效果。*
- **四档零焦虑硬件适配（Windows 11 & Ubuntu 本地优先 + 免费云算力兜底）**
  - 🟢 **入门档（8G–12G 独显 / 16G–32G 核显本）**：主跑 **`Gemma 4 E4B` (`Q4_K_M` / FLM NPU)**
  - 🔵 **甜点档（16G–24G 独显，如 RX 7800 XT / 7900 / 9070 系列）**：主跑 **`Qwen3.5-9B`** 与 **`Gemma 4 12B` (`Q4_K_M` / `Q8_0`)**
  - 🟣 **旗舰档（64G–128G 统一内存 AI PC，如 Ryzen AI Max+ 395）**：主跑 **`Gemma 4 12B` / `Qwen3.5-9B` (`BF16`/`Q8_0`)** 及更大 MoE 超长上下文
  - ☁️ **云端兜底档（手头暂无 AMD 显卡）**：免费接入 **AMD Radeon Cloud** 全程实操

---

## 二、🧭 用户学习体验旅程（User Learning Journey）

课程彻底摒弃“先啃枯燥底层概念”的传统路径，按照零基础学习者的**真实使用痛点与心理获得感曲线**设计为 **6 站递进式体验旅程**：

| 旅程站点 | 对应章节 | 学习者真实心声 / 触发痛点 | 核心动手体验 | 即时获得感与阶段交付物 |
|:---:|:---:|:---|:---|:---|
| **第 1 站<br/>破冰启程** | **第 0～1 章** | “我不懂底层编译，不知道自己显卡能跑多大模型，怕把环境搞崩。” | 3 分钟硬件对号入座，直观算清参数/显存/精度“三本账”，用 `llama-server` / `lemonade-server` / `vLLM` 拉起本地服务。 | 🎉 **掌控感**：15 分钟跑通本地 `/v1` 接口，产出《本机显存水位与速度初测卡》。 |
| **第 2 站<br/>办公提效** | **第 2 章**<br/>*(首发核心)* | “命令行能聊天了，但怎么帮我做 PPT、填 Excel、写代码？” | 将本地模型接入 **WorkBuddy**（跑通 `dashi-ppt-skill` 出 PPT、自动填表、定时任务）及 **`Qoder` / `Codex`**。 | 🚀 **惊艳感**：零 Token 费用自动交付可编辑 `.pptx` 与汇总报表，助手当天上岗！ |
| **第 3 站<br/>注入记忆** | **第 3 章** | “AI 不懂我的专业课件和私密资料，还总是一本正经瞎编！” | 闯关 6 节本地提示词课，做完「黑暗料理菜谱切分实验」，搭好 **`WorkBuddy + Obsidian`** 双链私人笔记库。 | 📚 **信赖感**：AI 能严格基于本地笔记带 `[[双链]]` 与出处角标精准作答。 |
| **第 4 站<br/>长出手脚** | **第 4 章** | “我的个人知识库能不能在 Qoder、Codex 里用，甚至分享给队友调用？” | 用 `FastMCP` 把个人知识库封装为标准 **`知识管理 MCP`**，并编写省显存的自定义办公 `SKILL.md`。 | 🛠️ **创造感**：建成 **「个人学习助手 v2.0」**，实现跨软件调用与打包分享！ |
| **第 5 站<br/>痛点攻坚** | **第 5～6 章** | “小模型抽中文通知格式偶发写歪、说话太机械；而且上下文长了首字变慢！” | **能力不够（第 5 章）**：魔搭中文集 LoRA 微调 + `WeClone`/小红书分身；<br/>**速度不够（第 6 章）**：`AMD Quark` 单模型量化前后对比 + 零 C++ 唤醒融合算子 + `Hyperloom` 自动寻优。 | ⚡ **工程突破感**：手握《中文测试集微调前后评估报告》与《量化加速前后实测对比表》。 |
| **第 6 站<br/>收获远航** | **第 7～9 章** | “怎么把这些成果变成简历竞争力？接下来往哪个细分方向深造？” | 打包作品集五件套与简历经历（第 7 章），畅玩 15 个趣味案例（第 8 章），探索 AMD 开源宝藏并进入专精课程（第 9 章）。 | 🏆 **成就感**：沉淀完整 GitHub 作品包与简历项目，开启下一门专精课程深造！ |

---

## 三、🔗 本课程与其他课程的引用与进阶导流关系

作为 **全景总览先导课**，`hello-rocm` 与 Datawhale 社区各垂直领域王牌课程形成严密的 **「上游精华引用（Inbound）+ 下游专精导流（Outbound）」一章一课精准映射关系**：

| 本课程章节 | 关联外部专精课程 | 上游内容引用与联动方式（在本课学什么） | 下游学完进阶导流方向（去专精课深造什么） |
|:---|:---|:---|:---|
| **第 1 章｜本地模型部署** | 🔗 **[`self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)** | 对齐开源大模型本地环境基线与标准 API 服务调通规范 | 解锁百余款国内外主流开源大模型的部署细节、多框架适配与参数全集 |
| **第 2 章｜AI 办公与工具接入** | 🔗 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)** | 直接参考并引用其 AI PPT 生成、AI 自动填表与定时办公任务实操范式 | 系统深造桌面智能体复杂办公流、多源数据联动与企业级办公自动化 |
| **第 3 章｜提示词与个人知识库** | 🔗 **[`all-in-rag`（RAG 技术全栈指南）](https://github.com/datawhalechina/all-in-rag)** | 引用其经典「今天吃什么」黑暗料理菜谱分块（Chunking）对比实验 | 进阶掌握生产级文档解析、混合检索、重排（Rerank）与知识图谱 GraphRAG |
| **第 4 章｜MCP、Skills 与智能体** | 🔗 **[`hello-agents`（从零开始构建智能体）](https://github.com/datawhalechina/hello-agents)** | 引用其 `观察→思考→行动` 核心循环理念及高德 MCP 智能旅行规划案例 | 穿透现成工具表象，从零手写 ReAct / Plan-and-Solve 框架、记忆系统与多 Agent 协同 |
| **第 5 章｜中文模型微调实战** | 🔗 **[`happy-llm`（从零开始的大语言模型原理与实践）](https://github.com/datawhalechina/happy-llm)** | 衔接大模型训练与指令微调（SFT / LoRA）核心脉络（第 9 章亦收录其多卡实战） | 深入理解 Transformer 底层手搓实现、预训练、微调对齐原理及从零训练专属大模型 |
| **第 6 章｜AMD 量化与算子加速** | 🔗 **[`hello-gpu`（GPU 架构与算子优化实战）](https://github.com/datawhalechina/hello-gpu)** | 建立 TTFT/TPS 性能体检与 AMD 硬件加速算子（`hipBLASLt`/`AITER`）直觉认知 | 从“零 C++ 开关加速”迈入硬核底层：AMD GPU 架构、ROCm 工具链与手写 HIP / Triton 算子优化 |
| **第 9 章｜AMD 优秀项目与行业案例** | 🔗 **[`every-embodied`](https://github.com/datawhalechina/every-embodied) / [`torch-rechub`](https://github.com/datawhalechina/torch-rechub)** | 收录本仓库基于 AMD ROCm 跑通的具身智能策略复刻与推荐系统工程案例 | 面向机器人具身智能（VLA/ACT）与工业级推荐系统方向展开专题研究 |

---

## 四、🚀 第一步重点推出内容：本地模型 + WorkBuddy 办公自动化三件套

打破“ROCm 教程太技术、不好学”的印象，**第一步重点首发内容**联动 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**，用本地免费 AMD 算力跑通人人每天刚需的三大办公场景：

1. **AI PPT 一键生成（唯一锁定开源王牌 `dashi-ppt-skill` 大师 PPT Skill）**
   - **开源仓库**：`github.com/chuspeeism/dashi-ppt-skill`（一行 `npx dashi-ppt-skill@latest` 装入 WorkBuddy）。
   - **核心亮点**：内置 12 套视觉主题 + 1,020 个商业/学术版式，自带浏览器局部微调台，**支持一键导出在 PowerPoint / WPS 中真实可二次编辑的 `.pptx` 文件**。
2. **AI 自动填表（非结构化单据/资料 ➡️ 标准 Excel 报表）**
   - 扫描本地文件夹中的多份报销单、简历或原始明细，严格按字段契约提取关键信息并自动追加填入目标 Excel 表格（缺失字段自动标“待确认”）。
3. **用 AI 定时任务自动处理日常办公事项（到点自动干活）**
   - 遵循「先在对话框手动跑通 Prompt ➡️ 绑定独立沙盒工作目录 ➡️ 配置定时周期（RRULE 每日/每周规则）并激活」标准流程，实现定时表单统计与日报周报自动生成。

---

## 五、📚 完整课程章节大纲（第 0 章 ～ 第 9 章）

### 📖 第 0 章｜导读：把你的 AMD 电脑变成学习和工作的 AI 助手
- **0.1 为什么从本地 AI 开始？**：零 Token 费用焦虑、办公文件与个人隐私 100% 不出本机、亲手定制专属助手。
- **0.2 学习者成功故事与 6 站体验旅程全景**：从零基础小白到打造全天候本地 AI 办公与学习助手的成长路径。
- **0.3 3 分钟硬件对号入座与免费云算力领用**：速查显卡与显存，对照「绿/蓝/紫/云」四档标签选定配置。
- **0.4 全栈总览地图、课程引用导航与内置伴学助教（`hello-rocm Skill`）**

---

### 🚀 第 1 章｜第一站：直观认识数据类型与三大引擎最低成本部署
> **本章目标**：直观搞懂参数量、文件大小、运行内存与精度的关系，并用 **三大标准服务引擎（`llama-server`、`lemonade-server`、`vLLM`）** 跑起本地模型服务。

- **1.1 【直觉必修课】数据类型与模型资源的“三本明白账”**
  - **第一本账：参数量（0.6B / 4B / 8B / 32B）是什么？** —— 模型“脑容量”与日常任务的匹配关系。
  - **第二本账：参数量、权重文件大小与运行内存/显存有什么区别？** —— 图解 **「字典净重（权重文件）+ 答题草稿纸（随上下文增长的 KV Cache）+ 桌面运转空间（运行时开销）」**。
  - **第三本账：为什么同一个模型有不同精度版本（BF16 / FP16 / FP8 / Q8_0 / Q4_K_M），以及如何选择？** —— 建立数据类型直觉，附 **《AMD 显存档位 × 精度版本选型速查表》**。
- **1.2 极简环境准备（Windows 11 & Ubuntu 双平台基线）**
- **1.3 轻量极速引擎：用 `llama-server`（`llama.cpp`）拉起本地 GGUF 服务**（`http://127.0.0.1:8080/v1`）
- **1.4 端侧一体与异构引擎：用 `lemonade-server` 拉起本地 GPU / NPU 服务**（`http://127.0.0.1:13305/v1`）
- **1.5 高吞吐并发引擎：用 `vLLM` 拉起标准推理服务**（`http://127.0.0.1:8000/v1`）及三大引擎选型对比表
- 🎯 **本章实战产出**：使用三大引擎任一拉起本地 `/v1` 模型服务，完成《本机显存水位与速度初测卡》。
- 🔗 **课程引用与进阶导流** ➡️ **[`self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)**（解锁百余款主流开源大模型的部署方案）。

---

### 🔌 第 2 章｜即刻用起来：零代码办公先行 —— 接入 WorkBuddy、Qoder 与 Codex
> **本章目标（首发核心章节）**：发布经过真机验证的版本号与配置文件，将本地模型服务无缝接入 **WorkBuddy、Qoder 与 Codex**！

- **2.1 读懂“万能插座”（OpenAI 兼容接口）与本地 Agent 三大安全铁律**（独立英文沙盒文件夹、非 C 盘缓存、批量操作 `dry-run` 预览确认）
- **2.2 🌟【第一步重点推出】接入 WorkBuddy：AI PPT 生成、AI 自动填表与定时办公自动化**
  - **发布官方已验证 `models.json` 与避坑清单**：对接 `llama-server` / `vLLM` / `lemonade-server`，讲清 UI `/v1` 与 `models.json` `/v1/chat/completions` 差异、非空 `apiKey` 校验及必开 `32768` 上下文防截断。
  - **实战一 · AI PPT 生成（唯一采用 `dashi-ppt-skill` 大师 PPT Skill）**：一键生成浏览器可微调、WPS/PowerPoint 原生可二次编辑的 `.pptx` 演示文稿。
  - **实战二 · AI 自动填表**：参考 [Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)，批量解析本地单据/简历文件夹自动填入汇总 Excel 表格。
  - **实战三 · 用 AI 定时任务自动处理日常办公事项**：参考 [Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) 配置 RRULE 定时规则，到点自动统计填表与生成日常汇报。
- **2.3 接入 `Qoder`：打造全离线零成本 AI 编程 IDE**
  - 在 `Qoder`（设置 → 模型 → `OpenAI Compatible` / `Chat Completions API`）接入本地三大引擎，一键校验并体验代码补全、对话重构与 `.agents/skills/` 技能加载。
- **2.4 接入 `Codex CLI` 与终端轻量智能体（含仓库 `toy-cli`）**
- 🎯 **本章实战产出**：零 API 费用跑通 WorkBuddy「AI PPT + 自动填表 + 定时办公任务」三件套，完成 `Qoder` / `Codex` 本地编程联调。
- 🔗 **课程引用与进阶导流** ➡️ **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**（深入解锁更多桌面 AI 办公工作流）。

---

### 📚 第 3 章｜让助手读懂你：提示词闯关与个人知识库实战（Prompt + RAG）
> **本章目标**：驯服本地中小模型听懂结构化指令，并通过 `WorkBuddy + Obsidian` 零代码搭建言之有据的私人双链知识库！

- **3.1 提示词闯关课：6 关本地交互式 Notebook 驯服中小模型**
  - 直连本机 API 边改边通关：角色边界、分隔符隔离、Few-Shot 示例、JSON/表格强约束、防幻觉认怂。
- **3.2 RAG 核心直觉实验：「黑暗料理菜谱切分对比实验」**
  - 引用联动 [Datawhale `all-in-rag`](https://github.com/datawhalechina/all-in-rag)「今天吃什么」菜谱实验，直观对比“硬切 150 字教出黑暗料理 vs 按标题结构切分变大厨”，秒懂分块（Chunking）与向量检索本质。
- **3.3 私人第二大脑实战：用 `WorkBuddy + Obsidian` 搭建双链开卷笔记库**
  - 配合官方 `obsidian-skills`，在本地课件、论文与工作笔记上实现保留 `[[双向链接]]` 与精确出处角标的私密问答。
- **3.4 知识库防幻觉自测清单**：如何设计 10 道“开卷考题”检验检索命中率与拒答边界。
- 🎯 **本章实战产出**：「个人学习助手 v1.5（结构化提示词模板 + `WorkBuddy + Obsidian` 私人笔记问答库）」。
- 🔗 **课程引用与进阶导流** ➡️ **[`all-in-rag`（RAG 技术全栈指南）](https://github.com/datawhalechina/all-in-rag)**（进阶深造混合检索、重排 Rerank、多模态文档解析与 GraphRAG）。

---

### 🤖 第 4 章｜让助手长出手脚：知识管理 MCP、Agent Skills 与智能体实战
> **本章目标**：理解智能体工具调用闭环，**亲手把第 3 章的个人知识库制作成标准 MCP 服务供他人或跨工具调用**，并编写自定义办公 `SKILL.md`！

- **4.1 智能体核心循环（`观察 → 思考 → 行动`）与趣味 MCP 初体验**
  - 引用联动 [Datawhale `hello-agents`](https://github.com/datawhalechina/hello-agents) ReAct 原理，先一键挂载体验 B站字幕总结、高德地图周边查询等趣味 MCP。
- **4.2 🌟 核心实战：用 `FastMCP` 把「个人知识库」封装成可共享的 `知识管理 MCP Server`**
  - 将第 3 章的 Obsidian 个人知识库封装为标准 MCP 服务（暴露语义检索笔记、新增知识卡片、按标签盘点等 `@mcp.tool()` 接口），不仅能在自己的 WorkBuddy、`Qoder`、Codex 中跨软件随调随用，还能直接打包分享给别人调用！
- **4.3 模块化技能包 Agent Skills（`SKILL.md`）：渐进式省显存秘籍与自定义办公技能**
  - 讲透为什么 `SKILL.md` 平时只加载两行描述（拯救本地小模型宝贵显存）；解剖本仓库 `hello-rocm-skill` 与 `dashi-ppt-skill` 结构，亲手为 WorkBuddy / `Qoder` / Codex 写出专属《日常办公检查/周报排版 `SKILL.md`》。
- **4.4 本地 Agent 安全护栏与真实痛点复盘（自然引出第 5、6 章）**
  - 防死循环与高危操作确认锁；复盘实战暴露的两大真痛点：小模型中文复杂指令抽取不稳（→ 第 5 章微调）与长系统提示词首字慢（→ 第 6 章加速）。
- 🎯 **本章实战产出**：**「个人学习助手 v2.0」（集成私人笔记 RAG + 知识管理 MCP + 自定义办公 `SKILL.md`）**。
- 🔗 **课程引用与进阶导流** ➡️ **[`hello-agents`（从零开始构建智能体）](https://github.com/datawhalechina/hello-agents)**（深入手写智能体底层框架、上下文记忆工程与多智能体协同）。

---

### 🔧 第 5 章｜能力不够怎么办？动手做第一次中文模型微调（魔搭纯中文生态）
> **本章目标**：基于 ModelScope 魔搭社区中文数据集，提供「`LLaMA-Factory WebUI` 零代码 + 60 行透明代码」双通道，完成中文指令抽取与个性化网感分身微调！

- **5.1 微调决策树：什么时候改提示词、加知识库，什么时候必须微调？**
- **5.2 直觉算清 LoRA 显存账 + 5 分钟「弱智吧（`COIG-CQIA`）」中文逻辑破冰实验**
  - 拉取中科院开源 `m-a-p/COIG-CQIA`（弱智吧中文子集，240–500 条），在 `LLaMA-Factory WebUI` + `SwanLab` 看板下，8G 显卡 **5 分钟** 见证小模型识破中文双关逻辑陷阱！
- **5.3 中文数据清洗与脱敏构造实战**
- **5.4 主线实战：基于 `MSAgent-Bench` 与 `InstructIE` 微调中文通知抽取与工具调用**
  - 让本地小模型从口语化中文办公/课程群通知中精准提取结构化待办与填表 JSON，解决小模型调工具格式不稳痛点。
- **5.5 自建中文测试集评估微调前后效果 & 导出上线**
  - 用 20 行评测脚本跑 50 条独立测试题，定量对比微调前后准确率，合并导出权重重新接入 `llama-server` / `vLLM` / `lemonade-server`。
- **5.6 趣味进阶实战（网感与分身）：`WeClone` 微信聊天记录数字分身 / 小红书爆款种草体**
  - 使用个人脱敏微信聊天记录（参考 `WeClone`）或魔搭小红书高赞文案集进行风格 LoRA 微调，消除机械“AI 客服腔”，通过破功词下降率与双盲测试验证效果。
- 🎯 **本章实战产出**：微调权重 + SwanLab 训练记录 + 《自建中文测试集微调前后效果评估报告》。
- 🔗 **课程引用与进阶导流** ➡️ **[`happy-llm`（从零开始的大语言模型原理与实践教程）](https://github.com/datawhalechina/happy-llm)**（深入透彻理解 Transformer 架构原理、预训练、SFT 微调与强化学习对齐底层实现；更多单模型微调食谱亦可查阅 [`self-llm`](https://github.com/datawhalechina/self-llm)）。

---

### ⚡ 第 6 章｜速度不够怎么办？AMD 量化对比、零 C++ 算子加速与 `Hyperloom` 自动优化
> **本章目标**：全程零 C++ 代码！找一个模型直观对比 `AMD Quark` 量化前后差异，一键唤醒 AMD 融合算子，并体验用 `AMD-AGI/Hyperloom` 全自动寻优！

- **6.1 给慢助手做 AI 体检：看懂 TTFT / TPS 指标 + 用 `AMD-AGI/TraceLens` 自动定位瓶颈**
- **6.2 AMD 官方量化直观实战：用 `AMD Quark` 跑通同一模型量化前后对比**
  - 选定一个基准开源模型，使用 AMD 官方 `AMD Quark` (`pip install amd-quark`) 跑通量化，**实测记录量化前 vs 量化后的文件大小（GB）、显存占用、生成速度（tok/s）与回答效果对比表**。
- **6.3 零 C++ 算子加速：`@torch.compile` 自动融合 + 一键唤醒 `hipBLASLt` / `rocWMMA FA` / `AITER`**
  - 15 行纯 Python 加 `@torch.compile` 体验 ROCm Triton 自动算子融合；在 `llama-server` 开启 `ROCBLAS_USE_HIPBLASLT=1 -fa on`，在 `vLLM` 开启 **`VLLM_ROCM_USE_AITER=1`**（启用 `ROCm/aiter` 高性能融合算子库），实测长文首字与吞吐翻倍！
- **6.4 让 WorkBuddy、`Qoder` 与 Codex 秒回：Prefix Caching 前缀缓存与投机采样**
  - 开启前缀缓存化解 Agent 数万字系统提示词重复计算（首字等待从 10s 压至 <1s），配合小模型投机采样与 `lemonade-server` NPU+GPU 分流。
- **6.5 【Agentic 自动寻优压轴】结合 [`AMD-AGI/Hyperloom`](https://github.com/AMD-AGI/Hyperloom) 与 `AMD Skills` 实现全自动优化**
  - 在 `Qoder` / Codex 中通过 `@hyperloom/inference_optimizer/SKILL.md` 一句话驱动 `Magpie → TraceLens → Arbor + GEAK` 全自动调参与重写热点算子闭环。
- 🎯 **本章实战产出**：《模型量化前后与算子加速实测对比表（体积 / 显存 / 速度 / 效果）》。
- 🔗 **课程引用与进阶导流** ➡️ **[`hello-gpu`（GPU 架构与算子优化实战）](https://github.com/datawhalechina/hello-gpu)**（系统深造 AMD GPU 硬件计算存储架构、ROCm 工具链与手写 HIP / Triton 高性能算子优化）。

---

### 🏆 第 7 章｜收获时刻：成功故事、个人作品集与简历成果锻造
- **7.1 撰写你的成功故事**：「我把自己的 AMD 电脑，变成了学习和工作的 AI 助手」成长纪实模板。
- **7.2 个人 AI 作品集五件套打包清单**：应用代码 + 演示实录 + 训练/微调记录 + 量化与测试集效果评估报告 + 个人贡献说明。
- **7.3 简历项目打磨与面试追问指南**：用实测数据证明**模型部署、数据处理、应用集成与评测能力**。
- **7.4 毕业深造驿站**：按职业规划一键选定下一门 Datawhale 专精课程（`self-llm` / WorkBuddy 教程 / `all-in-rag` / `hello-agents` / `happy-llm` / `hello-gpu`）。

---

### 🎮 第 8 章｜学以致用：15 个趣味实战案例库（首发办公置顶 · 全覆盖 8 大场景）

> 所有案例底层统一仅使用 **`llama-server`、`vLLM` 或 `lemonade-server`** 提供模型服务；每个案例强制包含 **「🎬 演示 + ✅ 支持配置 + 💻 代码与步骤 + 🔍 验证方法 + 🚀 扩展任务（附专精课程导流）」** 五要素，并执行**新手试学机制**持续完善常见问题（FAQ）。

| 编号 | 所属场景 | 案例名称 | 核心玩法与进阶课程导流衔接 | 最低硬件档位 |
|:---:|:---|:---|:---|:---:|
| **Case 01** | 📊 报告生成 | **🌟【首发王牌】一句话文档变可编辑汇报 PPT（WorkBuddy + `dashi-ppt-skill`）** | 套用 1020 套版式并浏览器微调，导出 WPS/Office 原生可编辑 `.pptx`（🔗 [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)） | 🟢 8G / 🔵 16G |
| **Case 02** | 🗂️ 文件整理 | **【办公必装】AI 自动填表与杂乱文件夹智能归档** | 批量解析报销单/简历 PDF 自动填入标准 Excel 表并分类归档目录（🔗 [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)） | 🟢 8G |
| **Case 03** | 🗺️ 任务规划 | **【到点自动干活】用 WorkBuddy AI 定时任务自动处理日常办公事项** | 配置 RRULE 定时任务自动完成统计填表与日报周报生成（🔗 [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)） | 🟢 8G |
| **Case 04** | 📊 报告生成 | **全本地万字深度调研报告专家（`Local Deep Research`）** | 多轮检索网页、论文与本地资料，输出带完整引用角标研报（🔗 [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) / [`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🔵 16G |
| **Case 05** | 📄 论文阅读 | **多篇英文文献横评对比表 & 组会汇报 PPT 一键生成（WorkBuddy + `dashi-ppt-skill`）** | `@` 多篇论文 PDF 先出方法局限性横评表，再直出组会 `.pptx`（🔗 [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) / [`all-in-rag`](https://github.com/datawhalechina/all-in-rag)） | 🔵 16G |
| **Case 06** | 📄 论文阅读 | **公式图表纹丝不乱：双语学术论文精翻神器（`PDFMathTranslate` + `Zotero`）** | 调用本地 `vLLM` 并发服务整本精翻英文 arXiv 论文 PDF（🔗 [`self-llm`](https://github.com/datawhalechina/self-llm)） | 🟢 8G |
| **Case 07** | 📚 个人知识库 | **`WorkBuddy + Obsidian` 双链第二大脑 & 可共享「知识管理 MCP」** | 本地笔记双链问答，并用 `FastMCP` 将知识库打包成 MCP 给别人调用（🔗 [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) + [`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🟢 8G |
| **Case 08** | 📚 个人知识库 | **复习讲义秒变双人播客 & 微信/小红书网感分身（`Open Notebook` + `WeClone`）** | PDF 讲义一键转双人播客 MP3，并用聊天记录微调专属网感分身（🔗 [`happy-llm`](https://github.com/datawhalechina/happy-llm)） | 🟢 8G / 🔵 16G |
| **Case 09** | 🎙️ 语音交互 | **全离线保密会议与网课纪要神器（`Meetily` + `Faster-Whisper`）** | 录音不上云，实时离线转写、区分发言人并提炼待办事项清单（🔗 [`self-llm`](https://github.com/datawhalechina/self-llm)） | 🟢 8G |
| **Case 10** | 🎙️ 语音交互 | **桌面 Live2D 实时语音伴侣与英语口语陪练（`Open-LLM-VTuber` / 仓库 `MiniCPM-o`）** | 与桌面有表情、支持随时打断的本地虚拟伙伴全离线实时练口语（🔗 [`self-llm`](https://github.com/datawhalechina/self-llm)） | 🟢 8G / 🔵 16G |
| **Case 11** | 💻 编程助手 | **`Qoder` 全离线 AI 编程搭档实战** | 将本地三大引擎接入 `Qoder` IDE，零成本实现补全、重构与单测生成（🔗 [`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🟢 8G / 🔵 16G |
| **Case 12** | 💻 编程助手 | **终端自主修 Bug 智能体与百行源码解剖（`OpenAI Codex CLI` & 仓库 `toy-cli`）** | 终端自动读报错改代码跑测试，30 分钟读懂极简 Coding Agent 循环（🔗 [`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🟢 8G / 🔵 16G |
| **Case 13** | 🗺️ 任务规划 | **网页自动化操控与智能旅行规划师（`Browser-Use` + 仓库高德 MCP `smart-travel-planner`）** | 本地 AI 操控浏览器比价查资料，结合 `HelloAgents` 调高德 MCP 排日程（🔗 [`hello-agents`](https://github.com/datawhalechina/hello-agents)） | 🔵 16G |
| **Case 14** | 🗺️ 任务规划 | **24 小时全隐私多端个人管家中枢（仓库 `OpenClaw` + Prefix Caching 加速）** | 手机飞书发消息，家里 AMD 电脑自动拆解复杂任务并利用前缀缓存秒回（🔗 [`hello-agents`](https://github.com/datawhalechina/hello-agents) / [`hello-gpu`](https://github.com/datawhalechina/hello-gpu)） | 🟣 旗舰 / ☁️ 云 |
| **Case 15** | 🎮 游戏互动 | **从扫雷挑战赛、《我的世界》AI 搭档到文字跑团剧场（仓库 `Minesweeper Agent` & `Mindcraft`）** | 在扫雷看清 Agent 推理闭环，再放进 Minecraft 自动生存或跑团剧场（🔗 [`hello-agents`](https://github.com/datawhalechina/hello-agents) / [`all-in-rag`](https://github.com/datawhalechina/all-in-rag)） | 🟢 8G / 🔵 16G |

---

### 🌟 第 9 章｜拓展视野：AMD 优秀开源项目、前沿行业案例与开发者资源百宝箱（收官专章）
> **本章目标**：汇聚 AMD 官方前沿 AI 开源生态、本仓库高阶行业/科研标杆案例与全套开发者资源，为深入探索 AMD 平台的同学提供长期索引字典！

- **9.1 AMD 官方明星开源 AI 与 Agent 项目巡礼**
  - **端侧智能体与本地服务栈**：[`amd/gaia`](https://github.com/amd/gaia)（AMD 官方本地隐私 Agent 框架，深度协同 Ryzen AI NPU+iGPU）、[`lemonade-sdk/lemonade`](https://github.com/lemonade-sdk/lemonade) 与 `FastFlowLM`、[`amd/skills`](https://github.com/amd/skills)（官方 ROCm 优化知识包）与 [`ROCm/rocm-cli`](https://github.com/ROCm/rocm-cli)。
  - **自动优化、量化与高性能算子基础设施**：[`AMD-AGI/Hyperloom`](https://github.com/AMD-AGI/Hyperloom)、[`AMD-AGI/TraceLens`](https://github.com/AMD-AGI/TraceLens)、[`amd/Quark`](https://github.com/amd/Quark)、[`ROCm/aiter`](https://github.com/ROCm/aiter)（高性能融合算子库）、[`AMD-AGI/Primus`](https://github.com/AMD-AGI/Primus) & `Primus-Turbo`（大规模训练加速框架）与 [`ROCm/TheRock`](https://github.com/ROCm/TheRock)（新一代模块化 ROCm 构建基座）。
- **9.2 基于 AMD 平台的前沿科研与行业高阶案例库（无损沉淀仓库硬核资产）**
  - 🤖 **具身智能机器人实战**：`Every Embodied` ROCm 具身智能策略复刻（MuJoCo 抓取任务复刻 ACT、SmolVLA、Pi0/Pi0.5，联动 [`every-embodied`](https://github.com/datawhalechina/every-embodied)）。
  - 📚 **工业级推荐系统实战**：`Torch-RecHub` ROCm 推荐系统建模与训练（CTR 精排、序列建模与召回，联动 [`torch-rechub`](https://github.com/datawhalechina/torch-rechub)）。
  - 🚀 **多机多卡从零训练大模型**：`Happy-LLM` AMD 集群分布式训练实战（联动 [`happy-llm`](https://github.com/datawhalechina/happy-llm)）。
  - ⚡ **旗舰 AI PC 百亿/千亿模型极限部署与视觉游戏 AI**：Ryzen AI Max+ 395（128GB 统一内存）原生部署 `DeepSeek-V4-Flash` / `Qwen3.8-Flash-CIRU`，以及 `YOLOv10 微信跳一跳` 视觉自动化实战。
  - 🔬 **底层算子与硬件编程实验室**（原 `03-infra` 归档）：AMD GPU 硬件架构解密、手写第一个 HIP 算子、自定义 PyTorch C++ 算子（直通 [`hello-gpu`](https://github.com/datawhalechina/hello-gpu)）。
- **9.3 AMD 开发者算力福利、架构字典与资源百宝箱**
  - ☁️ **算力与社区支持**：**AMD Developer Cloud / Radeon Cloud** 免费算力申领通道、高校科研支持计划与 Datawhale × AMD 飞书交流社群。
  - 🗺️ **官方权威资料与硬件速查总表**：[ROCm 官方文档](https://rocm.docs.amd.com/)、[ROCm 技术博客](https://rocm.blogs.amd.com/)、Radeon RX（RDNA 3/4）/ Ryzen AI（RDNA 3.5 + XDNA 2）/ Instinct（CDNA 3/4）全系 `gfx` Target 速查字典。
  - 🧰 **已验证配置总库与排错急救手册**：三大推理引擎与客户端（`WorkBuddy` / `Qoder` / `Codex`）已验证配置文件索引、新手试学全局 FAQ 与开源贡献指南（保持所有历史 URL 路由 100% 兼容零 404）。
