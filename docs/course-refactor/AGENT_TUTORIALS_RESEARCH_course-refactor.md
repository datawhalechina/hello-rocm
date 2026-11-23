# 全网优质有趣的 Agentic AI（Prompt / 知识库 RAG / 工具与 MCP / Skills）教程与实战调研报告

> **关联任务**：`course-refactor`（hello-rocm 2.0 第 3 章「玩转 Agentic AI」专项调研）  
> **调研目标**：告别枯燥概念灌输与重度框架源码阅读，搜集全网**最通俗、最好玩、互动感最强、且能直接在本地 AMD 模型 + WorkBuddy / Cherry Studio / Codex / VS Code 中跑出酷炫效果**的标杆教程与趣味小实验。

---

## 一、给小白的一张图直觉地图：再也不混淆 Prompt、RAG、MCP 与 Skills！

很多初学者一看到这四个词就头大。在 `hello-rocm 2.0` 第 3 章开篇，我们用 **「培养一位私人实习生」** 的直觉比喻一秒讲透四者分工：

| 核心模块 | 生活化直觉比喻 | 解决什么问题？ | 代表开源教程与生态 |
|:---|:---|:---|:---|
| **1. 提示词（Prompt）** | **🗣️ 交代任务的说话艺术** | 别让实习生猜心思：说清身份、给参考模板、规定输出表格/JSON 格式 | Anthropic 9关交互式闯关 Notebook、Datawhale `ai-prompting-for-everyone` |
| **2. 个人知识库（RAG）** | **📖 放在桌上的开卷参考书** | 实习生没背过你的私人笔记和课件，先翻书找到原文再回答，绝不瞎编 | Datawhale `all-in-rag`（「今天吃什么」菜谱切分实验）、`Cherry Studio + Obsidian` |
| **3. 工具与 MCP** | **🔌 万能 USB-C 外设手脚** | 光长嘴不行，给它接上查 12306 余票、刷 B 站字幕、查高德地图、读写文件的手脚 | HuggingFace `agents-course`、Datawhale `hello-agents`、`FastMCP` 中文趣味套件 |
| **4. 技能包（Skills）** | **📋 老师傅的 SOP 作业锦囊** | 平时合上不占显存脑容量，一旦喊它“做汇报 PPT / 写种草文 / 查代码”，按需翻开锦囊严丝合缝执行 | Datawhale `agent-skills-with-anthropic`、归藏 PPT Skill、本仓库 `hello-rocm-skill` |

---

## 二、四大模块全网最有趣标杆教程与实战玩法盘点

### 模块 1：🗣️ 提示词课程（Prompt）—— 把“背口诀”变成「本地 Jupyter 9 关闯关游戏」

- **全网最有趣标杆教程**：
  1. **`anthropics/prompt-eng-interactive-tutorial`（及中文汉化版 `TC999/anthropics-courses-cn`）**
     - **为什么全网封神**：它不是静态文档，而是一套自带判分与实时演练区（Playground）的 **9 关交互式闯关 Notebook**！每一关给你一个大模型“翻车回答”，让你动手改两行提示词（加角色、加 XML 分隔符、给 2 个 Few-shot 例子、加一步步思考、强制 JSON 格式、防幻觉认怂指令），跑通测试即过关！
  2. **Datawhale `ai-prompting-for-everyone` & `smart-prompt`**
     - **亮点**：专为非技术小白与中文语境打磨，重点讲清“语义压缩”与如何让参数量较小的模型稳定听话。
- **`hello-rocm 2.0` 改造落地设计**：
  - 将交互式闯关 Notebook 的接口直接指向学习者第 1 章刚跑起来的**本地 AMD 模型 API（`http://localhost:8000/v1`）**，设计 **《驯服本地 4B/8B 小模型 6 关挑战赛》**，30 分钟边玩边掌握本地模型最吃香的结构化提示词写法！

---

### 模块 2：📖 个人知识库（RAG）—— 用「黑暗料理菜谱实验」秒懂切片与向量检索

- **全网最有趣标杆教程**：
  1. **Datawhale `all-in-rag`（《RAG 技术全栈指南》）及其衍生趣味项目 `What-to-eat-today`（今天吃什么）**
     - **为什么好玩不枯燥**：很多 RAG 教程上来就讲余弦相似度公式把人看睡着；而用**「私人菜谱库（今天吃什么）」**做实验时——如果你用呆板的“每 150 字硬切一刀（Bad Chunking）”，会把《红烧肉》的“炒糖色火候”和《西红柿炒蛋》拼在一起，AI 当场教你做出黑暗料理！当你改成按 Markdown 标题语义切块 + 混合检索，AI 立刻精准告诉你冰箱里剩一颗土豆能做哪三道快手菜！
  2. **零代码个人阅读神器：`Cherry Studio + Obsidian 本地笔记库`**
     - **亮点**：无需写代码、无需装 Docker，直接用本地轻量 Embedding 模型（`bge-m3` / `nomic-embed-text`）关联本地课程笔记文件夹，句句带原文角标溯源。
- **`hello-rocm 2.0` 改造落地设计**：
  - **上半场（零代码 10 分钟体验）**：在桌面端把自己的一个 Markdown/PDF 课程文件夹变成开卷问答库；
  - **下半场（动手对比小实验）**：跑通 40 行 Python 的「菜谱/错题集切坏 vs 切对」对比实验，直观搞懂 Chunk Size、Overlap 与 Embedding 是怎么影响回答质量的。

---

### 模块 3：🔌 Agent 工具与 MCP —— 先插上「B站/12306/麦当劳趣味插座」，再 15 行代码手搓一个！

- **全网最有趣标杆教程**：
  1. **Hugging Face `agents-course`（全球超火零基础智能体课）& Datawhale `hello-agents`（《从零开始构建智能体》）**
     - **亮点**：用极简可视化循环讲清 `观察（Observation）→ 思考（Thought）→ 行动（Action）`（并辅以宝可梦/扫雷/旅行规划趣味场景），穿透框架迷雾。
  2. **`liaokongVFX/MCP-Chinese-Getting-Started-Guide`（中文 MCP 极速入门指南）+ `FastMCP` 框架**
     - **亮点**：把高深莫测的 MCP 协议简化为 Python 里的一个 `@mcp.tool()` 装饰器，**15 行代码、10 分钟** 就能手写出一个任何 AI 客户端都能识别的 MCP 服务！
- **全网最受年轻人欢迎的「中文趣味 MCP Server 体验清单」（在 WorkBuddy / Cherry Studio / VS Code 里一键挂载即玩）**：
  - 📺 **B站视频解析 MCP（`bilibili-mcp`）**：丢进一个 40 分钟网课视频链接，本地大模型直接拉取字幕与高赞弹幕，总结出时间轴笔记与核心考点；
  - 📖 **微信读书 MCP（`mcp-server-weread`）**：直连你的微信读书书架与划线笔记，让本地模型结合你读过的书帮你写读书报告；
  - 🚄 **12306 余票查询 MCP & 高德地图周边美食 MCP**：在聊天框一句话查周末回家高铁余票、搜学校周边步行 10 分钟高分小吃店；
  - 🍟 **麦当劳「薯条小宝」MCP（`mcd-desktop-pet`）** / **网易云音乐点歌 MCP**：查优惠套餐卡路里搭配、一句话让 AI 给你切一首适合写代码的纯音乐；
  - 🎨 **Blender 3D 建模 MCP（`blender-mcp`）**：一句话让大模型控制 Blender 自动搭出立体小房间！
- **`hello-rocm 2.0` 改造落地设计**：
  - **先玩现成的**：带学习者在本地客户端里挂上 1–2 个免复杂鉴权的趣味 MCP（如本地文件系统 + B站字幕/高德地图）；
  - **再自己造一个（极具成就感！）**：用 `FastMCP` 写一个 20 行代码的 **《我的 AMD 显卡健康管家 MCP Server》**（封装 `amd-smi` / 显存查询函数），直接在 WorkBuddy 或 VS Code 聊天框里问：“*摸摸我的显卡现在多少度？还能再开一个 8B 模型吗？*”，本地模型自动调你写的 MCP 实时汇报显卡状态！

---

### 模块 4：🧩 Agent Skills（`SKILL.md`）—— 搞懂“渐进式省显存秘籍”，给本地助手一键装大招！

- **全网最权威且易懂的标杆教程**：
  1. **Datawhale × Anthropic 官方课程中文版：`datawhalechina/agent-skills-with-anthropic`**
     - **核心认知突破**：专门讲透 **「渐进式披露（Progressive Disclosure）」** —— 以前大家把几千字规范全塞进全局提示词，本地小模型显存瞬间被占满且注意力分散；现在写成标准目录 `my-skill/SKILL.md`（头部只有两行 `name` 和 `description`），**平时几乎零显存开销，只有当你提到相关任务时，Agent 才自动翻开这本 `SKILL.md` 并调用配套 `scripts/` 脚本**！一次编写，同时兼容 **WorkBuddy (`.codebuddy/skills/`)、Codex (`.agents/skills/`)、Claude Code (`.claude/skills/`) 与 AMD GAIA**！
- **全网最实用好玩的开源 Skills 推荐体验单**：
  - 📊 **归藏高颜值动态网页 PPT 技能（`op7418/guizang-ppt-skill`）/ 官方 `pptx` 技能**：告别丑陋干瘪的纯文字大纲，一句话触发杂志墨水风 / 瑞士网格风可翻页汇报幻灯片；
  - 📕 **小红书种草工坊 Skill（`1-SKILL/zhongcao`）**：自动按痛点引入、Emoji 节奏与金字塔标签生成地道笔记；
  - 💼 **STAR 法则简历润色 Skill & Git 规范提交 `superpowers` Skill**：自动把日常代码改动整理成标准 Commit 与简历项目亮点；
  - 🚀 **近在眼前的活教材 —— 本仓库自带的 `src/hello-rocm-skill/SKILL.md` 与 AMD 官方 `amd/skills`**！
- **`hello-rocm 2.0` 改造落地设计**：
  - 直接拿本仓库现成的 `src/hello-rocm-skill/` 当第一份解剖标本，带小白看懂 `SKILL.md + references/ + scripts/` 三件套；
  - 手把手带小白在 **WorkBuddy / Codex** 里亲手写一个 **《课程实验报告 / 周报自动排版自检 Skill》**，输入 `/weekly-check` 自动检查格式、统计字数并生成漂亮汇报！

---

## 三、`hello-rocm 2.0` 第 3 章最终升级版课表结构

通过融合上述全网精华，第 3 章彻底告别“读死书”，变成 **4 节环环相扣、每节都有趣味作品产出的高互动实验课**：

- **`3.1` 提示词闯关课：6 关交互式 Notebook 驯服本地小模型**（借鉴 Anthropic 交互式闯关 + Datawhale `smart-prompt`，直连本机 API 实时通关）
- **`3.2` 个人知识库实战：从「黑暗料理菜谱切分实验」到私人笔记开卷问答库**（借鉴 Datawhale `all-in-rag`，零代码客户端建库 + 40 行切片对比实验）
- **`3.3` 万能插座 MCP 实战：玩转 B站/微信读书/地图趣味 MCP，并 20 行代码手搓「AMD 显卡管家 MCP」**（结合 `FastMCP` 与中文趣味 MCP 生态）
- **`3.4` 模块化技能包 Agent Skills：解剖渐进式省显存原理，在 WorkBuddy / Codex 中安装与定制专属 `SKILL.md`**（结合 `agent-skills-with-anthropic`、归藏 PPT Skill 与本仓库 `hello-rocm-skill`）
- **`3.5` 安全护栏与痛点悬念（无缝引出第 4、5 章！）**：
  - 给本地 Agent 加上防死循环与文件修改确认锁；
  - **留下真实悬念**：当你挂上多个 MCP 和长文档知识库后，如果发现 **4B 小模型偶尔把工具参数写歪了（→ 去第 4 章用中文微调治好它！）**，或者 **上下文变长后每次等回复要 10 秒钟（→ 去第 5 章开 Prefix Caching 秒回！）**，顺理成章进入下半程！
