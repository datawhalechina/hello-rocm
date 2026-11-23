# 全网本地 AI 与 Agentic AI 趣味实战案例调研与精选报告

> **关联任务**：`course-refactor`（hello-rocm 2.0 课程重构）  
> **调研范围**：GitHub 全球高星开源项目、国内开发者社区爆款项目、腾讯 WorkBuddy / VS Code / Codex 官方工作流、AMD 官方开源生态（`amd/gaia`、ROCm Blogs）以及 `hello-rocm` 现有优质沉淀。  
> **覆盖维度**：编程助手、个人知识库、论文阅读、文件整理、报告生成、语音交互、游戏互动、任务规划（8 大场景全覆盖）。

---

## 一、全网案例筛选的 5 大标尺（拒绝“玩具脚本”与“劝退巨石”）

为了让 AI 小白“看得懂、跑得通、用得爽、写得进简历”，本次全网筛选严格按以下 5 条标准过滤：

1. **强获得感与视觉冲击力**：做完立刻能解决一个每天都会遇到的真痛点（如 PDF 论文保留公式双语翻译、下载夹自动归档、录音秒变纪要），或者足够好玩能晒朋友圈（如《我的世界》AI 搭档、桌面语音虚拟伴侣、赛博数字分身）。
2. **无缝兼容 AMD 本地推理栈**：项目原生支持 **OpenAI 兼容 API（`vLLM` / `llama-server`）**、**Ollama** 或 **AMD Lemonade / GAIA**，无需魔改底层 CUDA 代码即可在 AMD ROCm 上运行。
3. **消费级硬件友好**：优先选择 **8GB–16GB 显存（或核显统一内存）** 即可流畅体验的项目；对显存/上下文要求高的项目明确标注参数调优方案或云算力兜底。
4. **巧妙反哺课程主线（微调 & 推理优化）**：案例在深入使用时能自然暴露出「小模型格式不稳/风格不像 → 引出第 4 章 LoRA 微调」或「长上下文首字慢 → 引出第 5 章 Prefix Caching 推理优化」。
5. **盘活现有仓库精华**：将 `hello-rocm` 已有的 `toy-cli`、`minesweeper-agent`、`minicpm-o`、`huanhuan-chat`、`smart-travel-planner`、`openclaw` 与全网标杆工具对标融合，避免重复造轮子。

---

## 二、8 大场景全网热门开源案例全景调研池（共 26 个真实项目）

### 场景 1：💻 编程助手（Coding Assistant）

| 项目 / 工具名称 | 来源 / GitHub | 核心玩法与全网热度 | AMD 本地接入方式 & 硬件门槛 | 课程教学价值评估 |
|:---|:---|:---|:---|:---|
| **Continue** | `continuedev/continue` (VS Code 插件, 25k+⭐) | 最适合本地显卡的开源 Copilot：支持 **1.5B 小模型做毫秒级 Tab 行内补全 + 7B/14B 模型做侧边栏问答重构** 的双模型分工 | 直连 `llama-server` / `vLLM` / `Ollama`；🟢 **8G 显存极度丝滑** | **【强烈推荐入课】** 小白零挫败感体验本地编程助手的首选 |
| **Cline** | `cline/cline` (VS Code 插件, 40k+⭐) | 全自主多文件 Coding Agent（AMD ROCm 官方博客推荐组合）：自动读仓库、执行终端命令、看报错自己改代码，原生支持 MCP | OpenAI 兼容接口连 `vLLM`（需开 16K–32K 上下文）；🔵 **建议 16G+ 显存** | **【强烈推荐入课】** 既是顶级编程 Agent，又是第 5 章 Prefix Caching 推理提速的最佳实测靶场 |
| **OpenAI Codex CLI** | `openai/codex` (官方开源终端 Agent) | 极客最爱的命令行编程智能体，支持 `--oss` 或 `~/.codex/config.toml` 配置本地 `vLLM` Provider 自动改代码跑单测 | `~/.codex/config.toml` 对接本地 `vLLM` / `Ollama`；🔵 **16G+ 显存佳** | **【强烈推荐入课】** 契合课程明确要求的 Codex 本地接入验证 |
| **Aider + `toy-cli`** | `Aider-AI/aider` (30k+⭐) & 仓库自有 `toy-cli` | `Aider` 是 Git 友好的终端结对编程标杆（ROCm 博客专门评测）；仓库 `toy-cli` 则是不到几百行看懂 Claude Code/Codex 核心循环的教学神器 | OpenAI 兼容 API；🟢 **8G 显存可跑**（如配 `AgentCPM-Explore 4B`） | **【强烈推荐融合】** 用 `Codex CLI` 教实用配置，用 `toy-cli` 拆解背后 100 行代码原理 |

---

### 场景 2：📚 个人知识库（Personal Knowledge Base）

| 项目 / 工具名称 | 来源 / GitHub | 核心玩法与全网热度 | AMD 本地接入方式 & 硬件门槛 | 课程教学价值评估 |
|:---|:---|:---|:---|:---|
| **Cherry Studio + Obsidian + MCP** | `CherryHQ/cherry-studio` (桌面端, 30k+⭐) | 国内最火全能桌面客户端：免 Docker 直接绑定本地 Obsidian 笔记文件夹建向量库（RAG），并通过 Filesystem MCP 让 AI 自动写笔记存回本地 | 接本地 `vLLM` / `llama.cpp` / `Ollama` + 本地 `bge-m3` Embedding；🟢 **8G 可跑** | **【强烈推荐入课】** 零代码门槛最高颜值“个人第二大脑 + MCP”闭环案例 |
| **Open Notebook / KnowNote** | `lfnovo/open-notebook` / `MrSibe/KnowNote` | **Google NotebookLM 开源本地平替**：导入课程讲义 PDF 和笔记，不仅能溯源问答，还能**一键把枯燥资料转成双人对话播客（Podcast MP3）** | 支持本地 OpenAI API / Ollama + 本地 TTS；🟢 **8G–16G 可跑** | **【强烈推荐入课】** “把期末复习资料变成双人播客路上听”对学生和职场人吸引力拉满 |
| **AMD GAIA** | `amd/gaia` (AMD 官方开源本地 Agent 工厂) | `pip install amd-gaia` 即可运行：专为 AMD Ryzen AI NPU 与 Radeon GPU 打造，支持 50+ 格式文档本地 RAG、`SKILL.md` 扩展与 `gaia eval` 自动评测修复 | 原生支持 Lemonade / `llama.cpp` ROCm / NPU；🟢/🔵 **AMD 平台亲儿子** | **【强烈推荐入课/主线联动】** 彰显 AMD 平台专属生态优势，自带评测闭环 |
| **AnythingLLM / Kotaemon / MaxKB** | `Mintplex-Labs/anything-llm` (40k+⭐) 等 | 老牌开箱即用多工作区文档问答桌面端/Web 端，内置 LanceDB 向量库，适合隔离管理多门学科或部门规章 | 原生支持本地 OpenAI API / Ollama；🟢 **8G 可跑** | 可作为 Cherry Studio / Open Notebook 的备选对比方案在文档中列出 |
| **WeClone（数字分身）** | `xming521/WeClone` (GitHub 爆款) | 导出自己的日常聊天记录或个人博客，用 **LoRA 微调 + 知识库** 训练一个说话语气、口头禅和你一模一样的本地“赛博分身” | ROCm 单卡 LoRA 微调 4B/7B 模型 + 本地部署；🔵 **16G 显存或云算力** | **【强烈推荐入课】** 全网极具传播力的个人微调 + 知识库结合案例，简历与社交分享双丰收 |

---

### 场景 3：📄 论文阅读（Paper Reading）

| 项目 / 工具名称 | 来源 / GitHub | 核心玩法与全网热度 | AMD 本地接入方式 & 硬件门槛 | 课程教学价值评估 |
|:---|:---|:---|:---|:---|
| **PDFMathTranslate + Zotero** | `Byaidu/PDFMathTranslate` (20k+⭐) | 科研党人手一个的神器：调用本地大模型翻译英文 arXiv PDF，**完美保留复杂数学公式、双栏排版与图表位置**，支持嵌入 Zotero 右键翻译 | CLI / GUI 直接配置本地 OpenAI 兼容端点；🟢 **8G 跑 4B/7B 翻译极快** | **【强烈推荐入课】** 刚需中的刚需！零 Token 成本无限翻整本英文教材与论文 |
| **WorkBuddy 论文精读与组会 PPT 生成** | 腾讯 WorkBuddy 桌面 Agent 工作台 | 用 `@` 拖入英文论文 PDF，开启 `Plan` 模式一键完成：① 提炼核心创新点；② 解构为结构化 Markdown；③ **直接生成组会汇报双语 PPT** | `~/.workbuddy-ai/models.json` 接本地 `vLLM` / `llama-server`；🔵 **16G 佳** | **【强烈推荐入课】** 完美展示 WorkBuddy 接入本地模型后的“直接交付成果”能力 |
| **Arxiv-Daily-Papers-Agent** | `ShiyuNee/Arxiv-Daily-Papers-Agent` | 每天自动抓取指定领域（如 AI Agent / CV）最新 arXiv 论文，调用本地模型过滤灌水文并生成中文「痛点 + 核心方案 TL;DR」早报 | Python 脚本调本地 `/v1/chat/completions`；🟢 **8G 轻松跑** | **【可与论文横评合并入课】** 代码简洁清晰，非常适合新手二创改造成自己的学科追踪器 |
| **GPT Academic（学术优化版）** | `binary-husky/gpt_academic` (65k+⭐) | 专为科研设计的交互界面：支持一键润色 LaTeX、批量总结 PDF 文件夹、剖析开源论文配套代码库 | 修改 `config.py` 指向本地 `vLLM` / `Ollama`；🟢 **8G–16G** | 知名度极高，可作为科研重度用户的扩展推荐 |

---

### 场景 4：🗂️ 文件整理（File Organization）

| 项目 / 工具名称 | 来源 / GitHub | 核心玩法与全网热度 | AMD 本地接入方式 & 硬件门槛 | 课程教学价值评估 |
|:---|:---|:---|:---|:---|
| **LlamaFS / Local-File-Organizer** | `iyaja/llama-fs` (6k+⭐) & `QiuYannnn/Local-File-Organizer` | **自组织智能文件夹**：开启隐私本地模式，自动读取乱糟糟的 `Downloads`/桌面（含 PDF 课件、发票截图、音频），按内容语义自动建分类目录并规范重命名，还支持 `Watch Mode` 实时监控归档 | 本地 LLM + 轻量视觉模型（如 `MiniCPM-V` / `Moondream`）；🟢 **8G 可跑** | **【强烈推荐入课】** 视觉效果极震撼（一秒把几百个乱名文件整整齐齐归类），且 100% 契合本地隐私诉求 |
| **WorkBuddy 本地杂乱表格清洗与归档** | 腾讯 WorkBuddy + 本地文件 `@` 引用 | 对着堆满格式不一 Excel/CSV 的文件夹发一句话：自动去重、统一日期与金额格式、跨表合并输出干净汇总表 | WorkBuddy 接本地 OpenAI 兼容服务；🟢/🔵 **8G–16G** | **【强烈推荐与文件整理/报告联动】** 办公党零代码福音 |

---

### 场景 5：📊 报告生成（Report Generation）

| 项目 / 工具名称 | 来源 / GitHub | 核心玩法与全网热度 | AMD 本地接入方式 & 硬件门槛 | 课程教学价值评估 |
|:---|:---|:---|:---|:---|
| **Local Deep Research** | `LearningCircuit/local-deep-research` / `langchain-ai/local-deep-researcher` | **OpenAI Deep Research 全本地开源平替**：针对一个复杂课题，自动多轮规划搜索词、检索网页/arXiv/本地私有文档、反思补齐信息盲区，输出带完整引用角标的万字深度调研报告 | 原生支持配置本地 `vLLM`、`llama.cpp`、`Ollama` 端点；🔵 **建议 16G+** | **【强烈推荐入课】** 极具代表性的多步反思 Agentic 工作流，产出物高级感十足 |
| **WorkBuddy 数据分析报告 + PPT 交付** | 腾讯 WorkBuddy（Craft / Plan 模式） | 上传本周工作流水账、实验 CSV 数据与参考 PDF，多 Agent 协同自动画统计图表、撰写结构化分析报告并同步生成汇报幻灯片 | WorkBuddy 接本地模型服务；🔵 **16G 佳** | **【强烈推荐入课】** 打通日常汇报全流程 |
| **Git 提交记录自动周报 Agent** | 社区热门 `git-weekly-report` / 自建百行脚本 | 自动扫描本地多个 Git 仓库本周的 `git log` 与 `diff`，按「核心进展 / Bug 修复 / 下周计划」秒出打工人/课题组周报 | 极简 Python 脚本调本地 `llama-server`；🟢 **8G 秒出结果** | **【推荐作为 30 分钟极速入门案例或扩展】** 代码不到 80 行，人人都能看懂改写 |

---

### 场景 6：🎙️ 语音交互（Voice Interaction）

| 项目 / 工具名称 | 来源 / GitHub | 核心玩法与全网热度 | AMD 本地接入方式 & 硬件门槛 | 课程教学价值评估 |
|:---|:---|:---|:---|:---|
| **Open-LLM-VTuber / MiniCPM-o 4.5** | `Open-LLM-VTuber/Open-LLM-VTuber` (8k+⭐) & 仓库自有 `minicpm-o` | **桌面实时语音伴侣 & 口语私教**：支持全离线语音识别（SenseVoice/Whisper）+ 本地大模型 + 本地 TTS + Live2D 桌宠形象，支持随时语音打断与调用 MCP 工具（或直接用仓库已有 `MiniCPM-o` 原生全双工音视频模型） | 直连本地 `vLLM` / `llama-server`；🟢/🔵 **8G–16G 畅玩** | **【强烈推荐入课】** 趣味性天花板！把冷冰冰的命令行瞬间变成桌面上能听会说有表情的 AI 伙伴 |
| **Meetily（全离线 AI 会议/课堂纪要）** | `Zackriya-Solutions/meetily` (6k+⭐) / `Faster-Whisper` + LLM | **隐私第一的本地会议与网课助手**：无需拉云端机器人进会，本地实时捕获系统声音与麦克风，Whisper 离线转写 + 说话人区分 + 本地大模型提炼议题结论与待办清单（Action Items） | 本地 Whisper (ROCm 加速) + 本地 OpenAI API / Ollama；🟢 **8G 可跑** | **【强烈推荐入课】** 学生录网课、职场开保密会的高频刚需，完美展示多模型本地流水线 |

---

### 场景 7：🎮 游戏互动（Game Interaction）

| 项目 / 工具名称 | 来源 / GitHub | 核心玩法与全网热度 | AMD 本地接入方式 & 硬件门槛 | 课程教学价值评估 |
|:---|:---|:---|:---|:---|
| **Mindcraft（我的世界 AI 智能体）** | `mindcraft-bots/mindcraft` (5k+⭐) | 全网爆火沙盒游戏 Agent：把本地大模型放进《Minecraft》，听懂指令自动寻路、砍树、挖矿、合成装备、盖房子，甚至支持加载专用微调权重多机器人协作生存 | 原生支持本地 OpenAI 兼容接口 / `vLLM` / `Ollama`；🔵 **8G–16G** | **【强烈推荐入课】** 极具传播力和趣味性的多步规划游戏 Agent 标杆 |
| **Minesweeper Agent（本地大模型玩扫雷）** | 仓库自有精品 `src/amd-yes/minesweeper_agent` | 无需安装大型游戏，在 4×4/8×8 扫雷棋盘上直观解剖 Agent 循环，对比「纯自主推理 vs 候选动作约束」胜率差距 | Jupyter + `llama.cpp` + Gemma4 E4B；🟢 **8G 显存 & 云端秒跑** | **【强烈推荐保留升级】** 最适合零基础看清 Agent 底层代码逻辑的轻量教学案例 |
| **AI 跑团守秘人（TRPG DM）/ 古风角色剧场** | `COC-AI-keeper` / `ZRIC-AI-TRPG-Engine` + 仓库自有 `Chat-甄嬛` | 本地模型扮演克苏鲁/奇幻文字冒险主持人（RAG 查规则书 + JSON 投骰检定 + 状态记忆），或结合自己微调的《甄嬛传》LoRA 玩古风宫廷互动剧场 | 本地 `llama-server` + LoRA 权重；🟢 **8G 可跑** | **【强烈推荐融合升级】** 把原本单纯聊天的 `Chat-甄嬛` 升级成带生命值/好感度状态机的文字冒险游戏！ |

---

### 场景 8：🗺️ 任务规划（Task Planning）

| 项目 / 工具名称 | 来源 / GitHub | 核心玩法与全网热度 | AMD 本地接入方式 & 硬件门槛 | 课程教学价值评估 |
|:---|:---|:---|:---|:---|
| **Browser-Use / UI-TARS Desktop** | `browser-use/browser-use` (50k+⭐) & `bytedance/UI-TARS-desktop` (15k+⭐) | **让本地 AI 替你操作浏览器和电脑桌面**：给一句高层目标（如“查三家平台某商品参数比价填表”或“汇总本周课程通知排成日程”），自动拆解步骤、点击网页、填表提取结果 | 对接本地 `vLLM` / `llama-server`（支持视觉或 DOM 解析）；🔵 **16G 佳** | **【强烈推荐入课】** 当前 Agentic AI 最直观、最震撼的“自主规划+执行”代表作 |
| **智能旅行规划师（HelloAgents + MCP）** | 仓库自有精品 `smart-travel-planner` | 输入预算、天数与喜好，Agent 自主规划调用高德地图、路线与天气 MCP 服务，生成精确到交通耗时的可执行攻略 | `HelloAgents` + MCP + 本地 OpenAI API；🔵 **8G–16G** | **【强烈推荐保留升级】** 学习手写 MCP Server 与多工具编排的标准范本 |
| **OpenClaw（全隐私多端个人管家中枢）** | 仓库自有精品 `openclaw` (`ValueCell-ai/ClawX`) | 飞书/Telegram/手机随时发一句话，家里的 AMD 电脑通过本地网关调度记忆（Memory）、技能（Skills）与工作区完成长线任务托管 | 本地长上下文模型（需配合第 5 章 Prefix Caching 优化 39K 初始上下文！）；🟣/☁️ | **【强烈推荐保留压轴】** 终极形态的“把家里 AMD 电脑变成 24 小时私人 AI 管家”大作 |

---

## 三、优中选优：最终入课的「黄金 15 案例」定稿推荐表（办公与学习案例全面最前置！）

从上述 26 个全网真实项目中，按 **「零代码日常 AI 办公案例（Case 01–04）放在最前面吸引所有小白 → 学术论文与知识库（Case 05–07）→ 会议纪要与语音（Case 08–09）→ 编程助手、自动化规划与游戏互动（Case 10–15）」** 的顺序编排，确保任何专业背景的学习者翻开第一页就能零门槛上手：

| 案例编号 | 所属场景 | 案例名称（绑定真实明星开源项目/工具） | 核心体验一句话 | 硬件门槛 | 对应开源基座 / 工具 |
|:---:|:---|:---|:---|:---:|:---|
| **Case 01** | 🗂️ 文件整理 | **【办公头牌】拯救杂乱下载夹：智能归档与文件盘点管家** | 拖入杂乱发票/课件/截图文件夹，自动建目录重命名并生成《文件清单盘点.md》（带 `dry-run` 安全预览） | 🟢 8G | `WorkBuddy` + `LlamaFS` + `MiniCPM-V` |
| **Case 02** | 📊 报告生成 | **【办公三件套连招】脏乱 Excel 清洗 ➡️ Word 分析报告 ➡️ 20页汇报 PPTX** | 告别繁琐公式与熬夜排版！`Ask→Plan→Craft` 三步全自动打通表格清洗、文字报告与去 AI 味幻灯片交付 | 🟢 8G / 🔵 16G | `WorkBuddy (Ask/Plan/Craft)` + 本地 API |
| **Case 03** | 📊 报告生成 | **一键高级感周报 Skill & 全本地深度调研报告专家** | 零散流水账/Git 记录 30 秒生成周报（封装专属 Skill），复杂课题多轮自动检索生成万字带引用研报 | 🟢 8G / 🔵 16G | `WorkBuddy Skill` + `Local Deep Research` |
| **Case 04** | 📄 论文阅读 | **多篇文献横评对比表 & 组会双语汇报 PPT 一键生成** | `@` 授权多篇英文论文或行业 PDF，自动输出研究方法局限性横评表，并直接交付组会汇报双语 PPT | 🔵 16G | `WorkBuddy (Plan模式)` + 本地长上下文 |
| **Case 05** | 📄 论文阅读 | **公式排版零损双语论文精翻器（PDFMathTranslate + Zotero）** | 调用本地模型一键把英文 arXiv 论文翻成双语对照 PDF，数学公式与双栏图表丝毫不乱 | 🟢 8G | `Byaidu/PDFMathTranslate` + `Zotero` |
| **Case 06** | 📚 个人知识库 | **Obsidian 私人第二大脑（Cherry Studio / AMD GAIA + MCP）** | 本地笔记免上传秒变向量库，还能通过 MCP 让 AI 自动写复盘存回本地目录 | 🟢 8G | `Cherry Studio` / `amd/gaia` + `Obsidian` |
| **Case 07** | 📚 个人知识库 | **讲义秒变双人播客 & 赛博数字分身（Open Notebook / WeClone）** | 把复习 PDF 转成双人对话播客路上听，或用微信聊天记录微调专属“数字分身” | 🟢 8G / 🔵 16G | `lfnovo/open-notebook` + `WeClone` |
| **Case 08** | 🎙️ 语音交互 | **全离线保密会议与网课纪要神器（Meetily + Whisper）** | 录音一秒不上云，实时捕获音频离线转写、区分发言人并提炼待办事项清单（Action Items） | 🟢 8G | `Zackriya-Solutions/meetily` + `Faster-Whisper` |
| **Case 09** | 🎙️ 语音交互 | **桌面 Live2D 实时语音伴侣与口语陪练（Open-LLM-VTuber / MiniCPM-o）** | 告别打字！和桌面上有表情、支持随时打断的本地虚拟伙伴实时练英语口语 | 🟢 8G / 🔵 16G | `Open-LLM-VTuber` + 仓库 `minicpm-o` |
| **Case 10** | 💻 编程助手 | **VS Code 离线双模编程搭档（Continue + Cline）** | 1.5B 毫秒级行内补全 + 8B/14B 自动改多文件代码，断网零成本写项目 | 🟢 8G / 🔵 16G | `Continue` + `Cline` + `vLLM`/`llama.cpp` |
| **Case 11** | 💻 编程助手 | **终端自主修 Bug 智能体（Codex CLI & `toy-cli`）** | 在命令行自动读报错、改代码跑单测，并花 30 分钟拆解百行 Agent 源码 | 🟢 8G / 🔵 16G | `OpenAI Codex CLI` + 仓库 `toy-cli` |
| **Case 12** | 🗺️ 任务规划 | **网页自动化操控与智能旅行规划师（Browser-Use & MCP）** | 让本地 AI 自动操作浏览器比价查资料，并调用高德 MCP 排好整趟旅行日程 | 🔵 16G | `browser-use` + 仓库 `smart-travel-planner` |
| **Case 13** | 🗺️ 任务规划 | **24 小时全隐私多端个人管家（OpenClaw 本地中枢）** | 手机飞书随手发指令，家里 AMD 电脑跨应用自动规划执行长线待办任务 | 🟣 旗舰 / ☁️ 云 | 仓库 `openclaw` + 第 5 章 Prefix Caching |
| **Case 14** | 🎮 游戏互动 | **从扫雷挑战赛到《我的世界》AI 生存搭档（Minesweeper & Mindcraft）** | 先在轻量扫雷棋盘看透 Agent 循环，再放进 Minecraft 自动砍树盖房 | 🟢 8G / 🔵 16G | 仓库 `minesweeper-agent` + `mindcraft` |
| **Case 15** | 🎮 游戏互动 | **AI 跑团守秘人（TRPG DM）与古风 NPC 互动剧场** | 结合 LoRA 微调人设与好感度/理智值状态机，打造单人可玩的文字冒险游戏 | 🟢 8G | 仓库 `huanhuan-chat` 升级 + `COC-AI-keeper` |

---

## 四、全网案例如何反哺第 4 章（微调）与第 5 章（推理优化）？

这批真实全网案例最大的妙处在于——它们在运行本地小模型时的**真实短板**，恰好成为第 4、5 章最生动的教学实验题：

| 真实案例痛点现场 | 触发的课程进阶章节 | 学习者可做出的简历级实验成果 |
|:---|:---|:---|
| 用 **WorkBuddy（Case 01–04）** 或 **Codex（Case 11）** 时，4B 小模型偶发写歪工具参数 JSON 导致执行中断 | **第 4 章：中文工具调用与通知抽取微调** | 基于魔搭中文工具集做 LoRA 微调，在自建 50 题测试集上格式合法率从 62% 提升至 98% |
| 用 **WeClone（Case 07）** 或 **古风跑团 NPC（Case 15）** 时，光靠提示词模仿语气生硬、一会儿就出戏 | **第 4 章：中文网感与人设微调（LoRA）** | 清洗 300 条个人/剧本中文语料微调，双盲辨识胜率飙升至 85%+ |
| 用 **WorkBuddy（内置数万字工具说明书）**、**Cline（Case 10）** 或 **OpenClaw（Case 13）** 时，每次回复首字要卡顿十几秒 | **第 5 章：AMD 算子唤醒 + Prefix Caching 前缀缓存** | 开启 `AITER` / `hipBLASLt` 与前缀缓存后，多轮办公对话首字延迟从 12s 骤降至 0.8s！ |
| 用 **PDFMathTranslate（Case 05）** 翻整本 30 页英文论文时，串行逐段请求太慢 | **第 5 章：AMD Quark 量化 + vLLM 并发加速** | 量化瘦身并开启多并发批处理，整篇论文双语翻译耗时从 8 分钟缩短至 55 秒！ |
