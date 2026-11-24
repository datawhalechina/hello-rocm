# 第 0 章｜导读：把你的 AMD 电脑变成学习和工作的 AI 助手

> **hello-rocm** —— **每个 AI 小白，从零开始，动手实践的第一站。**  
> **基于 AMD 平台的 AI 与 Agentic AI 入门实践手册（全栈总览先导课）。**

---

## 0.1 为什么从你手头的电脑开始养一个本地 AI？

很多同学对“本地跑大模型”的第一印象是：是不是得懂底层 C++ 编译？是不是得买几万块的专业显卡？平时用网页版 AI 不就够了吗？

当你真正把自己的 **AMD 游戏本、轻薄本或台式机** 变成专属本地 AI 助手后，你会立刻收获网页端无法替代的三大自由：

1. **零 Token 账单焦虑**：无论是让 AI 一次性精翻 50 页英文论文 PDF、反复修改 20 版汇报 PPT，还是挂着定时任务每天自动汇总表格，电费之外 **0 元 API 费用**，大胆试错不心疼。
2. **100% 隐私不出本机**：未公开的实验数据、财务报销明细、个人微信聊天记录、课堂录音与私人 Obsidian 笔记全程留在你的硬盘里，断网也能照常工作。
3. **亲手改造的工程掌控感**：你不再只是“只会聊天的普通用户”，而是亲手走完 **「模型服务部署 → 办公/编程工具接入 → 私人知识库与 MCP 封装 → 中文微调 → 量化加速」** 全链路的创造者，最终沉淀出写进简历的硬核项目成果。

---

## 0.2 学习者成功故事与 6 站体验旅程

本手册贯穿始终的主线故事只有一句话：

> **“我把自己的 AMD 电脑，变成了学习和工作的 AI 助手。”**

我们彻底摒弃“先背枯燥概念、再查硬件架构表”的传统写法，按照零基础同学的**真实痛点与获得感曲线**设计了 **6 站体验旅程**：

```mermaid
flowchart LR
    S1["第1站：破冰启程<br/>(第0~1章)<br/>算清显存三本账<br/>15分钟跑通本地服务"] --> S2["第2站：办公提效<br/>(第2章·首发重点)<br/>接入WorkBuddy出PPT/填表<br/>接入Qoder与Codex"]
    S2 --> S3["第3站：注入记忆<br/>(第3章)<br/>提示词6关闯关<br/>WorkBuddy+Obsidian知识库"]
    S3 --> S4["第4站：长出手脚<br/>(第4章)<br/>把知识库做成可共享MCP<br/>自定义办公SKILL.md"]
    S4 --> S5["第5站：痛点攻坚<br/>(第5~6章)<br/>能力不够→中文LoRA微调<br/>速度不够→Quark量化+算子加速"]
    S5 --> S6["第6站：收获远航<br/>(第7~9章)<br/>简历五件套 + 15个趣味案例<br/>AMD生态与专精课深造"]
```

| 旅程站点 | 对应章节 | 你可能会遇到的真实想法 | 本站核心成果 |
|:---:|:---:|:---|:---|
| **第 1 站：破冰启程** | [第 1 章](../01-first-model/index.md) | “不知道我的显卡能跑多大模型，怕装坏环境。” | 算清参数/显存/精度三本账，用三大引擎之一跑通 `/v1` 接口 |
| **第 2 站：办公提效** | [第 2 章](../02-connect-tools/index.md) | “黑框框能聊天了，怎么帮我做 PPT、填 Excel、写代码？” | 跑通 **WorkBuddy（大师 PPT + 自动填表 + 定时任务）** 与 **`Qoder` / `Codex`** |
| **第 3 站：注入记忆** | [第 3 章](../03-prompt-and-rag/index.md) | “AI 不懂我的专业课件，还总爱一本正经瞎编！” | 闯关 6 节提示词课，搭好 **`WorkBuddy + Obsidian`** 双链私人笔记库 |
| **第 4 站：长出手脚** | [第 4 章](../04-mcp-and-skills/index.md) | “我的个人知识库能不能跨软件用，甚至分享给队友调用？” | 用 `FastMCP` 封装 **可共享「知识管理 MCP」** + 自定义办公 `SKILL.md` |
| **第 5 站：痛点攻坚** | [第 5 章](../05-fine-tune/index.md) & [第 6 章](../06-optimize-speed/index.md) | “小模型抽中文通知偶发写歪；长文对话首字响应慢！” | 完成首次中文 LoRA 微调评测 + `AMD Quark` 单模型量化前后对比与 `Hyperloom` 加速 |
| **第 6 站：收获远航** | [第 7 章](../07-portfolio-resume/index.md)～[第 9 章](../09-amd-ecosystem/index.md) | “怎么把成果写进简历？接下来去哪门专精课深造？” | 打包简历作品集五件套、畅玩 15 个趣味案例、探索 AMD 开源宝藏与 Datawhale 专精课 |

---

## 0.3 3 分钟硬件对号入座（含无卡免费云算力兜底）

无需背诵复杂的底层代号！打开你的电脑看一眼显存/内存大小，直接领取你的专属颜色档位：

- **Windows 11 查看方式**：按 `Ctrl + Shift + Esc` 打开「任务管理器」→ 点击「性能」→ 点击 **GPU** 查看「专用 GPU 内存」或「共享 GPU 内存」。
- **Ubuntu 查看方式**：在终端输入 `amd-smi` 或 `rocminfo | grep Marketing`。

| 你的硬件档位 | 典型代表设备 | 全书主推模型规格 | 学习畅玩度 |
|:---|:---|:---|:---|
| **🟢 入门档**<br/>（8G–12G 独显 / 16G–32G 核显本） | RX 7600 / 7650 GRE、Ryzen AI 7/9 轻薄本（780M / 880M / 890M） | **`Gemma 4 E4B`**（`Q4_K_M` / FLM NPU） | 畅玩部署、WorkBuddy 办公、`Qoder`、知识库 MCP、`Gemma 4 E4B` 微调与量化对比 |
| **🔵 甜点档**<br/>（16G–24G 独显） | RX 7800 XT / 7900 GRE / 7900 XTX / RX 9070 XT | **`Qwen3.5-9B`** & **`Gemma 4 12B`**（`Q4_K_M` / `Q8_0`） | 全模块丝滑畅玩（中文办公填表、`Qwen3.5-9B` 微调、`AMD Quark` 量化对比与 `vLLM` 并发） |
| **🟣 旗舰档**<br/>（64G–128G 统一内存 AI PC） | Ryzen AI Max+ 395（Radeon 8060S + XDNA 2 NPU）迷你工作站 | **`Gemma 4 12B` / `Qwen3.5-9B` (`BF16`/`Q8_0`)** 及更大 MoE | 满血运行超长上下文多端智能体（如 `OpenClaw`）、多模型常驻与异构加速 |
| **☁️ 云端兜底档**<br/>（手头暂无 AMD 显卡 / Mac 用户） | 浏览器或 SSH 接入免费云端算力 | 同甜点档 / 旗舰档 | 立即阅读 **[AMD Radeon Cloud 使用指南](../cloud/amd-radeon-cloud.md)** 或 **[AUP Learning Cloud 指南](../cloud/aup-learning-cloud.md)** 免费实操全书内容！ |

---

## 0.4 全栈总览定位：本课程与 Datawhale 专精课程的引用关系

`hello-rocm` 是你在 AI 世界的**全景总览第一站**。我们在各章中精选引入了 Datawhale 社区多门王牌课程的趣味实验，并在每章结尾为你准备了直通专精课程的深造桥梁：

| 在 `hello-rocm` 学习的章节 | 双向联动的 Datawhale 专精课程 | 学完本章后去专精课深造什么？ |
|:---|:---|:---|
| **第 1 章｜本地模型部署** | 🔗 **[`self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)** | 解锁百余款主流开源大模型的部署方案与生态全集 |
| **第 2 章｜AI 办公与工具接入** | 🔗 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)** | 深入解锁全套桌面 AI 办公工作流与多场景自动化 |
| **第 3 章｜提示词与个人知识库** | 🔗 **[`all-in-rag`（RAG 技术全栈指南）](https://github.com/datawhalechina/all-in-rag)** | 掌握生产级文档解析、混合检索、重排（Rerank）与 GraphRAG |
| **第 4 章｜知识管理 MCP 与 Skills** | 🔗 **[`hello-agents`（从零开始构建智能体）](https://github.com/datawhalechina/hello-agents)** | 从零手写 ReAct / Plan-and-Solve 智能体框架与多 Agent 协同 |
| **第 5 章｜第一次中文模型微调** | 🔗 **[`happy-llm`（从零开始的大语言模型原理与实践）](https://github.com/datawhalechina/happy-llm)** | 透彻理解 Transformer 手搓实现、预训练、微调与强化学习对齐原理 |
| **第 6 章｜AMD 量化与算子加速** | 🔗 **[`hello-gpu`（GPU 架构与算子优化实战）](https://github.com/datawhalechina/hello-gpu)** | 深入 AMD GPU 硬件架构、ROCm 工具链与手写 HIP / Triton 高性能算子 |

### 💡 出发前装上伴学助教：`hello-rocm Skill`

如果你正在使用 WorkBuddy、`Qoder`、Claude Code 或 Codex，可以把下面这句话发给它，让本仓库内置的伴学助教随时陪你查文档、排报错：

```text
请加载当前仓库的 src/hello-rocm-skill 作为伴学技能，根据我的硬件环境引导我完成 hello-rocm 课程学习与问题排查。
```

准备好了吗？让我们正式进入 **[第 1 章｜直观认识数据类型与三大引擎最低成本部署](../01-first-model/index.md)**！
