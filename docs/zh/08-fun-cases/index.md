# 第 8 章｜学以致用：15 个趣味实战案例库（首发办公置顶 · 全覆盖 8 大场景）

> **设计准则**：所有案例底层统一仅使用 **`llama-server`、`lemonade-server` 或 `vLLM`** 三大标准服务引擎；全部严格遵循 **[统一案例「五要素」标准模板](./_CASE_TEMPLATE.md)**（🎬 演示 + ✅ 支持配置 + 💻 代码与步骤 + 🔍 验证方法 + 🚀 扩展任务与专精课程导流），并配套 **新手试学闭环机制** 持续沉淀真实 FAQ！

---

## 一、15 个趣味案例全景导航表（8 大日常与创意场景 100% 覆盖）

你可以学完第 2 章就先挑置顶的 **办公三件套（Case 01–03）** 马上用起来，也可以按自己最感兴趣的场景自由点菜：

| 编号 | 场景分类 | 案例名称与直达入口 | 最低档位 | 进阶导流衔接 |
|:---:|:---|:---|:---:|:---|
| **Case 01** | 📊 报告生成 | 🌟 **[一句话文档变可编辑汇报 PPT（WorkBuddy + `dashi-ppt-skill`）](./office-and-study-cases.md#case-01)** | 🟢 8G | [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) |
| **Case 02** | 🗂️ 文件整理 | **[AI 自动填表与杂乱文件夹智能归档](./office-and-study-cases.md#case-02)** | 🟢 8G | [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) |
| **Case 03** | 🗺️ 任务规划 | **[用 WorkBuddy AI 定时任务自动处理日常办公事项](./office-and-study-cases.md#case-03)** | 🟢 8G | [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) |
| **Case 04** | 📊 报告生成 | **[全本地万字深度调研报告专家（`Local Deep Research`）](./office-and-study-cases.md#case-04)** | 🔵 16G | [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) / [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 05** | 📄 论文阅读 | **[多篇英文文献横评对比表 & 组会汇报 PPT 一键生成](./office-and-study-cases.md#case-05)** | 🔵 16G | [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) / [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) |
| **Case 06** | 📄 论文阅读 | **[公式图表纹丝不乱：双语学术论文精翻神器（`PDFMathTranslate`）](./office-and-study-cases.md#case-06)** | 🟢 8G | [`self-llm`](https://github.com/datawhalechina/self-llm) |
| **Case 07** | 📚 个人知识库 | **[`WorkBuddy + Obsidian` 双链第二大脑 & 可共享「知识管理 MCP」](./office-and-study-cases.md#case-07)** | 🟢 8G | [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) + [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 08** | 📚 个人知识库 | **[复习讲义秒变双人播客 & 微信/小红书网感分身（`Open Notebook` + `WeClone`）](./office-and-study-cases.md#case-08)** | 🟢 8G | [`happy-llm`](https://github.com/datawhalechina/happy-llm) |
| **Case 09** | 🎙️ 语音交互 | **[全离线保密会议与网课纪要神器（`Meetily` + `Faster-Whisper`）](./voice-code-agent-game-cases.md#case-09)** | 🟢 8G | [`self-llm`](https://github.com/datawhalechina/self-llm) |
| **Case 10** | 🎙️ 语音交互 | **[桌面 Live2D 实时语音伴侣与英语口语陪练（`Open-LLM-VTuber` / `MiniCPM-o`）](./voice-code-agent-game-cases.md#case-10)** | 🟢 8G | [`self-llm`](https://github.com/datawhalechina/self-llm) |
| **Case 11** | 💻 编程助手 | **[`Qoder` 全离线 AI 编程搭档实战](./voice-code-agent-game-cases.md#case-11)** | 🟢 8G | [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 12** | 💻 编程助手 | **[终端自主修 Bug 智能体与百行源码解剖（`Codex CLI` & `toy-cli`）](./voice-code-agent-game-cases.md#case-12)** | 🟢 8G | [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 13** | 🗺️ 任务规划 | **[网页自动化操控与智能旅行规划师（`Browser-Use` + 高德 MCP）](./voice-code-agent-game-cases.md#case-13)** | 🔵 16G | [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 14** | 🗺️ 任务规划 | **[24 小时全隐私多端个人管家中枢（`OpenClaw` + Prefix Caching）](./voice-code-agent-game-cases.md#case-14)** | 🟣 旗舰 / ☁️ 云 | [`hello-agents`](https://github.com/datawhalechina/hello-agents) / [`hello-gpu`](https://github.com/datawhalechina/hello-gpu) |
| **Case 15** | 🎮 游戏互动 | **[从扫雷挑战赛、《我的世界》AI 搭档到文字跑团剧场](./voice-code-agent-game-cases.md#case-15)** | 🟢 8G | [`hello-agents`](https://github.com/datawhalechina/hello-agents) / [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) |

---

## 二、分辑实操入口

- 👉 **[上辑（Case 01 ～ Case 08）：办公三件套、深度研报、论文精读精翻与双链知识库](./office-and-study-cases.md)**
- 👉 **[下辑（Case 09 ～ Case 15）：离线语音伴侣、`Qoder`/Codex 编程、任务规划与游戏互动](./voice-code-agent-game-cases.md)**

---

## 三、🧪 新手试学机制（欢迎成为试学官点亮贡献者徽章！）

为了彻底杜绝“作者觉得很简单、新手第一步就报错”，本章节所有案例均实行 **新手试学闭环机制**：
1. 任意零基础同学在独立跟做案例时，凡遇到停顿超过 5 分钟的步骤或环境差异报错，均可直接前往仓库 Issues 提交 **「🧪 新手试学反馈卡」**；
2. 凡有 ≥2 位同学在同一处卡壳，维护团队将直接升级默认脚本与正文说明，并将你的排错方案署名沉淀至对应案例的 `新手试学常见问题（FAQ）`，自动计入仓库 **Contributors**！
