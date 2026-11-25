# 第 8 章｜学以致用：15 篇独立五要素趣味实战案例超市（办公置顶 · 覆盖 8 大场景）

> **本章定位（全场景应用游乐场）**：所有 15 个精选案例均采用 **「单案例单篇独立教程」** 设计，底层统一仅连接你在第 1 章启动的 **`llama-server`（`:8080`）**、**`lemonade-server`（`:13305`）** 或 **`vLLM`（`:8000`）** 标准 `/v1` 服务；每一篇严格对齐 **[统一案例「五要素」标准模板](./_CASE_TEMPLATE.md)**（🎬 效果演示 + ✅ 支持配置清单 + 💻 完整代码与步骤 + 🔍 验收标准 + 🚀 扩展任务与 Datawhale 专精课程导流 + ❓ 新手试学 FAQ）！

---

## 一、15 篇独立案例全景选菜矩阵（覆盖全部 8 大高频场景）

你可以刚学完第 2 章就直接点开置顶的 **办公自动化三连王牌（Case 01～03）** 马上解放双手，也可以按自己的兴趣、专业和显卡档位自由点菜：

| 案例编号 | 所属场景 | 独立完整教程入口 | 最低算力档位 | 推荐主体模型 | 对接的专精深造课程 |
|:---:|:---|:---|:---:|:---|:---|
| **Case 01** | 📊 报告生成 | 🌟 **[一句话文档变可编辑汇报 PPT（WorkBuddy + `dashi-ppt-skill`）](./case-01-dashi-ppt.md)** | 🟢 8G | `Qwen3.5-9B` / `Gemma 4 E4B` | [Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) |
| **Case 02** | 🗂️ 文件整理 | 🌟 **[AI 零幻觉自动填 Excel 与杂乱文件夹智能归档](./case-02-auto-excel-archive.md)** | 🟢 8G | `Qwen3.5-9B` / `Gemma 4 E4B` | [Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) |
| **Case 03** | 🗺️ 任务规划 | 🌟 **[用 WorkBuddy AI 定时任务自动处理日常办公事项](./case-03-scheduled-office.md)** | 🟢 8G / NPU | `Gemma 4 E4B` / `Qwen3.5-9B` | [Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) |
| **Case 04** | 📊 报告生成 | **[全本地万字深度调研报告专家（`Local Deep Research`）](./case-04-local-deep-research.md)** | 🔵 16G | `Qwen3.5-9B` / `Gemma 4 12B` | [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) · [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 05** | 📄 论文阅读 | **[多篇英文文献横评对比表 & 组会汇报 PPT 串联生成](./case-05-paper-review-ppt.md)** | 🔵 16G | `Gemma 4 12B` / `Qwen3.5-9B` | [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) · [WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) |
| **Case 06** | 📄 论文阅读 | **[公式图表纹丝不乱：双语学术论文精翻神器（`PDFMathTranslate`）](./case-06-pdf-math-translate.md)** | 🟢 8G / 🔵 16G | `Qwen3.5-9B` / `Gemma 4 E4B` | [`self-llm`](https://github.com/datawhalechina/self-llm) |
| **Case 07** | 📚 个人知识库 | **[`WorkBuddy + Obsidian` 双链第二大脑 & 可共享「知识管理 MCP」](./case-07-obsidian-knowledge-mcp.md)** | 🟢 8G | 四大主体模型全通 | [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) · [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 08** | 📚 个人知识库 | **[复习讲义秒变双人对话播客 & `WeClone` 网感分身联动](./case-08-podcast-and-weclone.md)** | 🟢 8G / 🔵 16G | `Qwen3.5-9B` / `Gemma 4 E4B` | [`happy-llm`](https://github.com/datawhalechina/happy-llm) |
| **Case 09** | 🎙️ 语音交互 | **[全离线保密会议与网课纪要神器（`Meetily` + `Faster-Whisper`）](./case-09-offline-meeting-minutes.md)** | 🟢 8G | `Qwen3.5-9B` / `Gemma 4 E4B` | [`self-llm`](https://github.com/datawhalechina/self-llm) |
| **Case 10** | 🎙️ 语音交互 | **[桌面 Live2D 实时语音伴侣与英语口语陪练（`Open-LLM-VTuber`）](./case-10-live2d-voice-tutor.md)** | 🟢 8G / 🔵 16G | `Gemma 4 E4B` / `MiniCPM-o` | [`self-llm`](https://github.com/datawhalechina/self-llm) |
| **Case 11** | 💻 编程助手 | **[`Qoder` 全离线 AI 编程搭档：内网代码重构与单测生成](./case-11-qoder-offline-ide.md)** | 🟢 8G / 🔵 16G | `Gemma 4 12B` / `Qwen3.5-9B` | [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 12** | 💻 编程助手 | **[终端自主修 Bug 智能体与百行源码手搓（`Codex CLI` & `toy-cli`）](./case-12-codex-and-toy-cli.md)** | 🟢 8G | `Gemma 4 12B` / `Gemma 4 E4B` | [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 13** | 🗺️ 任务规划 | **[网页自动化操控与高德 MCP 智能旅行规划师](./case-13-travel-planner-browser.md)** | 🔵 16G | `Qwen3.5-9B` / `Qwen3.6-35B-A3B` | [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **Case 14** | 🗺️ 任务规划 | **[24 小时全隐私飞书多端个人管家中枢（`OpenClaw` + Prefix Caching）](./case-14-openclaw-hub.md)** | 🟣 旗舰 / ☁️ 云 | **`Qwen3.6-35B-A3B`** | [`hello-agents`](https://github.com/datawhalechina/hello-agents) · [`hello-gpu`](https://github.com/datawhalechina/hello-gpu) |
| **Case 15** | 🎮 游戏互动 | **[从扫雷挑战赛、《我的世界》AI 搭档到文字跑团剧场](./case-15-minesweeper-and-game-ai.md)** | 🟢 8G / 🔵 16G | `Qwen3.5-9B` / `Gemma 4 E4B` | [`hello-agents`](https://github.com/datawhalechina/hello-agents) · [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) |

---

## 二、按你的身份一键领取「周末推荐三连套餐」

- **💼 职场白领 / 社团骨干高效套餐**：`Case 01（大师 PPT）` ➡️ `Case 02（自动填表与归档）` ➡️ `Case 03（定时任务值守）`
- **🎓 大学生期末 / 研究生组会套餐**：`Case 06（双语公式论文精翻）` ➡️ `Case 05（多篇论文横评直出组会 PPT）` ➡️ `Case 08（讲义转双人播客路上听）`
- **💻 极客与硬核开发者炫技套餐**：`Case 11 & 12（Qoder + Codex/toy-cli 全离线编程）` ➡️ `Case 14（OpenClaw 飞书全隐私管家）` ➡️ `Case 15（扫雷推理与 Minecraft AI 搭档）`

---

## 三、🧪 新手试学闭环机制（欢迎提交反馈点亮仓库 Contributors 徽章！）

为了彻底消灭“作者觉得很简单、新手第一步就卡住”的教程通病，本章全部 15 篇案例均实行 **社区试学共建机制**：
1. 任何零基础同学在独立跟做任一案例时，若在某一步停顿超过 5 分钟或遇到系统差异报错，欢迎前往仓库 Issues 提交 **[🧪 新手试学反馈卡](https://github.com/datawhalechina/hello-rocm/issues/new/choose)**；
2. 你的排错经验一经采纳，将直接署名增补进该案例底部的 `❓ 新手试学常见问题（FAQ）` 专区，自动成为 `hello-rocm` 官方开源贡献者；
3. 如果你也想贡献第 16、第 17 个酷炫本地应用案例，直接复制 **[五要素标准案例模板 `_CASE_TEMPLATE.md`](./_CASE_TEMPLATE.md)** 即可发起 PR！

---

👉 马上点开置顶第一号人气案例：**[Case 01｜一句话文档变可编辑汇报 PPT（WorkBuddy + `dashi-ppt-skill`）](./case-01-dashi-ppt.md)**！
