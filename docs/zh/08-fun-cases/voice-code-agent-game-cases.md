# 第 8 章案例下辑（Case 09 ～ Case 15）：离线语音、`Qoder`/Codex 编程、任务规划与游戏互动

---

## Case 09：全离线保密会议与网课纪要神器（`Meetily` + `Faster-Whisper`） {#case-09}

- **🎬 效果演示**：会议录音与网课音频一秒不上云！本机实时离线语音转写，结束后自动调用你的本地模型提炼出《核心结论 + 负责人待办事项表（Action Items）》。
- **✅ 支持配置**：🟢 8G 入门档即可；ROCm / CPU `Faster-Whisper` + 本地 `llama-server` / `lemonade-server` + 开源 [`Zackriya-Solutions/meetily`](https://github.com/Zackriya-Solutions/meetily)。
- **💻 代码与步骤**：
  1. 启动本地 `llama-server`（端口 `8080`）；
  2. 在 `Meetily` 设置中选择 Custom OpenAI Endpoint（填入 `http://127.0.0.1:8080/v1`）；
  3. 录制一段 3 分钟模拟讨论或播放一段课程视频，点击 `Generate Summary`。
- **🔍 验证方法**：核对中文专业术语转写准确，且自动提取出结构化待办事项清单。
- **🚀 扩展任务与导流**：将生成的会议纪要通过第 4 章的「知识管理 MCP」自动存入 Obsidian；🔗 进阶学习 **[`self-llm`](https://github.com/datawhalechina/self-llm)**。

---

## Case 10：桌面 Live2D 实时语音伴侣与英语口语陪练（`Open-LLM-VTuber` / `MiniCPM-o`） {#case-10}

- **🎬 效果演示**：彻底告别打字！桌面上有丰富表情动作、支持随时插话打断的 AI 虚拟伙伴，陪你全离线练英语口语模拟面试，零尴尬、零流量费！
- **✅ 支持配置**：🟢 8G（[`Open-LLM-VTuber`](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) + `llama-server`）/ 🔵 16G（亦可体验原生全双工多模态 [`MiniCPM-o`](https://github.com/OpenBMB/MiniCPM-o)）。
- **💻 代码与步骤**：
  - **轻量路线**：克隆 `Open-LLM-VTuber`，在 `conf.yaml` 中将 `llm_provider` 设为 `openai_compatible_llm` 并指向本地 `8080` 端口；
  - **原生全双工路线**：参考 [`OpenBMB/MiniCPM-o`](https://github.com/OpenBMB/MiniCPM-o) 官方仓库与 [`self-llm`](https://github.com/datawhalechina/self-llm) 多模态部署指南一键拉起端到端实时音视频对话！
- **🔍 验证方法**：对着麦克风用英语自我介绍并在 AI 说话中途插话打断，确认语音响应延迟低于 2 秒且自然切换。
- **🚀 扩展任务与导流**：切换不同系统人设（雅思口语考官 / 技术面试官）；🔗 进阶学习 **[`self-llm`](https://github.com/datawhalechina/self-llm)**。

---

## Case 11：`Qoder` 全离线 AI 编程搭档实战 {#case-11}

- **🎬 效果演示**：在飞机高铁断网环境下或涉密内网代码库中，使用 `Qoder` IDE 直连本机 AMD 显卡服务，零成本完成跨文件代码解释、重构优化与单元测试自动生成。
- **✅ 支持配置**：🟢 8G（`Gemma 4 E4B`）/ 🔵 16G（推荐 `Gemma 4 12B` 或 `Qwen3.5-9B` 编程与重构能力更强）；配置对照 [`src/configs/qoder-openai-compatible.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/qoder-openai-compatible.json)。
- **💻 代码与步骤**：按 **[第 2.3 节完整配置步骤](../02-connect-tools/05-qoder-ide.md)** 在 `Qoder` 添加本地 `OpenAI Compatible` 模型并通过一键校验，开启第 6.4 节 Prefix Caching 实现多轮改代码秒回。
- **🔍 验证方法**：让 `Qoder` 为本仓库 `src/main-assistant/knowledge_mcp/server.py` 新增一个按日期过滤笔记的函数并生成单测，运行单测全部通过。
- **🚀 扩展任务与导流**：为你的项目编写专属代码规范 `.agents/skills/`；🔗 进阶学习 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)**。

---

## Case 12：终端自主修 Bug 智能体与百行源码解剖（`Codex CLI` & `toy-cli`） {#case-12}

- **🎬 效果演示**：在命令行里把一段报错的 Python 脚本丢给本地智能体，看它自动读取文件、分析 Traceback、修改源码并重新运行测试直到变绿；随后用 30 分钟彻底读懂 `toy-cli` 的百行核心代码！
- **✅ 支持配置**：🟢 8G / 🔵 16G；`Codex CLI`（配置见 [`src/configs/codex-config.toml`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/codex-config.toml)）+ 轻量级终端 Agent [`KMnO4-zx/toy-cli`](https://github.com/KMnO4-zx/toy-cli)。
- **💻 代码与步骤**：
  1. 先按 **[第 2.4 节完整教程](../02-connect-tools/06-codex-and-toy-cli.md)** 用 `Codex CLI` 连接本地服务修复一个故意写错缩进或除零错误的脚本；
  2. 继续跟随 **[第 2.4 节 `toy-cli` 百行源码剖析](../02-connect-tools/06-codex-and-toy-cli.md)**，亲手运行并逐行读懂一个极简 Coding Agent 是如何把 Shell 工具返回拼回 `messages` 循环的。
- **🔍 验证方法**：成功自动修复目标 Bug 脚本，并能在 `toy-cli` 中自定义新增一个查询本机 `amd-smi` 显存的命令工具。
- **🚀 扩展任务与导流**：动手从零扩展属于你自己的多工具编程 Agent；🔗 进阶学习 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)**。

---

## Case 13：网页自动化操控与智能旅行规划师（`Browser-Use` + 仓库高德 MCP） {#case-13}

- **🎬 效果演示**：让本地模型不仅能调用高德地图 MCP 查真实景点距离与天气排好 3 天旅行攻略，还能驱动真实浏览器自动打开网页查票价与整理攻略卡片！
- **✅ 支持配置**：🔵 16G 甜点档及以上；本仓库已验证精品源码 **[`src/cases/hello-agents/smart-travel-planner`](https://github.com/datawhalechina/hello-rocm/tree/dev_1.0/src/cases/hello-agents/smart-travel-planner)** + 开源 [`browser-use`](https://github.com/browser-use/browser-use)。
- **💻 代码与步骤**：
  1. 克隆或打开本仓库源码 [`src/cases/hello-agents/smart-travel-planner`](https://github.com/datawhalechina/hello-rocm/tree/dev_1.0/src/cases/hello-agents/smart-travel-planner)，配置高德 API Key（存入 `.env`，切勿提交 Git！）并将 `OPENAI_BASE_URL` 指向本地模型服务；
  2. 输入“帮我规划周末去杭州西湖 2 天 1 夜美食文化游”，查看自动生成的多日路线与天气穿衣建议。
- **🔍 验证方法**：确认规划结果中的景点地理顺路、餐饮推荐真实存在且包含实时天气查询反馈。
- **🚀 扩展任务与导流**：结合第 2 章 `dashi-ppt-skill` 把生成的旅行攻略一键导出成精美家庭出游路书 PPT；🔗 进阶学习 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)**。

---

## Case 14：24 小时全隐私多端个人管家中枢（`OpenClaw` + Prefix Caching 加速） {#case-14}

- **🎬 效果演示**：人不在电脑前，用手机飞书随手发一句消息，家里开着的 AMD 电脑立刻唤醒本地智能体拆解复杂任务、检索本地资料并把结果发回你手机，全程隐私数据不经第三方AI云！
- **✅ 支持配置**：🟣 旗舰档统一内存 AI PC（如 Ryzen AI Max+ 395 搭载旗舰主体模型 **`Qwen3.6-35B-A3B`**）或 ☁️ 云算力；开源本地智能体平台 [`OpenClaw`](https://github.com/openclaw/openclaw)。
- **💻 代码与步骤**：
  1. 启动本地 `llama-server` 或 `vLLM` 时**务必打开第 6.4 节讲过的 Prefix Caching**（`OpenClaw` 系统提示词长达 ~39K Tokens，开启缓存后次轮响应从十余秒直降至秒回！）；
  2. 在 `OpenClaw` 模型配置中接入本地 `http://127.0.0.1:8080/v1` 或 `http://127.0.0.1:13305/v1` 完成本地网关配置。
- **🔍 验证方法**：通过客户端或手机端发送多轮任务指令，观察第 2 轮起命中 Prefix Cache 实现秒级回复。
- **🚀 扩展任务与导流**：把第 4 章你的「知识管理 MCP」接入 `OpenClaw`，实现在手机上随时随地查家里电脑的 Obsidian 笔记；🔗 进阶学习 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)** 与 **[`hello-gpu`](https://github.com/datawhalechina/hello-gpu)**。

---

## Case 15：从扫雷挑战赛、《我的世界》AI 搭档到文字跑团剧场 {#case-15}

- **🎬 效果演示**：玩着游戏把 Agent 核心工程原理悟透！先在扫雷棋盘上直观对比“约束式 vs 自主式 Agent”的推理稳定性，再把本地模型放进《我的世界（Minecraft）》当生存搭档，或加载古风 LoRA 驱动单人文字冒险剧场！
- **✅ 支持配置**：🟢 8G / 🔵 16G；本仓库内置 **[`src/cases/minesweeper_agent`](https://github.com/datawhalechina/hello-rocm/tree/dev_1.0/src/cases/minesweeper_agent)**、古风数据集 [`src/fine-tune/datasets/huanhuan-100.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/fine-tune/datasets/huanhuan-100.json) 以及开源 [`Mindcraft`](https://github.com/kolbytn/mindcraft)。
- **💻 代码与步骤**：
  1. **第一关（必玩推荐！）**：打开仓库配套源码 [`src/cases/minesweeper_agent/minesweeper_agent.ipynb`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/cases/minesweeper_agent/minesweeper_agent.ipynb)，连接本地 `llama-server` 亲眼观察模型如何“看棋盘 → 逻辑推理 → 调工具翻格子”；
  2. **第二关（创意拓展）**：在 [`Mindcraft`](https://github.com/kolbytn/mindcraft) 的 `profiles/` 中将模型指向本地 `/v1` 接口召唤 AI 矿工，或用 [第 5.5 节人设 LoRA](../05-fine-tune/05-weclone-and-xhs.md) 结合规则书 RAG 搭建专属文字跑团守秘人（TRPG DM）！
- **🔍 验证方法**：成功跑完一局初级扫雷棋盘并对比记录约束式与自主式 Agent 的胜率差异。
- **🚀 扩展任务与导流**：尝试优化扫雷 Agent 的提示词让通关步数更少；🔗 进阶学习 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)** 与 **[`all-in-rag`](https://github.com/datawhalechina/all-in-rag)**。
