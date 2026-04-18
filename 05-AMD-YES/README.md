<div align=center>
  <h1>AMD-YES 🎮</h1>
  <strong>玩转 AMD GPU 的酷炫项目合集</strong>
</div>

<div align="center">

*让AI更有趣，让创意更自由* ✨

[返回主页](../README.md)

</div>

## 🚀 快速导航

| 本地运行（单卡） | 集群部署（多卡） |
|---------|---------|
| [🧸 toy-cli](./01-toy-cli/README.md) ✅️ | [🚀 happy-llm](./04-happy-llm/README.md) ✅️ |
| [🎮 微信跳一跳](./02-wechat-jump/README.md) ✅️| |
| [🎭 Chat-甄嬛](./03-huanhuan-chat/README.md) ✅️| |
| [✈️ 智能旅行规划助手](./05-hello-agents/README.md) ✅️| |

> ✅ 支持 | 🚧 开发中 

## 简介

&emsp;&emsp;AMD-YES 精选了一系列使用 AMD GPU 和 ROCm 平台的优秀项目实例。无论你是想快速体验大模型应用、学习计算机视觉、还是深入学习分布式训练，都能在这里找到合适的案例项目。

&emsp;&emsp;本模块分为两个阶段：首先在本地单卡快速上手各类应用，然后进阶到集群部署和分布式训练。每个项目都包含完整的代码和详细的教程指导。

## 项目列表

### 本地运行（单卡）

适合个人开发者快速体验，一块 AMD 显卡就能玩转！

---

#### 🧸 toy-cli - LLM 轻量化终端助手

[toy-cli](https://github.com/KMnO4-zx/toy-cli) 是一个简化的代码 Agent，提供极简的命令行调用大模型接口。

- **适合人群**：初学者、想快速学习 API 调用的用户
- **难度等级**：⭐
- **预计时间**：30 分钟

📖 [开始学习 toy-cli](./01-toy-cli/README.md)

---

#### 🎮 YOLOv10 微信跳一跳

[YOLOv10 微信跳一跳](https://github.com/KMnO4-zx/wechat-jump) 是基于 YOLOv10 目标检测的自动化游戏 AI，实时识别跳跃目标，精准计算距离。

- **适合人群**：对计算机视觉和游戏 AI 感兴趣的开发者
- **难度等级**：⭐⭐
- **预计时间**：1 小时

📖 [开始学习微信跳一跳](./02-wechat-jump/README.md)

---

#### 🎭 Chat-甄嬛 - 后宫语言模型

[Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) 是基于《甄嬛传》台词训练的 LoRA 微调模型，完美模仿甄嬛的语气和说话风格。

- **适合人群**：想学习模型微调和 LoRA 技术的开发者
- **难度等级**：⭐⭐
- **预计时间**：1.5 小时

📖 [开始学习 Chat-甄嬛](./03-huanhuan-chat/README.md)

---

#### ✈️ 智能旅行规划助手

[智能旅行规划助手](./05-hello-agents/智能旅行规划助手简易实战/README.md) 是基于 HelloAgents 框架的智能 Agent 应用，集成 MCP 协议调用高德地图 API，在 AMD GPU 上本地运行大模型。

- **适合人群**：想学习 Agent 框架和 MCP 协议的开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：2 小时

📖 [开始学习智能旅行规划助手](./05-hello-agents/智能旅行规划助手简易实战/AMD395×HelloAgents实战：智能旅行规划助手.md)

---

### 集群部署（多卡）

进阶玩法，分布式训练部署！

---

#### 🚀 Happy-LLM - 从零训练大模型

[Happy-LLM](https://github.com/datawhalechina/happy-llm) 提供分布式多机多卡训练的完整教程，包括大模型核心原理讲解和手把手的训练流程实现。

- **适合人群**：想深入学习大模型训练的进阶开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：3 小时+

📖 [开始学习 Happy-LLM](./04-happy-llm/README.md)

---

## 环境要求

### 硬件要求

- AMD GPU（支持 ROCm 的显卡，如 RX 7000 系列、MI 系列等）
- 建议显存 8GB 以上

### 软件要求

- 操作系统：Linux (Ubuntu 22.04+) 或 Windows 11
- ROCm 7.12.0/7.2.0 或更高版本
- Python 3.10~3.12

## 常见问题

<details>
<summary>Q: 如何确认我的 AMD GPU 是否支持 ROCm？</summary>

请参考 [ROCm 官方支持列表](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) 查看支持的 GPU 型号。

</details>

<details>
<summary>Q: 各个项目的学习顺序是什么？</summary>

建议按以下顺序学习：
1. 先从 `toy-cli` 开始，熟悉大模型调用基础
2. 选择感兴趣的案例深入学习（游戏 AI、对话模型、Agent 应用）
3. 完成基础项目后，挑战 `Happy-LLM` 进阶分布式训练

</details>

## 参考资源

- [self-llm 完整教程](https://github.com/datawhalechina/self-llm)
- [HelloAgents AI Agent 框架](https://github.com/datawhalechina/hello-agents)
- [ROCm 官方文档](https://rocm.docs.amd.com/)
- [hello-rocm 主项目](../)

---

<div align="center">

**欢迎贡献更多项目案例！** 🎉

[提交 Issue](https://github.com/datawhalechina/hello-rocm/issues) | [提交 PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
