# ✈️ 智能旅行规划助手 - HelloAgents AMD ROCm 版本

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)
[![HelloAgents](https://img.shields.io/badge/HelloAgents-Framework-blue)](https://github.com/datawhalechina/hello-agents)

</div>

**智能旅行规划助手** 是一个基于 [HelloAgents](https://github.com/datawhalechina/hello-agents) 框架开发的 AI Agent 应用，能够在 AMD 395 AI PC 上本地运行大语言模型，结合 MCP 协议调用高德地图 API，自动生成完整的旅行规划方案。所有计算都在本地完成，保护你的隐私数据。

> HelloAgents 项目地址：[*Link*](https://github.com/datawhalechina/hello-agents)

***接下来我将带领大家亲自动手，一步步实现智能旅行规划助手的搭建和使用，让 AI 帮你规划完美旅行！***

## 项目亮点

- 🏠 **本地部署**：在 AMD 395 AI PC 上运行 30B 参数大模型，无需联网
- 🔒 **隐私安全**：旅行偏好、预算信息等敏感数据不上传云端
- 🗺️ **真实数据**：集成高德地图 API，获取真实景点和天气信息
- 📝 **自动生成**：输出完整的 Markdown 格式旅行计划文档
- 🤖 **Agent 架构**：基于 HelloAgents 框架，支持工具调用和多轮对话

## 技术栈

- **AI 框架**：[HelloAgents](https://github.com/datawhalechina/hello-agents) - 简化 Agent 应用开发
- **协议标准**：MCP (Model Context Protocol) - AI 调用外部服务的标准协议
- **推理平台**：AMD ROCm - 开源 GPU 计算平台
- **模型部署**：Ollama / LM Studio / 玲珑"智玲同学"
- **外部服务**：高德地图 API（景点搜索、天气查询）

## Step 1: 环境准备

本文基础环境如下：

```
----------------
AMD Ryzen™ AI 9 HX 395 处理器
Python 3.10+
Ollama / LM Studio / 玲珑"智玲同学"（三选一）
----------------
```

> 本文默认学习者使用的是 AMD 395 AI PC 或其他搭载 AMD ROCm 支持的显卡设备。关于本地模型部署，请参考 [Getting Started with ROCm Deploy](../../01-Deploy/README.md)

### 1.1 安装 Python 依赖

首先 `pip` 换源加速下载并安装依赖包：

```shell
# 升级 pip
python -m pip install --upgrade pip

# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 HelloAgents 和相关依赖
pip install hello-agents requests python-dotenv uv
```

### 1.2 获取高德地图 API Key

1. 访问 [高德开放平台](https://lbs.amap.com/)
2. 注册并登录账号
3. 进入控制台，创建应用
4. 获取 API Key（选择 Web 服务 API）

## Step 2: 本地模型部署

### 方案选择

根据你的系统和使用习惯，选择以下三种方案之一：

| 工具               | 适用平台          | 特点                                | 推荐场景               |
| ------------------ | ----------------- | ----------------------------------- | ---------------------- |
| **Ollama**         | Windows/Mac/Linux | 命令行工具，轻量级，模型管理简单    | 适合熟悉命令行的开发者 |
| **LM Studio**      | Windows/Mac/Linux | 图形界面，可视化操作，适合新手      | 适合需要图形界面的用户 |
| **玲珑"智玲同学"** | 玲珑AI工作站      | 预装工具，一键部署，针对AMD 395优化 | 适合玲珑工作站用户     |

### 2.1 使用 Ollama（推荐）

Ollama 是一个轻量级的本地模型管理工具，支持 AMD ROCm 加速。

**安装 Ollama：**

```shell
# Windows/Mac/Linux 访问官网下载
# https://ollama.ai/

# 或使用命令行安装（Linux）
curl -fsSL https://ollama.ai/install.sh | sh
```

**下载并运行模型：**

```shell
# 下载 Qwen2.5 32B 模型（推荐）
ollama pull qwen2.5:32b

# 或下载其他模型
ollama pull deepseek-r1:32b
ollama pull llama3.1:70b
```

**启动 Ollama 服务：**

```shell
# Ollama 会自动在后台运行
# 默认端口：http://localhost:11434
ollama serve
```

### 2.2 使用 LM Studio

1. 下载并安装 [LM Studio](https://lmstudio.ai/)
2. 在模型库中搜索并下载模型（推荐 Qwen2.5 32B）
3. 在 "Local Server" 标签页启动服务
4. 确认端点地址（默认：`http://127.0.0.1:1234`）
5. 在设置中选择 **AMD ROCm** 作为推理引擎

### 2.3 使用玲珑"智玲同学"

如果你使用的是玲珑 AI 工作站：

1. 打开预装的"智玲同学"应用
2. 选择合适的模型（推荐 30B+ 参数）
3. 一键启动本地服务
4. 记录服务端点地址

## Step 3: 配置高德地图 API Key

当前示例通过 `uvx amap-mcp-server` 直接拉起 MCP 服务，因此不需要额外创建 `mcp_config.json`，也不需要单独维护 `mcp_amap_server.py`。

### 3.1 安装 uv

```shell
pip install uv
```

### 3.2 配置 API Key

你可以任选一种方式：

1. 直接编辑 `travel_planner_mcp.py` 里的 `amap_api_key`
2. 或者将脚本中的 `amap_api_key` 置空，再通过环境变量注入

```shell
setx AMAP_MAPS_API_KEY "your_amap_api_key_here"
```

## Step 4: 运行智能旅行规划助手

### 4.1 下载项目代码

```shell
# 克隆 hello-rocm 仓库
git clone https://github.com/datawhalechina/hello-rocm.git

# 进入项目目录
cd hello-rocm/05-AMD-YES/05-hello-agents/smart-travel-planner
```

### 4.2 配置模型端点

编辑 `travel_planner_mcp.py`，修改 `HelloAgentsLLM(...)` 中的模型配置：

```python
self.llm = HelloAgentsLLM(
    model="Qwen3-30B-2507-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key="amd395"
)
```

如果你使用的是其他本地模型端点，只需要把 `model` 和 `base_url` 改成自己的配置即可。

### 4.3 运行助手

```shell
python travel_planner_mcp.py
```

### 4.4 使用示例

当前示例脚本默认会直接生成一份“杭州 3 日游”规划；如果你想换成自己的城市和预算，请修改 `main()` 里的参数：

```python
result = planner.plan_travel(
    destination="杭州",
    days=3,
    budget=3000,
    preferences="自然风光和历史文化"
)
```

AI 助手会自动：
1. 调用高德地图 API 搜索景点
2. 查询天气信息
3. 根据预算和偏好规划行程
4. 生成完整的 Markdown 旅行计划文档

## Step 5: 查看生成的旅行计划

生成的旅行计划会保存为 Markdown 文件，文件名会随目的地变化，例如 `杭州_3日游_MCP.md`。仓库中还提供了中英文两个示例输出，位于 `smart-travel-planner/examples/` 目录。

文件包含：
- 📅 每日详细行程安排
- 🏛️ 景点介绍和推荐理由
- 🍜 美食推荐
- 💰 预算分配建议
- 🌤️ 天气信息
- 🚇 交通建议

## 项目结构

```
smart-travel-planner/
├── assets/
│   ├── picture7-1.png
│   └── picture7-2.png
├── docs/
│   ├── amd395-helloagents-smart-travel-planner-en.md
│   ├── amd395-helloagents-smart-travel-planner-zh.md
│   └── amd395-helloagents-smart-travel-planner-zh.pdf
├── examples/
│   ├── hangzhou-3-day-mcp.md
│   └── 杭州_3日游_MCP.md
└── travel_planner_mcp.py
```

## 常见问题

### Q1: 模型运行速度慢怎么办？

- 确保使用 AMD ROCm 作为推理引擎
- 选择量化版本的模型（如 Q4_K_M）
- 降低模型参数量（如使用 7B 或 14B 模型）

### Q2: 高德地图 API 调用失败？

- 检查 API Key 是否正确配置
- 确认 API Key 的服务类型为 "Web 服务"
- 检查网络连接是否正常

### Q3: 如何更换其他地图服务？

- 可以替换为百度地图、腾讯地图等
- 修改 `travel_planner_mcp.py` 中 `MCPTool(...)` 的服务配置
- 替换相应地图服务的 API Key 和查询参数

## 进阶扩展

- 🌐 **多语言支持**：添加英文、日文等多语言旅行规划
- 🎨 **UI 界面**：使用 Streamlit 或 Gradio 构建 Web 界面
- 📊 **数据可视化**：生成行程地图和预算图表
- 🤝 **多人协作**：支持多人共同规划旅行
- 🔄 **实时更新**：根据天气变化动态调整行程

## 参考资源

- [HelloAgents 官方文档](https://github.com/datawhalechina/hello-agents)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [高德地图 API 文档](https://lbs.amap.com/api/)
- [AMD ROCm 文档](https://rocm.docs.amd.com/)
- [中文详细教程](./smart-travel-planner/docs/amd395-helloagents-smart-travel-planner-zh.md)
- [English Tutorial](./smart-travel-planner/docs/amd395-helloagents-smart-travel-planner-en.md)

---

<div align="center">

**用 AMD GPU 打造你的专属 AI 旅行助手！** 🗺️

Made with ❤️ by the hello-rocm community

</div>

