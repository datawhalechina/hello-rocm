# ✈️ Intelligent Travel Planning Assistant - HelloAgents AMD ROCm Edition

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)
[![HelloAgents](https://img.shields.io/badge/HelloAgents-Framework-blue)](https://github.com/datawhalechina/hello-agents)

</div>

**Intelligent Travel Planning Assistant** is an AI Agent application developed based on the [HelloAgents](https://github.com/datawhalechina/hello-agents) framework that can run large language models locally on AMD 395 AI PCs. It combines the MCP protocol to call Amap APIs and automatically generates complete travel planning solutions. All computations are completed locally, protecting your privacy data.

> HelloAgents Project: [*Link*](https://github.com/datawhalechina/hello-agents)

***In the following sections, I will guide you step by step to implement the setup and usage of the Intelligent Travel Planning Assistant, allowing AI to help you plan the perfect trip!***

## Project Highlights

- 🏠 **Local Deployment**: Run 30B parameter large language models on AMD 395 AI PC without internet connection
- 🔒 **Privacy & Security**: Sensitive data such as travel preferences and budget information are not uploaded to the cloud
- 🗺️ **Real Data**: Integrates Amap APIs to obtain real-time attraction and weather information
- 📝 **Auto-generation**: Outputs complete Markdown format travel planning documents
- 🤖 **Agent Architecture**: Based on HelloAgents framework, supports tool calling and multi-turn conversations

## Technology Stack

- **AI Framework**: [HelloAgents](https://github.com/datawhalechina/hello-agents) - Simplifies Agent application development
- **Protocol Standard**: MCP (Model Context Protocol) - Standard protocol for AI to call external services
- **Inference Platform**: AMD ROCm - Open-source GPU computing platform
- **Model Deployment**: Ollama / LM Studio / Linglong "Smart Linglong Classmate"
- **External Services**: Amap API (attraction search, weather query)

## Step 1: Environment Preparation

The basic environment for this guide is as follows:

```
----------------
AMD Ryzen™ AI 9 HX 395 Processor
Python 3.10+
Ollama / LM Studio / Linglong "Smart Linglong Classmate" (choose one)
----------------
```

> This guide assumes learners are using an AMD 395 AI PC or other devices with AMD ROCm support. For information about local model deployment, please refer to [Getting Started with ROCm Deploy](../../01-Deploy/README_EN.md)

### 1.1 Install Python Dependencies

First, switch pip source for faster downloads and install dependency packages:

```shell
# Upgrade pip
python -m pip install --upgrade pip

# Switch to faster PyPI source
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install HelloAgents and related dependencies
pip install hello-agents requests python-dotenv uv
```

### 1.2 Obtain Amap API Key

1. Visit [Amap Open Platform](https://lbs.amap.com/)
2. Register and log in to your account
3. Enter the console and create an application
4. Obtain the API Key (select Web Service API)

## Step 2: Local Model Deployment

### Plan Selection

Choose one of the following three solutions based on your system and usage preferences:

| Tool               | Supported Platforms | Features                                  | Recommended For        |
| ------------------ | ------------------- | ----------------------------------------- | ---------------------- |
| **Ollama**         | Windows/Mac/Linux   | CLI tool, lightweight, simple model mgmt  | Developers familiar with CLI |
| **LM Studio**      | Windows/Mac/Linux   | GUI, visual operation, beginner-friendly  | Users needing GUI      |
| **Linglong "Smart Linglong Classmate"** | Linglong AI Workstation | Pre-installed, one-click deployment, optimized for AMD 395 | Linglong Workstation Users |

### 2.1 Using Ollama (Recommended)

Ollama is a lightweight local model management tool that supports AMD ROCm acceleration.

**Install Ollama:**

```shell
# Visit the official website to download for Windows/Mac/Linux
# https://ollama.ai/

# Or install via command line (Linux)
curl -fsSL https://ollama.ai/install.sh | sh
```

**Download and Run Models:**

```shell
# Download Qwen2.5 32B model (recommended)
ollama pull qwen2.5:32b

# Or download other models
ollama pull deepseek-r1:32b
ollama pull llama3.1:70b
```

**Start Ollama Service:**

```shell
# Ollama will automatically run in the background
# Default port: http://localhost:11434
ollama serve
```

### 2.2 Using LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Search for and download models in the model library (Qwen2.5 32B recommended)
3. Start the service in the "Local Server" tab
4. Confirm the endpoint address (default: `http://127.0.0.1:1234`)
5. Select **AMD ROCm** as the inference engine in settings

### 2.3 Using Linglong "Smart Linglong Classmate"

If you are using a Linglong AI Workstation:

1. Open the pre-installed "Smart Linglong Classmate" application
2. Select an appropriate model (30B+ parameters recommended)
3. Start the local service with one click
4. Record the service endpoint address

## Step 3: Configure the AMap API Key

This example launches the MCP service directly with `uvx amap-mcp-server`, so you do not need a separate `mcp_config.json` file or a standalone `mcp_amap_server.py`.

### 3.1 Install uv

```shell
pip install uv
```

### 3.2 Configure the API Key

You can choose either approach:

1. Edit the `amap_api_key` value directly in `travel_planner_mcp.py`
2. Or set `amap_api_key` to empty in the script and inject the key through an environment variable

```shell
setx AMAP_MAPS_API_KEY "your_amap_api_key_here"
```

## Step 4: Run the Intelligent Travel Planning Assistant

### 4.1 Download Project Code

```shell
# Clone the hello-rocm repository
git clone https://github.com/datawhalechina/hello-rocm.git

# Enter the project directory
cd hello-rocm/05-AMD-YES/05-hello-agents/smart-travel-planner
```

### 4.2 Configure Model Endpoint

Edit `travel_planner_mcp.py` and update the `HelloAgentsLLM(...)` configuration:

```python
self.llm = HelloAgentsLLM(
    model="Qwen3-30B-2507-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key="amd395"
)
```

If you are using a different local endpoint, just replace `model` and `base_url` with your own values.

### 4.3 Run the Assistant

```shell
python travel_planner_mcp.py
```

### 4.4 Usage Example

The current sample script runs directly with a built-in Hangzhou 3-day example. To change the destination, budget, or preferences, edit the parameters in `main()`:

```python
result = planner.plan_travel(
    destination="杭州",
    days=3,
    budget=3000,
    preferences="自然风光和历史文化"
)
```

The AI assistant will automatically:
1. Call Amap APIs to search for attractions
2. Query weather information
3. Plan the itinerary based on budget and preferences
4. Generate a complete Markdown travel planning document

## Step 5: View Generated Travel Plans

The generated travel plan is saved as a Markdown file, and the filename changes with the destination, such as `杭州_3日游_MCP.md`. The repository also includes Chinese and English sample outputs under `smart-travel-planner/examples/`.

Files contain:
- 📅 Detailed daily itinerary arrangements
- 🏛️ Attraction introductions and recommendations
- 🍜 Food recommendations
- 💰 Budget allocation suggestions
- 🌤️ Weather information
- 🚇 Transportation suggestions

## Project Structure

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

## Frequently Asked Questions

### Q1: Model runs too slowly, what should I do?

- Make sure to use AMD ROCm as the inference engine
- Choose quantized versions of models (e.g., Q4_K_M)
- Reduce model parameters (e.g., use 7B or 14B models)

### Q2: Amap API call fails?

- Check if the API Key is correctly configured
- Confirm that the API Key's service type is "Web Service"
- Check if the network connection is normal

### Q3: How to switch to other map services?

- Can be replaced with Baidu Maps, Tencent Maps, etc.
- Adjust the service configuration inside `MCPTool(...)` in `travel_planner_mcp.py`
- Replace the map provider API key and query parameters accordingly

## Advanced Extensions

- 🌐 **Multi-language Support**: Add English, Japanese, and other language travel planning
- 🎨 **UI Interface**: Build web interface using Streamlit or Gradio
- 📊 **Data Visualization**: Generate itinerary maps and budget charts
- 🤝 **Multi-person Collaboration**: Support multiple people planning trips together
- 🔄 **Real-time Updates**: Dynamically adjust itinerary based on weather changes

## Reference Resources

- [HelloAgents Official Documentation](https://github.com/datawhalechina/hello-agents)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Amap API Documentation](https://lbs.amap.com/api/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Detailed Tutorial](./smart-travel-planner/docs/amd395-helloagents-smart-travel-planner-en.md)
- [Chinese Tutorial](./smart-travel-planner/docs/amd395-helloagents-smart-travel-planner-zh.md)

---

<div align="center">

**Build Your Own AI Travel Assistant with AMD GPU!** 🗺️

Made with ❤️ by the hello-rocm community

</div>
