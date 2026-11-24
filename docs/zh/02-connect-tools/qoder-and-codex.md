# 2.3 & 2.4 接入 `Qoder` 全离线编程 IDE 与 `Codex CLI` 终端智能体

搞定了日常办公三件套后，我们只需花 5 分钟复用同一个本地 `/v1` 服务接口，就能把你的代码编辑器和命令行终端也升级成**全离线、零流量费、代码绝不外泄**的 AI 编程搭档！

---

## 2.3 接入 `Qoder`：打造全离线零成本 AI 编程 IDE

`Qoder` 原生支持挂载标准的 `OpenAI Compatible`（`Chat Completions API`）自定义模型，并且原生兼容 `.agents/skills/` 模块化技能包。完整参数可对照仓库 **[`src/configs/qoder-openai-compatible.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/qoder-openai-compatible.json)**：

### 三步配置流程

1. **打开模型配置入口**：启动 `Qoder` IDE → 打开 **设置（Settings）** → 选择左侧 **模型（Model）** → 点击 **「添加模型 / Add Custom Model」**。
2. **填入本地三大引擎参数**（与 WorkBuddy `models.json` 不同，此处 `Base URL` **只填到 `/v1` 即可**）：
   - **接口协议（Provider / Protocol）**：选择 `OpenAI Compatible`（`Chat Completions API`）
   - **接口地址（Base URL）**：
     - 若用 `llama-server`：`http://127.0.0.1:8080/v1`
     - 若用 `lemonade-server`：`http://127.0.0.1:13305/v1`
     - 若用 `vLLM`：`http://127.0.0.1:8000/v1`
   - **API Key**：填 `rocm-local`（Lemonade 填 `lemonade`）
   - **Model ID**：填 `qwen3.5-4b-q4_k_m`（或你启动的模型名）
   - **上下文长度（Context Length）**：填 `32768`，最大输出填 `4096`，勾选支持 Tool Calling。
3. **一键校验并挂载项目技能**：
   - 点击右下角 **「校验并添加模型」**，提示连通成功后将其设为当前对话/编辑模型；
   - 在你的代码项目根目录执行以下命令，把本仓库伴学技能装入 `.agents/skills/`，让 `Qoder` 秒懂 ROCm 开发规范：
     ```bash
     mkdir -p .agents/skills
     cp -r src/hello-rocm-skill .agents/skills/hello-rocm
     ```

### 冒烟验证
在 `Qoder` 侧边栏选中任意一段 Python 函数，输入：“请为该函数补充类型注解，并用 `unittest` 编写包含正常输入与空值边界的单元测试”，观察本地显卡风扇轻转、毫秒级流式吐出完整单测代码！

---

## 2.4 接入 `Codex CLI` 与轻量终端智能体（含仓库 `toy-cli`）

如果你喜欢在终端命令行里直接让 AI 帮你查报错日志、写 Shell 脚本、自动跑测试修 Bug，可以直接将官方 **OpenAI Codex CLI**（以及本仓库内置的百行极简终端智能体 **`toy-cli`**）连上本地模型：

### 1. 配置 `~/.codex/config.toml`

直接复制本仓库已验证模板 **[`src/configs/codex-config.toml`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/codex-config.toml)** 到 `~/.codex/config.toml`：

```toml
model_provider = "rocm_llama"
model = "qwen3.5-4b-q4_k_m"

[model_providers.rocm_llama]
name = "AMD ROCm llama-server (Port 8080)"
base_url = "http://127.0.0.1:8080/v1"
env_key = "ROCM_LOCAL_API_KEY"
wire_api = "chat"
```

在终端设置环境变量并启动：

```bash
export ROCM_LOCAL_API_KEY="rocm-local"
codex "请检查当前目录下所有 Python 脚本的语法错误并给出修复建议"
```

### 2. 30 分钟彩蛋：解剖仓库百行代码终端 Agent `toy-cli`

想知道 `Qoder` 和 `Codex` 背后到底是怎么用几十行代码让大模型自动调命令行工具的吗？本仓库专门准备了一个零黑盒的极简实现：**[🧸 toy-cli：LLM 轻量化终端助手](../05-amd-yes/toy-cli.md)**（我们也会在第 8 章 Case 12 中深入拆解它）。

---

## 🎯 第 2 章通关里程碑

到这里，你的 AMD 电脑已经正式上岗成为全天候生产力工作台了：
- ✅ **日常办公**：WorkBuddy + `dashi-ppt-skill` 一键出可编辑 `.pptx` + 自动填 Excel + 定时处理日常工作；
- ✅ **日常编程**：`Qoder` IDE + `Codex CLI` 全离线写代码修 Bug。

但很快你就会遇到下一个真实问题：**“AI 虽然能干活了，但它完全不了解我的专业课讲义、组会文献和私人笔记，问深一点就开始瞎编！”**

👉 马上进入 **[第 3 章｜让助手读懂你：提示词闯关与 `WorkBuddy + Obsidian` 个人知识库实战](../03-prompt-and-rag/index.md)**！
