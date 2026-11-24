# hello-rocm 2.0 已验证配置模板库（三大服务引擎 × 三大客户端）

本目录存放经过实测验证的本地 OpenAI 兼容接入配置，统一覆盖 **`llama-server`（端口 `8080`）**、**`lemonade-server`（端口 `13305`）** 与 **`vLLM`（端口 `8000`）**。

## 文件清单与放置路径

| 配置文件 | 适用客户端 | 本机放置位置 / 使用方法 | 核心避坑点 |
|:---|:---|:---|:---|
| [`workbuddy-models.json`](./workbuddy-models.json) | **WorkBuddy** | 复制至 `~/.workbuddy-ai/models.json`（或 `~/.codebuddy/models.json`） | `url` **必须**以 `/v1/chat/completions` 结尾（而 UI 界面填 `/v1`）；`apiKey` 不可为空；`maxContextTokens` 必须 ≥ `32768` |
| [`qoder-openai-compatible.json`](./qoder-openai-compatible.json) | **Qoder IDE** | 打开 `Qoder` → 设置 → 模型 → 添加自定义模型（`OpenAI Compatible`）对照填入 | `Base URL` 填到 `/v1` 为止，填完点击「校验并添加模型」 |
| [`codex-config.toml`](./codex-config.toml) | **Codex CLI** | 复制至 `~/.codex/config.toml`，并在终端设置 `export ROCM_LOCAL_API_KEY="rocm-local"` | 本地引擎推荐显式设置 `wire_api = "chat"` |

## 服务端启动必带参数速查（防空白回复 / 防工具调用报错）

- **`llama-server`**：`./llama-server -m <model.gguf> -ngl 99 --ctx-size 32768 --jinja --port 8080`
- **`lemonade-server`**：`lemonade load Gemma-4-E4B-it-GGUF --llamacpp vulkan --ctx-size 32768`
- **`vLLM`**：`vllm serve Qwen/Qwen3.5-4B --served-model-name Qwen3.5-4B --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes --port 8000`
