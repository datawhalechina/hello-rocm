# hello-rocm 2.0 已验证配置模板库（三大服务引擎 × 四大分档主体模型）

本目录存放经过验证的本地 OpenAI 兼容接入配置，围绕课程四大分档主体模型 **`Gemma 4 E4B`**（入门档）、**`Qwen3.5-9B`** & **`Gemma 4 12B`**（甜点档）与 **`Qwen3.6-35B-A3B`**（旗舰档），统一覆盖 **`llama-server`（端口 `8080`）**、**`lemonade-server`（端口 `13305`）** 与 **`vLLM`（端口 `8000`）**。

## 文件清单与放置路径

| 配置文件 | 适用客户端 | 本机放置位置 / 使用方法 | 核心避坑点 |
|:---|:---|:---|:---|
| [`workbuddy-models.json`](./workbuddy-models.json) | **WorkBuddy** | 复制至 `~/.workbuddy-ai/models.json`（或 `~/.codebuddy/models.json`） | `url` **必须**以 `/v1/chat/completions` 结尾（而 UI 界面填 `/v1`）；`apiKey` 不可为空；`maxContextTokens` 必须 ≥ `32768` |
| [`qoder-openai-compatible.json`](./qoder-openai-compatible.json) | **Qoder IDE** | 打开 `Qoder` → 设置 → 模型 → 添加自定义模型（`OpenAI Compatible`）对照填入 | `Base URL` 填到 `/v1` 为止，填完点击「校验并添加模型」 |
| [`codex-config.toml`](./codex-config.toml) | **Codex CLI** | 复制至 `~/.codex/config.toml`，并在终端设置 `export ROCM_LOCAL_API_KEY="rocm-local"` | 本地引擎推荐显式设置 `wire_api = "chat"` |

## 四大主体模型服务端启动命令速查

- **🟢 入门档 `llama-server`（`Gemma 4 E4B` GGUF）**：
  `./llama-server -m <gemma-4-E4B-it-Q4_K_M.gguf> --alias gemma-4-E4B-it-Q4_K_M -ngl 99 --ctx-size 32768 --jinja --port 8080`
- **🔵 甜点档 `vLLM`（`Qwen/Qwen3.5-9B` 或 `google/gemma-4-12B-it`）**：
  `vllm serve Qwen/Qwen3.5-9B --served-model-name Qwen3.5-9B --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes --port 8000`
- **🟣 旗舰档 `lemonade-server`（`Qwen3.6-35B-A3B-GGUF`，约 20–23 GiB，适合 64G–128G 统一内存 AI PC）**：
  `lemonade pull Qwen3.6-35B-A3B-GGUF --source modelscope && lemonade load Qwen3.6-35B-A3B-GGUF --llamacpp vulkan --ctx-size 32768`
