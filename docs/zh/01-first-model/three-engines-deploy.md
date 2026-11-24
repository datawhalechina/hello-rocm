# 1.3～1.5 三大标准引擎极速部署：`llama-server`、`lemonade-server` 与 `vLLM`

为了彻底消除新手“推理工具太多不知道装哪个”的选择困难，全书统一仅使用以下 **三大标准推理服务引擎** 拉起本地 OpenAI 兼容服务（`/v1`）：

| 服务引擎 | 默认本地接口地址 | 最佳适用硬件与场景 | 上手难度 |
|:---|:---|:---|:---:|
| **1. `llama-server` (`llama.cpp`)** | `http://127.0.0.1:8080/v1` | **全系通吃首选！** 单文件免装环境、最省显存，Windows / Ubuntu 独显核显即下即跑 | ⭐ 最简单 |
| **2. `lemonade-server`** | `http://127.0.0.1:13305/v1` | **锐龙 Ryzen AI 笔记本 / AI PC 神器！** 一条命令管理模型，原生支持 Radeon GPU + XDNA 2 NPU | ⭐ 最省心 |
| **3. `vLLM` (ROCm 版)** | `http://127.0.0.1:8000/v1` | **16G+ 独显 / 云端高吞吐首选！** 批量翻论文、并发评测与第 6 章 `AITER` 算子加速最强底座 | ⭐⭐ 标准级 |

> 💡 **你只需任选下面其中 1 个引擎跑通即可直接进入第 2 章**！所有客户端配置均已在 `src/configs/` 为三大引擎全部配好。

---

## 1.3 引擎一：用 `llama-server`（`llama.cpp`）5 分钟拉起本地 GGUF 服务

### Step 1：获取预构建可执行文件与 GGUF 模型

1. 前往 [lemonade-sdk/llamacpp-rocm Releases](https://github.com/lemonade-sdk/llamacpp-rocm/releases) 或 [ggml-org/llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载对应你系统（Windows x64 / Ubuntu x64）的 ROCm / Vulkan 预编译压缩包并解压。
2. 从魔搭社区（ModelScope，国内满速免代理）下载一个 `Q4_K_M` 量化权重（以 `Qwen3.5-4B` 为例）：
   ```bash
   uv pip install modelscope
   modelscope download --model Qwen/Qwen3.5-4B-GGUF --include "*q4_k_m*.gguf" --local_dir ~/ai_models/qwen3.5
   ```

### Step 2：带齐「Agent 防截断参数」启动服务

> ⚠️ **关键避坑**：后续接入 WorkBuddy、`Qoder`、Codex 时，客户端会发送上万字的工具调用定义。启动时**务必加上 `--ctx-size 32768 --jinja`**！

```bash
# Ubuntu 示例（Windows 在 PowerShell 中将 ./llama-server 换为 .\llama-server.exe 即可）
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

./llama-server \
  -m ~/ai_models/qwen3.5/qwen3.5-4b-q4_k_m.gguf \
  --alias qwen3.5-4b-q4_k_m \
  -ngl 99 \
  --ctx-size 32768 \
  --jinja \
  --port 8080
```

---

## 1.4 引擎二：用 `lemonade-server` 一键拉起 GPU / NPU 服务

如果你使用的是 **Ryzen AI 300 / 400 / Max+ 395** 笔记本或迷你主机，AMD 生态原生的 [Lemonade](https://github.com/lemonade-sdk/lemonade) 可以直接从魔搭拉取模型并在 `13305` 端口暴露服务（详见 [Lemonade 环境准备](../00-environment/lemonade.md)）：

```bash
# 1. 检查后端并从国内魔搭源拉取模型
lemonade backends
lemonade pull Gemma-4-E4B-it-GGUF --source modelscope

# 2. 加载模型并在后台启动 13305 端口服务（支持 --llamacpp vulkan / rocm，或 NPU flm 后端）
lemonade load Gemma-4-E4B-it-GGUF --llamacpp vulkan --ctx-size 32768
lemonade status
```

此时服务同时兼容 `http://127.0.0.1:13305/v1` 与 `http://127.0.0.1:13305/api/v1`（默认 API Key 为 `lemonade`）。

---

## 1.5 引擎三：用 `vLLM` 拉起高吞吐并发推理服务

如果你拥有 16G+ 显存的 Radeon 显卡（如 RX 7800 XT / 7900 / 9070 XT）或使用云端算力，推荐使用官方 ROCm Docker 镜像启动 `vLLM`，并开启自动工具调用解析：

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add=video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v ~/ai_models:/app/models \
  -p 8000:8000 \
  vllm/vllm-openai-rocm:latest \
  --model Qwen/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
```

---

## 🎯 本章实战验收：10 秒冒烟测试与《初测卡》

新开一个终端窗口，对你刚刚启动的端口（`8080`、`13305` 或 `8000`）发送一条标准验证请求：

```bash
# 将 PORT 替换为你启动的引擎端口（8080 / 13305 / 8000），MODEL 替换为对应模型名
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer rocm-local" \
  -d '{
    "model": "qwen3.5-4b-q4_k_m",
    "messages": [{"role": "user", "content": "请用一句话向我打招呼，并告诉我你已准备就绪！"}]
  }'
```

收到正常的中文 JSON 回复后，恭喜你完成了第 1 站破冰！请随手记下你的第一份实测数据（第 7 章写简历时会直接用到）：

- **我的显卡 / 档位**：`__________________`
- **选用的推理引擎与模型**：`__________________`
- **空载与满载显存水位**：`____ GB -> ____ GB`

---

## 🔗 延伸阅读与专精课程导流

- **仓库多模型细化参考**：想查看 Qwen3 / Qwen3.6 / Gemma 4 / MiniCPM-V / MiniCPM-o 在不同引擎下的专项命令，可查阅 **[大模型部署档案库](../01-deploy/index.md)**。
- **🚀 专精课程进阶导流**：想系统学习 **100+ 款国内外主流开源大模型** 的部署与全套生态玩法，推荐收藏并进阶学习 Datawhale **[`self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)**！

👉 **下一章（第一步重头戏）**：马上把刚刚跑起来的本地服务接进 **[第 2 章｜WorkBuddy 办公三件套（AI PPT + 自动填表 + 定时任务）、Qoder 与 Codex](../02-connect-tools/index.md)**！
