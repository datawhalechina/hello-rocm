# 1.3 标准引擎一：`llama-server`（`llama.cpp`）轻量全能部署实战（端口 `8080`）

> **本节定位（全平台首选推荐！）**：无需安装庞大的 Docker 虚拟机，无需复杂的 Python 依赖——只需一个几十 MB 的独立可执行程序 + 一个 `.gguf` 量化模型文件，就能在 **Windows 11 和 Ubuntu 24.04 的任何 AMD 独显/核显** 上跑起带可视化聊天网页和标准 OpenAI `/v1` 接口的满血服务！

开源社区传奇项目 **[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)** 内置的官方服务端程序 **`llama-server`** 是个人电脑日常运行本地 AI 助手最稳、最省显存的工业级神器：
- **双后端保险**：既有针对 AMD RDNA 3 / 3.5 / 4 深度加速的 **ROCm HIP 后端**，也有免装任何开发包、插卡即跑的 **Vulkan 跨平台显卡加速后端**；
- **Agent 满血就绪**：只要加上 `--jinja` 参数，就能原生解析 `Gemma 4` 和 `Qwen3.5 / 3.6` 的工具调用模板，完美驱动第 2 章的 WorkBuddy 大师 PPT 与 `Qoder`！

---

## Step 1：3 分钟获取 `llama-server` 程序

请根据你的系统任选最省心的方式获取预编译好的二进制包（无需自己花半小时从源码编译！）：

### 🪟 Windows 11 用户（解压即用！）

1. 打开浏览器访问 [`llama.cpp` 官方 Releases 页面](https://github.com/ggml-org/llama.cpp/releases)；
2. 在最新版本的 `Assets` 列表中按显卡选择一种下载（两个包里都自带 `llama-server.exe`）：
   - **懒人通吃最稳版（强烈推荐新手）**：下载 **`llama-bXXXX-bin-win-vulkan-x64.zip`**（约 30 MB，只要装了 AMD Adrenalin 显卡驱动，所有 RX 6000 / 7000 / 9000 独显与锐龙核显零配置秒跑！）；
   - **ROCm HIP 原生版**：下载对应的 **`llama-bXXXX-bin-win-hip-radeon-x64.zip`**；
3. 将压缩包解压到纯英文目录，例如 **`D:\ai_workspace\llama-cpp\`**。

### 🐧 Ubuntu 24.04 用户（预编译包或官方 ROCm 容器二选一）

**方式 A（推荐，直接解压预编译 Vulkan / Linux 包）**：

```bash
mkdir -p ~/ai_workspace/llama-cpp && cd ~/ai_workspace/llama-cpp
# 下载官方最新 Ubuntu Vulkan x64 预编译包（自动获取最新 Release 链接）
LATEST_URL=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
  | grep "browser_download_url.*bin-ubuntu-vulkan-x64.tar.gz" | cut -d '"' -f 4 | head -n 1)
curl -L "$LATEST_URL" -o llama-vulkan.tar.gz
tar -xzf llama-vulkan.tar.gz
# 验证版本并确认识别到 AMD Radeon GPU
./build/bin/llama-server --version
```

*(补充方式 B：如果你在 Linux 上追求极致 ROCm HIP 原生编译或使用官方容器，也可以直接拉取 AMD 官方维护的 `docker pull rocm/llama.cpp:llama.cpp-b4855_rocm6.3_ubuntu22.04_server` 或执行 `cmake -B build -DGGML_HIP=ON && cmake --build build --config Release -j $(nproc)`。)*

---

## Step 2：从魔搭（ModelScope）国内满速下载主体模型 GGUF

激活我们在 `1.2` 节建好的 `.venv` 虚拟环境，按你在 [0.3 节](../00-guide/03-hardware-and-cloud.md) 选定的档位复制**其中一条**命令即可满速拉取 `Q4_K_M` 黄金量化权重：

```bash
# 🟢 档位一（8G～12G 入门独显 / 核显本 / 全员首次验证首选）：Gemma 4 E4B (~2.9 GB)
modelscope download --model unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q4_K_M.gguf --local_dir ./models/gemma-4-e4b

# 🔵 档位二（16G～24G 甜点独显中文办公王牌）：Qwen3.5-9B (~5.6 GB)
modelscope download --model unsloth/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-Q4_K_M.gguf --local_dir ./models/qwen3.5-9b

# 🔵 档位二（16G～24G 甜点独显代码与逻辑王牌）：Gemma 4 12B (~7.3 GB)
modelscope download --model unsloth/gemma-4-12B-it-GGUF \
  gemma-4-12B-it-Q4_K_M.gguf --local_dir ./models/gemma-4-12b

# 🟣 档位三（64G～128G 旗舰统一内存 AI PC 如 Max+ 395）：Qwen3.6-35B-A3B (~21.5 GB)
modelscope download --model unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-Q4_K_M.gguf --local_dir ./models/qwen3.6-35b-a3b
```

*(小贴士：Windows PowerShell 同学把行尾的反斜杠 `\` 换成反引号 `` ` `` 或合并成一行运行即可，例如下载到 `D:\ai_workspace\models\gemma-4-e4b`！)*

---

## Step 3：一条命令点火！在 `8080` 端口启动标准 OpenAI 服务

下面以最通用的 **`Gemma 4 E4B`**（若你下了 `Qwen3.5-9B`，只需把 `-m` 后面的文件路径换掉即可）为例，启动服务端：

### 🐧 Ubuntu / Linux 启动命令

```bash
./build/bin/llama-server \
  -m ./models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf \
  --alias gemma-4-e4b \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl 99 \
  --ctx-size 32768 \
  -fa on \
  --jinja \
  --cache-reuse 256
```

### 🪟 Windows PowerShell 启动命令

```powershell
D:\ai_workspace\llama-cpp\llama-server.exe `
  -m D:\ai_workspace\models\gemma-4-e4b\gemma-4-E4B-it-Q4_K_M.gguf `
  --alias gemma-4-e4b `
  --host 127.0.0.1 `
  --port 8080 `
  -ngl 99 `
  --ctx-size 32768 `
  -fa on `
  --jinja `
  --cache-reuse 256
```

### 📖 6 个关键启动参数逐词白话解读（知其然更知其所以然！）

| 参数写法 | 白话翻译与为什么必须加？ |
|:---|:---|
| `--alias gemma-4-e4b` | 给模型起一个干净短名字！这样第 2 章在 WorkBuddy 和 `Qoder` 里填 `Model ID` 时直接写 `gemma-4-e4b`（或 `qwen3.5-9b`），不用抄长长的文件路径。 |
| `--host 127.0.0.1 --port 8080` | **安全铁律**：只监听本机回环地址 `127.0.0.1` 的 `8080` 端口，防止在咖啡厅公共 Wi-Fi 下被陌生人偷连你的电脑。 |
| `-ngl 99` | `n-gpu-layers`：把模型全部网络层 100% 塞进你的 AMD 显卡显存里高速运算（如果不加或设为 0 就会退回慢吞吞的 CPU 跑！）。 |
| `--ctx-size 32768` | **WorkBuddy 生命线参数**！把上下文草稿本开到 `32K`（默认只有 4K，装不下大师 PPT 和办公工具的提示词说明书！）。*8G 显存紧俏时可同时加 `-ctk q8_0 -ctv q8_0` 省显存。* |
| `-fa on` | 开启 **Flash Attention（闪电注意力）**，在长文本阅读时既提速又大幅降低显存峰值。 |
| `--jinja` & `--cache-reuse 256` | `--jinja` 激活模型原厂工具调用（Tool Calling）模板；`--cache-reuse 256` 开启前缀 KV 缓存复用，多轮对话第 2 句起秒回！ |

当终端最后一行停在 **`main: server is listening on http://127.0.0.1:8080 - starting the main loop`** 时，请**不要关闭这个终端窗口**（关掉它就等于拔掉了服务电源）！

---

## Step 4：三连验收！亲眼见证你的本地 AI 开口说话并测出 `tok/s`

保持刚才的服务窗口在后台运行，现在体验三种超爽的调用方式：

### 验收 1：零代码打开浏览器内置聊天室

直接在浏览器地址栏输入 👉 **`http://127.0.0.1:8080`** 回车！  
你会立刻看到 `llama-server` 自带的高颜值极简网页对话框，随便输入一句 *“你好！请用三句话向零基础小白介绍 AMD ROCm 是什么？”*，右下角还会实时显示 **`XX.X tokens/s`** 生成速度！

### 验收 2：新开一个终端，用 `curl` 验证标准 `/v1/chat/completions` 接口

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-rocm" \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [
      {"role": "user", "content": "请用一句俏皮的话庆祝我在 AMD 显卡上成功点亮第一个本地大模型！"}
    ],
    "temperature": 0.7
  }'
```

### 验收 3：15 行 Python 标准 OpenAI SDK 流式测速脚本

运行下面这段短脚本，精确测出你电脑的 **首字响应时间（TTFT）** 与 **每秒吐字速度（`tok/s`）**（这两项数字马上填进你的第一张作品集卡片！）：

```bash
python -c "
import time
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:8080/v1', api_key='local-rocm')
t0 = time.perf_counter()
first_tok_time = None
chars = 0

stream = client.chat.completions.create(
    model='gemma-4-e4b',
    messages=[{'role': 'user', 'content': '请列出在本地电脑部署大模型的 3 个核心优势，每条 30 字以内。'}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ''
    if delta and first_tok_time is None:
        first_tok_time = time.perf_counter()
    print(delta, end='', flush=True)
    chars += len(delta)

total = time.perf_counter() - t0
ttft_ms = ((first_tok_time or time.perf_counter()) - t0) * 1000
print(f'\n\n📊 [本机实测] 首字延迟 TTFT: {ttft_ms:.0f} ms | 总耗时: {total:.2f} s | 服务端精确 tok/s 请看 llama-server 窗口日志！')
"
```

---

## 🚑 `llama-server` 常见小卡点排查

1. **启动日志里写着 `using CPU backend` 而不是 `Vulkan0: AMD Radeon...` 或 `ROCm0`？**  
   说明你下载成了纯 `cpu` 版压缩包，或者忘了加 `-ngl 99`。换回 `vulkan-x64` 或 `hip-radeon-x64` 包瞬间提速 5～10 倍！
2. **报错 `error: failed to bind to 127.0.0.1:8080`？**  
   说明本机已经有另一个软件占用了 `8080` 端口（或者你开了两个 `llama-server` 窗口）。关掉旧窗口，或把 `--port 8080` 改为 `--port 8081` 即可。
3. **加载时报错 `out of memory`（显存不足）？**  
   加上对称 KV 压缩参数 `-ctk q8_0 -ctv q8_0`，或者把 `--ctx-size 32768` 调至 `16384`。

---

## 👉 下一步怎么走？

- 🎉 **如果你已经跑通了 `8080` 端口并且迫不及待想玩 AI 生成 PPT 和自动填表**：你现在就可以直接跳去 **[第 2 章｜零代码办公先行：接入 WorkBuddy 与 `Qoder`](../02-connect-tools/index.md)**！
- 💻 **如果你手头是 Ryzen AI 笔记本或 Ryzen AI Max+ 395 迷你主机**：强烈推荐继续往下看专为 AMD AI PC 打造的原生双擎管家 ➡️ **[1.4 标准引擎二：`lemonade-server` 原生双擎部署实战（端口 `13305`）](./04-lemonade-server.md)**！
