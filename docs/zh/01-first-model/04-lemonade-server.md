# 1.4 标准引擎二：`lemonade-server` AMD AI PC 原生双擎部署实战（端口 `13305`）

> **本节定位（Ryzen AI 笔记本 & Strix Halo 旗舰主机绝配！）**：由 AMD 生态团队开源打造的 **[`lemonade-sdk/lemonade`](https://github.com/lemonade-sdk/lemonade)**，是目前唯一能用一套优雅的命令行与托盘界面**同时统一调度 Radeon GPU（`llama.cpp` 后端）与 Ryzen AI XDNA 2 NPU（`FastFlowLM` 超低功耗后端）** 的原生服务管家！

如果你使用的是 **AMD Ryzen AI 300 / 400 系列轻薄本**（如 HX 370）、**Ryzen AI Max+ 395（Strix Halo 64G/128G 统一内存工作站）**，或者装有 Radeon RX 7000 / 9000 显卡的 Windows / Ubuntu 电脑，`lemonade-server` 会给你带来两大独家惊喜：
1. **原生内置魔搭（ModelScope）拉取通道**：一条 `lemonade-server pull --source modelscope` 命令自动下载并注册模型，免手动找文件路径；
2. **GPU + NPU 异构自由切换**：插电攻坚时调用强劲的 Radeon GPU 跑 `Qwen3.6-35B-A3B` 或 `Qwen3.5-9B`；出门不带充电器时，切到功耗仅几瓦的 **XDNA 2 NPU** 跑 `Gemma 4 E4B` 后台值守，风扇安静如水！

---

## Step 1：3 分钟安装 `lemonade-server`

### 🪟 Windows 11 安装步骤（图形安装包一键搞定）

1. 访问 [`lemonade-sdk/lemonade` 官方 Releases 页面](https://github.com/lemonade-sdk/lemonade/releases)，下载最新版本的 **`Lemonade_Server_Installer.msi`** 双击安装；
2. 安装完成后，右下角系统托盘会出现一个可爱的 🍋 柠檬图标，同时在 PowerShell 中直接输入 `lemonade-server --version` 即可验证（推荐版本 `v9.x / v11.x+`）。

### 🐧 Ubuntu 24.04 安装步骤（官方 PPA 软件源）

```bash
# 1. 添加 Lemonade 官方稳定版 PPA 并安装
sudo add-apt-repository -y ppa:lemonade-team/stable
sudo apt update
sudo apt install -y lemonade-server

# 2. 验证安装版本与本机硬件检测报告
lemonade-server --version
```

> [!TIP]
> **💡 锐龙 AI 300 / Max+ 395 用户开启超大共享显存（GTT）与 NPU 小贴士**：
> - **大统一内存释放（必看！）**：如果你在 Ubuntu 24.04 上用 64GB/128GB 内存的 **Ryzen AI Max+ 395** 跑 35B 大模型，BIOS 里把 `UMA Frame Buffer Size` 设为 `Auto`（或按需分配），Linux 默认允许核显通过 GTT 动态调用一半系统内存；如需解锁高达 96GB+ 给大模型使用，可在 `/etc/default/grub` 的 `GRUB_CMDLINE_LINUX_DEFAULT` 中加入 `ttm.pages_limit=25165824`（对应 96GB 页上限）并 `sudo update-grub && sudo reboot`！
> - **XDNA 2 NPU 检查**：若你想在锐龙 AI 300 / Max+ 395 上启用 NPU 推理，确认已安装 AMD XDNA 驱动（可通过 `xrt-smi examine` 看到 `NPU Phoenix/Strix` 设备正常在线）。

---

## Step 2：一条命令从魔搭拉取并启动本地服务（端口 `13305`）

`lemonade-server` 默认监听 **`13305`** 端口（部分早期版本默认 `8000`，本书统一显式加上 `--port 13305`，以便和 `llama-server :8080`、`vLLM :8000` 互不冲突、并排对比！）。

请根据你的电脑档位任选对应配方运行：

### 配方 A：🟢 轻薄本与入门独显档 —— 启动 `Gemma 4 E4B`

```bash
# 1. Radeon GPU 高速模式（Windows / Ubuntu 通用，兼容所有独显与核显）
lemonade-server run Gemma-4-E4B-it-GGUF \
  --llamacpp vulkan \
  --ctx-size 32768 \
  --port 13305
```

*(⚡ **锐龙 AI XDNA 2 NPU 省电彩蛋**：如果你的笔记本带有 XDNA 2 NPU 并安装了 FastFlowLM 后端组件，只需将后端切为 `--recipe flm` 拉起 `gemma4-it:e4b`，即可在不占一滴 GPU 显存的情况下让 NPU 独立完成流畅问答！)*

### 配方 B：🔵 16G～24G 甜点独显档 —— 启动 `Qwen3.5-9B` 或 `Gemma 4 12B`

```bash
# 启动中文办公主力 Qwen3.5-9B（若首次使用未缓存，可先执行 lemonade-server pull Qwen3.5-9B-GGUF --source modelscope）
lemonade-server run Qwen3.5-9B-GGUF \
  --llamacpp vulkan \
  --ctx-size 32768 \
  --port 13305
```

### 配方 C：🟣 旗舰统一内存 AI PC（Ryzen AI Max+ 395）—— 启动王牌 `Qwen3.6-35B-A3B`！

在拥有 64G～128G 高带宽统一内存的 Strix Halo 主机上，用 `lemonade-server` 拉起 **350 亿总参数、30 亿激活参数的 `Qwen3.6-35B-A3B`** 简直是一种享受：

```bash
# 1. 从魔搭高速拉取 Qwen3.6-35B-A3B GGUF 权重
lemonade-server pull Qwen3.6-35B-A3B-GGUF --source modelscope

# 2. 在 13305 端口满血拉起 32K 上下文服务
lemonade-server run Qwen3.6-35B-A3B-GGUF \
  --llamacpp vulkan \
  --ctx-size 32768 \
  --port 13305
```

---

## Step 3：双通道验收测试（Web 控制台 + 标准 `/v1` API）

### 1. 浏览器打开柠檬控制台

在浏览器访问 👉 **`http://127.0.0.1:13305`**，你可以直接在内置可视化面板中管理已下载的模型库、切换活跃模型并在线聊天测试。

### 2. 终端验证 `/v1/models` 与 `/v1/chat/completions` 接口

> [!NOTE]
> **路径小贴士**：`lemonade-server` 同时兼容 `/api/v1/...` 与标准 `/v1/...` 路由！为了让第 2 章的 WorkBuddy、`Qoder` 和 Codex 配置保持全书高度统一，我们直接访问 **`http://127.0.0.1:13305/v1`**（先用 `/v1/models` 查一下它注册的精确模型 ID）：

```bash
# 1. 查看当前已加载模型的精确 ID 名称
curl http://127.0.0.1:13305/v1/models

# 2. 发送首条标准 OpenAI 对话请求（将 model 替换为上一步查到的 id，如 Gemma-4-E4B-it-GGUF）
curl http://127.0.0.1:13305/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-rocm" \
  -d '{
    "model": "Gemma-4-E4B-it-GGUF",
    "messages": [
      {"role": "user", "content": "你好 Lemonade！请用一句话说明 MoE 混合专家模型为什么又聪明又快？"}
    ]
  }'
```

---

## 🚑 `lemonade-server` 与 NPU 常见排错清单

| 常见现象 | 原因定位 | 快速解决动作 |
|:---|:---|:---|
| Windows 下提示 `13305` 端口已被占用 | 系统右下角托盘里的 🍋 Lemonade 后台服务已经在自动运行了！ | 无需重复启动！直接右键右下角柠檬图标打开控制台，或先执行 `lemonade-server stop` 再带自定义参数重启。 |
| 拉取模型速度慢或连接 HuggingFace 超时 | 漏加了魔搭源参数 | 先显式运行 `lemonade-server pull <模型名> --source modelscope` 走阿里云国内 CDN 满速拉取。 |
| 锐龙 AI 本切换 NPU（`flm`）时报找不到设备 | BIOS 中关闭了 IPU/NPU，或驱动版本低于 NPU MCDM 要求 | 在 Windows 任务管理器确认能看到 `NPU 0`，前往 AMD 官网更新最新 Ryzen AI NPU 驱动后重启。 |

---

## 👉 下一步怎么走？

- 🎉 **已经点亮 `13305` 端口的同学**：你可以立刻带着热乎的服务直冲 **[第 2 章｜零代码办公先行：WorkBuddy 大师 PPT 与自动填表](../02-connect-tools/index.md)**（本仓库 [`src/configs/workbuddy-models.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/workbuddy-models.json) 已经为你预置好了 `13305` 端口配置）！
- 🔬 **想了解工业级高并发推理引擎 `vLLM`，并领取本章《三大引擎选型对比表》与《测速记录卡》的同学**：请继续阅读本章收官节 ➡️ **[1.5 标准引擎三：`vLLM` 工业级高并发部署、三大引擎横评与 `self-llm` 导流（端口 `8000`）](./05-vllm-server.md)**！
