# ROCm 10.0.0 版本说明：从 7.14.0 到 ROCm.AI

<div align="center">

*2026-08-26 · ROCm Core SDK 10.0.0 · [官方文档 latest](https://rocm.docs.amd.com/en/latest/) · [官方 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)*

[返回环境基线](/zh/00-environment/) · [English](/00-environment/rocm-10-0-0-release-notes)

</div>

ROCm 10.0.0 是 7.x 之后的第一次大版本跳跃。7.14.0 把 [TheRock](https://github.com/ROCm/TheRock) 做成了生产级构建与发布基座；10.0.0 则在这块基座上，第一次把「装环境、写代码、调性能」收成一套面向 AI 助手的工作流，名字叫 **ROCm.AI**。

对 hello-rocm 的读者来说，最该先看懂的不是库版本号，而是这三个新入口：

| 组件 | 一句话 | 你什么时候用它 |
|:---|:---|:---|
| **AMD Skills** | 把 AMD 官方优化知识做成 Claude / Cursor / Codex 能直接加载的 Skill | 你用 AI 写 ROCm / HIP / vLLM 代码时，让它按 AMD 最佳实践来 |
| **Hyperloom** | 开源自动优化引擎：剖析、找瓶颈、重写内核、调参数、再验证 | 你已经能跑通推理，想把吞吐再往上推 |
| **ROCm CLI** | 装环境、验证、部署、管理，一条命令走完 | 你不想再手抄 pip / 驱动 / 镜像命令 |

本页先讲清 10.0.0 和 7.14.0 的差别，再把这三个入口拆开看。环境安装步骤见 [00-Environment](/zh/00-environment/)。

---

## 1. 一张表看完 7.14.0 → 10.0.0

| 维度 | ROCm 7.14.0（2026-07-15） | ROCm 10.0.0（2026-08-26） |
|:---|:---|:---|
| 版本定位 | TheRock 正式成为构建 / 发布基座 | 十年节点的大版本；TheRock 之上补齐 **ROCm.AI** |
| 开发体验 | 文档 + wheel + Docker，人自己拼流程 | **AMD Skills + Hyperloom + ROCm CLI** 三件套 |
| pip 索引 | `https://repo.amd.com/rocm/whl-multi-arch/` | `https://stable.repo.amd.com/rocm/whl-next/` |
| 系统包仓库 | 旧 `repo.amd.com` 分发 | 统一到 [stable.repo.amd.com](https://stable.repo.amd.com)（ROCm / 驱动 / 工具同一套结构） |
| PyTorch | 2.12.0 | **2.13.0**（另支持 2.12.0 / 2.11.0） |
| torchvision / torchaudio | 0.27.0 / 2.11.0 | **0.28.0 / 2.11.0.2** |
| JAX | 0.10.0 | **0.11.0**（另支持 0.10.2 / 0.10.0） |
| vLLM | 0.23.0 | **0.27.0** |
| SGLang | 0.5.13 | **0.5.15** |
| TensorFlow | 2.21 | 2.21（另支持 2.20 / 2.19.1） |
| MIGraphX / ONNX Runtime | 2.16 / 1.23.2 一代 | **2.17 / 1.29.0** |
| Windows 驱动 | Adrenalin **26.5.1** | Adrenalin **26.8.1** |
| Linux 驱动 | amdgpu 31.30.0 一带 | amdgpu **31.50.0** |
| 新增硬件 | gfx1153（Ryzen AI 7 445 / AI 5 435 等） | **Radeon RX 9050 / 9050 4GB（gfx1200）** |
| Windows SDK | HIP SDK 与 Linux Core SDK 分轨 | **HIP SDK 退役**，Windows / Linux 共用 ROCm Core SDK |
| ASAN 包 | 7.14.0 暂不提供 | **随标准包一起提供** |
| 推荐安装入口 | `uv pip` + 官方镜像 | 仍可用 `uv pip`；新增 **`rocm install sdk`** |

> 官方总入口请以 [ROCm 10.0.0 文档](https://rocm.docs.amd.com/en/latest/) 为准。兼容性以 [Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) 为准。

---

## 2. 为什么从 7 跳到 10

7.14.0 解决的是「怎么把 ROCm 拆开、按需装、Windows / Linux 同一套 TheRock 流水线」。10.0.0 解决的是下一层问题：

- 装完之后，开发者仍然要自己记 gfx extras、自己选镜像、自己对驱动、自己读优化文档。
- AI 编程助手默认按 CUDA / NVIDIA 习惯写代码，不会自动走 AMD 的 kernel、调度和量化路径。
- 推理优化仍然是「人看 profile → 人改 kernel → 人再测」，周期以周计。

所以 10.0.0 的版本号不是「库又升了一档」，而是把平台从 **给库** 推到 **给工作流**。AMD 官方博客把这一点写得很直白：过去的 ROCm 主要给你更好的原语；ROCm.AI 管的是原语周围的安装、验证、服务和优化。[ROCm 10.0 官方解读](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html)

TheRock 没有被替换。10.0.0 的 Core SDK、框架 wheel、验证镜像都从同一条 TheRock 流水线出来，大约每六周发一个小版本。

---

## 3. 重点一：AMD Skills

### 3.1 它到底是什么

[AMD Skills](https://github.com/amd/skills) 不是又一份 Markdown 文档站。它是一套按 [Agent Skills](https://github.com/anthropics/skills) 标准写好的目录：每个 Skill 里有 `SKILL.md`、可选脚本和参考资料。Claude Code、Cursor、Codex、Gemini CLI 看到描述后，会在任务匹配时把对应 Skill 读进上下文。

文档讲「API 有哪些选项」；Skill 讲「AMD 工程师默认会怎么选」——用哪张镜像、哪个 `gfx`、哪组环境变量、先检查什么再启动服务。你用 AI 写 ROCm 代码时，助手会按这些已经验证过的路径走，而不是临场猜 CUDA 习惯。

安装：

```bash
npx skills add amd/skills
```

指定 Skill 和助手：

```bash
npx skills add amd/skills --skill local-ai-use --agent cursor
npx skills add amd/skills --list
```

也可以手动拷到 `~/.cursor/skills/`、`~/.claude/skills/` 或 `$HOME/.agents/skills`。

### 3.2 目录怎么分层

官方目录按「端侧 / 跨栈 / 服务器」分成三块，正好覆盖 hello-rocm 读者最常见的两条线：Ryzen AI PC 本地跑模型，以及 Instinct 上的生产推理。

**Client-native（Ryzen AI / 本地 AI PC）**

| Skill | 做什么 |
|:---|:---|
| `local-ai-use` | 把生图、TTS、STT 转到本地服务，少烧云端 token |
| `local-ai-app-integration` | 给已有云端 LLM 应用补上离线、隐私和本地推理 |

**Cross-stack（从客户端到云）**

| Skill | 做什么 |
|:---|:---|
| `rocm-doctor` | 对照已知错误配置诊断 ROCm / HIP / PyTorch / llama.cpp，必要时驱动 `rocm examine` / `diagnose` / `fix`（规划中） |
| `lemonade-router-builder` | 按内容、敏感度、能力把请求分到 Lemonade 路由 |
| `hrr-replay-analysis` | 用 HIP Record and Replay 回放并分析 GPU 负载（规划中） |

**Server-native（Instinct / EPYC）**

| Skill | 做什么 |
|:---|:---|
| `serving-llms-on-instinct` | 检测硬件、核对模型显存、套 vLLM recipe、拉起可测吞吐的 endpoint |
| `serving-llms-on-epyc` | 在 EPYC + zentorch / vLLM 上做 CPU 推理 |
| `hyperloom-workload-optimizer` | 安装 Hyperloom，对 Instinct 上的 LLM 推理做端到端自动优化 |
| `magpie-kernel-evaluator` | 评测 kernel 正确性与性能，对照 vLLM / SGLang |
| `tracelens-analysis-orchestrator` | 用 TraceLens 并行拆解 PyTorch profile，输出优先级报告 |

### 3.3 和本仓库 hello-rocm Skill 怎么分工

本仓库的 [`src/hello-rocm-skill`](https://github.com/datawhalechina/hello-rocm/tree/main/src/hello-rocm-skill) 管的是「这个教程项目怎么学、怎么部署、怎么排障」。AMD Skills 管的是「AMD 官方已经验证过的平台最佳实践」。两者可以同时装：

- 问「我该看 hello-rocm 哪一章、gfx 怎么填」→ hello-rocm Skill
- 问「按 AMD 官方路径在 MI300X 上起 vLLM / 用 Hyperloom 榨吞吐」→ AMD Skills

---

## 4. 重点二：Hyperloom

### 4.1 它解决什么问题

[Hyperloom](https://rocm.docs.amd.com/projects/hyperloom/en/latest/) 是开源的 agentic 优化系统（源码：[AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom)）。目标不是再给你一个手动 profiler GUI，而是把「看 trace → 找瓶颈 → 改 kernel / 调参 → 验证正确性和性能」收成闭环。

官方描述的循环是：

```text
Profile → Analyze → Plan → Optimize → Validate → 再来一轮
```

人不再逐轮开车。你给它模型、框架、GPU、并行度和序列长度，它自己做剖析、提案、改代码、跑基准、核对正确性。AMD 的说法是：过去按周计的手工优化，可以被压到按小时，并且会扫到人手时间不够时不会去试的搜索空间。

### 4.2 里面有哪些零件

| 组件 | 角色 |
|:---|:---|
| TraceLens-Agent | 自动从 profile 里标出瓶颈 |
| Magpie | 评测 / 对比 kernel，给 vLLM、SGLang 做基准 |
| IntelliKit | 对话式 profiling |
| GEAK | 多 agent 自动改 kernel，覆盖 Triton、HIP、CK、FlyDSL、TileLang |
| Arbor | 在优化空间里做自演化搜索 |
| AgentKernelArena | 用标准任务给不同优化 agent 做 A/B |

10.0.0 当前公开支持 **MI300X / MI325X / MI355X**，框架侧覆盖 **vLLM 和 SGLang**。端侧 Radeon / Ryzen 上，hello-rocm 仍以先跑通部署为主；Hyperloom 更适合已经有 Instinct 机器、要抠生产吞吐的读者。

### 4.3 怎么接到你的工作流

两种用法：

1. **走 AMD Skills**：装 `hyperloom-workload-optimizer`，直接对助手说「用 Hyperloom 把这个 Instinct 上的推理吞吐再抬一截」。
2. **走官方 CLI**：按 [Run a Hyperloom optimization](https://rocm.docs.amd.com/projects/hyperloom/en/latest/how-to/optimize.html) 启动会话，例如指定框架、模型路径、`TP` / `CONC` / `ISL` / `OSL` 和目标增益。

它不会替代 7.14.0 已经有的 rocprofiler-sdk。7.14.0 把 PyTorch Profiler 后端切到了 rocprofiler-sdk；10.0.0 继续加 rocSHMEM / hipFile / OpenMP 追踪、HIP Graph 逐节点归属，以及 gfx1150 / 1151 / 1152 的 Roofline。Hyperloom 吃的就是这些 trace，再往上做「改代码」这一跳。

---

## 5. 重点三：ROCm CLI

### 5.1 一条命令走完的那一层

[ROCm CLI](https://github.com/ROCm/rocm-cli) 是 ROCm.AI 的命令行面。它是单个预编译二进制，Linux / Windows x86_64 都能装，**不要求机器上已经有 ROCm、Python 或 Rust**。当前是 **Technology Preview**：命令和界面还可能变，但从 7.13 起就能管多版本运行时；10.0 正式支持按官方说明会随后跟上。

Linux / WSL：

```bash
curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh
```

Windows PowerShell：

```powershell
irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
```

日常就这几条：

```bash
rocm                 # 启动器：安装 / 服务 / 诊断 / 聊天 / 仪表盘
rocm examine         # 看 GPU、驱动、ROCm、引擎是否就绪
rocm install sdk     # 把 TheRock wheel 装进 CLI 托管的 Python 环境
rocm install driver  # Linux 上装 amdgpu 驱动
rocm serve qwen      # 拉起本地 OpenAI 兼容推理服务
rocm dash            # 全屏遥测（Linux / WSL）
```

7.14.0 的路径是「人自己 `uv venv` + 填 extras + 对镜像 tag」。10.0.0 多了一条托管路径：`rocm install sdk` 负责下载 TheRock wheel 和配套 PyTorch，`rocm examine` 负责告诉你缺的是驱动、权限还是运行时，`rocm serve` 按 GPU 选引擎——Ryzen / Radeon 走 Lemonade（GGUF），Instinct 走 vLLM（safetensors）。

### 5.2 它比「再写一套安装脚本」多在哪

- **多运行时并存**：`rocm runtimes activate` / `rollback`，同一台机器可以挂多个 ROCm，升级能回滚。
- **引擎适配**：现在是 Lemonade（客户端）和 vLLM（Instinct）；Windows 目前有 CLI + Lemonade，没有 live dashboard / vLLM。
- **给 AI 助手留了口**：`rocm-doctor` 会薄薄包一层 `examine` / `diagnose` / `fix`，Skills 和 CLI 是同一套状态机。
- **隔离旧环境**：机器上如果已经有 7.14.0 的 `/opt/rocm` 或手建 `.venv`，`rocm examine` 会标成 `legacy_rocm_status: detected_unmanaged`，再 `rocm install sdk` 不会直接覆盖，而是在旁边建托管运行时。

最低 Linux 要求是 **Ubuntu 24.04**（glibc 2.38+）。Ubuntu 22.04 跑不了这份预编译 Lemonade。macOS 没有官方包。

hello-rocm 仍然把 `uv pip` 写成可复现基线，方便对照官方 wheel 版本。新机器如果只想先跑起来，可以优先试 ROCm CLI，再用本页和 [环境基线](/zh/00-environment/) 核对版本。

---

## 6. 平台与软件栈还变了什么

### 6.1 安装与分发

- GPU 软件开始往 [stable.repo.amd.com](https://stable.repo.amd.com) 收：ROCm 包、amdgpu 驱动、公开工具同一套布局，多版本、多架构可以并排放。
- pip 主索引从 `repo.amd.com/rocm/whl-multi-arch/` 换成 `stable.repo.amd.com/rocm/whl-next/`。`[device-gfxXXXX]` extras 没变。
- 已安装的 RPM / DEB / runfile 包改为嵌入 **RPATH**（不再是 RUNPATH），二进制会优先解析自己那棵 ROCm 库，减少一台机器多版本时被 `LD_LIBRARY_PATH` 带跑。tarball 仍用 RUNPATH，方便自己拼环境。
- Windows 上 HIP SDK 退役，和 Linux 共用 Core SDK 与发版节奏。当前 Windows 以 tarball 为主，原生安装器要到 2026 年晚些时候。

### 6.2 框架与镜像

本项目 pip 基线对齐官方推荐：

```bash
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" \
  "torchaudio==2.11.0.2+rocm10.0.0"
```

vLLM 官方验证镜像（注意：镜像内 PyTorch 是 2.12.0，和 pip 的 2.13.0 不要混写）：

```bash
docker pull rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
```

另外还有 Ryzen AI MAX 上的 Unsloth 本地微调，以及 ComfyUI 对 Wan2.2、FLUX.2 KLEIN、SD 3.5 等模型的现成优化。这些会逐步补进 02-Fine-tune / 实践案例，不挡 01-Deploy 先升级环境。

### 6.3 通信、数学库、工具

- **RCCL / rocSHMEM**：这是 10.0.0 里投资最大的一块。RCCL 跟上游 NCCL 的合并推进到 2.30.4，补对称内存、GPU 发起网络（GIN）、Python API；rocSHMEM 继续追 NVSHMEM 3.6.5 的 API 缺口。
- **hipBLASLt**：可在本地给自己的 GEMM 形状做 kernel 选择，权重不用离开机房。
- **ROCm Optiq 1.0 GA**：把 Systems Profiler 时间线和 Compute Profiler 分析放进同一个可视化环境。
- **ASAN 包**：7.14.0 缺的 Address Sanitizer 包，10.0.0 可以当普通包来装。

---

## 7. 对 hello-rocm 读者的升级建议

1. **新装机器**：按 [环境基线](/zh/00-environment/) 走 `uv pip` + ROCm 10.0.0；如果只想先验证 GPU，用 `rocm examine` / `rocm install sdk`。
2. **已经在 7.14.0 上跑通的人**：不必为了版本号立刻重装。要新框架（PyTorch 2.13 / vLLM 0.27）或 ROCm.AI 再升。升级前先卸旧 HIP SDK / 旧 wheel，Windows 驱动升到 Adrenalin 26.8.1。
3. **用 AI 写代码的人**：在现有 hello-rocm Skill 之外，再装一份 [amd/skills](https://github.com/amd/skills)。两套可以并存。
4. **有 Instinct、要抠吞吐的人**：把 Hyperloom 当成 10.0.0 的主菜，而不是再手写一轮 kernel 调参笔记。
5. **Radeon / Ryzen 本地部署**：环境命令已经切到 10.0.0；LM Studio / Ollama / llama.cpp 的操作步骤大体不变，变的是底层 ROCm / 驱动版本。

---

## 8. 官方入口

| 资源 | 链接 |
|:---|:---|
| ROCm 10.0.0 文档首页 | <https://rocm.docs.amd.com/en/latest/> |
| Core SDK Release Notes | <https://rocm.docs.amd.com/en/latest/about/release-notes.html> |
| 安装选择器 | <https://rocm.docs.amd.com/en/latest/install/rocm.html> |
| 兼容性矩阵 | <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html> |
| TheRock 迁移指南 | <https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html> |
| 官方解读博客 | <https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html> |
| ROCm.AI 新闻稿 | <https://newsroom.amd.com/news/rocm-10-software-ai-native-developer-experiences/> |
| AMD Skills | <https://github.com/amd/skills> |
| ROCm CLI | <https://github.com/ROCm/rocm-cli> |
| Hyperloom 文档 | <https://rocm.docs.amd.com/projects/hyperloom/en/latest/> |
| Hyperloom 源码 | <https://github.com/AMD-AGI/Hyperloom> |
