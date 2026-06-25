<div align=center>
  <h1>MiniCPM-o 4.5 部署</h1>
  <strong>🎙️ 在 AMD GPU 上运行全模态大模型</strong>
</div>

<div align="center">

*语音 · 视觉 · TTS · 全双工对话 · ROCm 7+ · llama.cpp-omni*

[返回主页](/zh/) · [English](/01-deploy/minicpm-o/) · [返回部署总览](/zh/01-deploy/)

</div>

## 简介

&emsp;&emsp;**MiniCPM-o 4.5** 是面壁智能（OpenBMB）推出的端侧全模态大模型，支持**文本、语音输入 + 语音输出**（TTS）以及**图像理解**，并可通过 Omni 全双工模式实现类电话的实时交互——无需 NVIDIA GPU，在 AMD ROCm 上同样可以完整运行。

&emsp;&emsp;本模块基于 OpenBMB 官方开源的 [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) 推理引擎和 [`OpenBMB/MiniCPM-o-Demo`](https://github.com/OpenBMB/MiniCPM-o-Demo) 演示仓库（`Comni` 分支），在 **AMD Ryzen AI MAX+ 395（gfx1151 / Strix Halo APU）** 上验证通过，同样适用于其他支持 ROCm 的 AMD GPU。

```
01-Deploy/minicpm-o/
├── minicpm-o-model.md               # MiniCPM-o 4.5 模型介绍
├── llamacpp-omni-rocm7-deploy.md    # llama.cpp-omni CLI 推理部署
└── webdemo-rocm7-deploy.md          # MiniCPM-o Web Demo 全双工部署
```

---

## 教程列表

### MiniCPM-o 4.5 模型介绍

&emsp;&emsp;了解 MiniCPM-o 4.5 的全模态架构——语音输入、视觉编码、TTS 语音合成与全双工对话设计。读完后可清楚知道需要哪些 GGUF 子模型文件，以及不同 GPU 的显存要求。

- **适合人群**：希望在部署前了解模型架构的读者
- **难度等级**：⭐
- **预计时间**：15 分钟

📖 [阅读 MiniCPM-o 4.5 模型介绍](./minicpm-o-model.md)

---

### llama.cpp-omni CLI 零基础部署

&emsp;&emsp;使用 llama.cpp-omni 推理引擎，在 AMD GPU 上编译并运行 MiniCPM-o 4.5。本教程覆盖 HIP 编译、GGUF 模型下载、音频输入测试和 TTS 语音输出，让你用命令行跑通全模态推理。

- **适合人群**：希望轻量验证推理能力、或为 Web Demo 准备推理后端的开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：2 小时（含模型下载）

📖 [开始学习 llama.cpp-omni 部署教程](./llamacpp-omni-rocm7-deploy.md)

---

### MiniCPM-o Web Demo 全双工部署

&emsp;&emsp;基于 `OpenBMB/MiniCPM-o-Demo`（`Comni` 分支），在 AMD GPU 上搭建完整的 Web Demo，支持**轮询对话 / 半双工 / Omni 全双工 / 纯语音全双工**四种交互模式，在浏览器中即可与模型语音+视频实时对话。前置条件：已完成 llama.cpp-omni 编译。

- **适合人群**：希望搭建完整交互演示的开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：1 小时（推理后端已就绪的情况下）

📖 [开始学习 Web Demo 全双工部署](./webdemo-rocm7-deploy.md)

---

## 环境要求

### 硬件要求

| 场景 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 文本推理（仅 LLM） | 8 GB 显存 | — |
| 语音输入（音频编码器） | 12 GB 显存 | — |
| 完整全模态（LLM + 视觉 + 音频 + TTS） | **16 GB 显存** | 64 GB 统一内存（Strix Halo APU） |

> AMD Ryzen AI MAX+ 395 / 890M 系列的统一内存架构非常适合 MiniCPM-o——64 GB 统一内存可将全部 GGUF 子模型（~8.3 GB）完整载入 GPU。

### 软件要求

- 操作系统：Linux（Ubuntu 22.04 / 24.04）
- ROCm 7.10.0 或更高版本（系统安装）
- CMake 3.21+，GCC / Clang（用于 HIP 编译）
- Python 3.10+（Web Demo 依赖）

### 本教程测试环境

```
硬件：AMD Ryzen AI MAX+ PRO 395 / Radeon™ 890M（Strix Halo）
架构：gfx1151
统一内存：64 GB（全部可作 VRAM 使用）
ROCm：7.12.0（系统）+ TheRock 7.12.0a alpha（修复 Tensile）
OS：Ubuntu 24.04
```

---

## 常见问题

<details>
<summary>Q: MiniCPM-o 4.5 和普通 LLM 有什么区别，为什么不能直接用 vLLM / Ollama 部署？</summary>

MiniCPM-o 4.5 是全模态模型，包含独立的**语音编码器**、**视觉编码器**、**TTS 语音合成**（token2wav）等多个子模块，这些模块目前尚未被主流推理框架原生支持。llama.cpp-omni 是专门为此模型设计的推理引擎 fork，可以同时加载并调度这些子模型。

</details>

<details>
<summary>Q: gfx1151（Strix Halo）需要特殊处理吗？</summary>

是的。gfx1151 是 2025 年底推出的新架构，系统 `/opt/rocm` 内的 rocBLAS Tensile 库**尚不包含 gfx1151 的完整 GEMM 内核**，会导致推理时崩溃（`hipErrorInvalidImage`）。解决方法是安装 TheRock 7.12.0a alpha SDK 并在运行时指向其 rocBLAS 库目录——详见 [llama.cpp-omni 部署教程](./llamacpp-omni-rocm7-deploy.md)。

其他 AMD GPU（如 gfx1100 / RX 7900 XTX、gfx1150 / RX 9070 XT 等）不受此问题影响，可直接使用系统 rocBLAS。

</details>

<details>
<summary>Q: 如何确认我的 AMD GPU 架构（gfx 编号）？</summary>

```bash
rocminfo | grep -A 5 "Agent [0-9]" | grep -E "Name|gfx"
# 或
amd-smi
```

</details>

---

## 参考资源

- [MiniCPM-o 官方仓库（OpenBMB）](https://github.com/OpenBMB/MiniCPM-o)
- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [MiniCPM-o-Demo 官方仓库](https://github.com/OpenBMB/MiniCPM-o-Demo)
- [MiniCPM-o-4_5-gguf（ModelScope）](https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf)
- [ROCm 官方文档](https://rocm.docs.amd.com/)
- [TheRock Nightly SDK（gfx1151 修复）](https://rocm.nightlies.amd.com/v2/gfx1151/)

---

<div align="center">

**欢迎贡献更多部署教程！** 🎉

[提交 Issue](https://github.com/datawhalechina/hello-rocm/issues) | [提交 PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
