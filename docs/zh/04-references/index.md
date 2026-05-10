<div align=center>
  <h1>04-References</h1>
  <strong>📚 ROCm 优质参考资料</strong>
</div>

<div align="center">

*精选的 AMD 官方与社区资源*

[返回主页](/zh/)

</div>

## 简介

&emsp;&emsp;本模块收集整理了 ROCm 和 AMD GPU 相关的优质学习资源，包括官方文档、社区教程、技术博客和相关新闻。帮助你快速找到所需的参考资料。

## hello-rocm Skill

hello-rocm Skill 是本项目内置的 AI 助手导航能力。它会把本项目的学习路径、Reference 索引、GPU 架构表、部署教程与排障清单提供给支持 Skills、Rules 或 Agent 配置的 AI 编程工具使用。

| 你想问 | Skill 会索引 |
|------|-------------|
| 我的 GPU 属于什么架构、对应哪个 gfx？ | `docs/zh/00-environment/rocm-gpu-architecture-table.md` |
| 我想最快跑通第一个模型 | `src/hello-rocm-skill/references/quick-deploy/SKILL.md` |
| PyTorch / vLLM / Ollama / llama.cpp 在 ROCm 上怎么装？ | 本页“框架与推理服务” |
| ROCm / PyTorch / HIP 报错怎么排？ | `src/hello-rocm-skill/references/troubleshooting/SKILL.md` |
| 该从哪个章节开始学习？ | README 与各章节 `index.md` |

### 一键复制使用提示

将下面这句话复制给你的 AI 编程工具，让它根据自身支持的 Skills、Rules 或 Agent 配置方式自动判断如何加载：

```text
请使用当前仓库的 src/hello-rocm-skill 作为 hello-rocm Skill；如果你的工具支持 Skills、Rules 或 Agent 配置，请把它安装或加载到合适位置（例如 .claude/skills、.cursor/skills 或 .agents/skills），然后根据该 Skill 帮我学习、部署和排查 AMD ROCm。
```

如果你想手动安装，也可以按工具复制到对应目录：

::: code-group

```bash [Claude Code]
mkdir -p .claude/skills
cp -r src/hello-rocm-skill .claude/skills/hello-rocm
```

```bash [Cursor]
mkdir -p .cursor/skills
cp -r src/hello-rocm-skill .cursor/skills/hello-rocm
```

```bash [通用 Agent]
mkdir -p .agents/skills
cp -r src/hello-rocm-skill .agents/skills/hello-rocm
```

:::

安装或加载后，新开一个会话并尝试：

```text
请加载 hello-rocm skill，帮我判断我的 AMD GPU 应该从哪个 ROCm 教程开始。
```

如果遇到故障排查与常见问题，也可以加入 [飞书社区讨论](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)。

## 官方资源

### AMD 官方文档

| 资源 | 描述 | 链接 |
|------|------|------|
| ROCm 文档 | ROCm 平台官方文档 | [rocm.docs.amd.com](https://rocm.docs.amd.com/) |
| ROCm Release Notes | 版本发布说明 | [Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) |
| HIP 编程指南 | HIP API 和编程指南 | [HIP Docs](https://rocm.docs.amd.com/projects/HIP/en/latest/) |
| AMD GitHub | AMD 开源项目仓库 | [github.com/amd](https://github.com/amd) |
| ROCm GitHub | ROCm 项目仓库 | [github.com/ROCm](https://github.com/ROCm) |

### AMD GPU 架构白皮书

| 架构 | 适用方向 | 官方资料 |
|------|----------|----------|
| AMD CDNA 架构 | 数据中心 GPU 与 AI/HPC 加速 | [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html) |
| AMD CDNA 2 架构 | Instinct MI200 系列与矩阵计算加速 | [AMD CDNA 2 Architecture](https://www.amd.com/en/technologies/cdna-2.html) |
| AMD CDNA 3 架构 | Instinct MI300 系列与生成式 AI/HPC | [AMD CDNA 3 Architecture](https://www.amd.com/en/technologies/cdna-3.html) |
| AMD CDNA 4 架构 | Instinct MI350 系列与新一代 AI 加速 | [AMD CDNA 4 Architecture](https://www.amd.com/en/technologies/cdna-4.html) |
| AMD RDNA 架构 | Radeon 图形与游戏 GPU | [AMD RDNA Architecture](https://www.amd.com/en/technologies/rdna.html) |
| AMD RDNA 2 架构 | Radeon RX 6000 系列与 Infinity Cache | [AMD RDNA 2 Architecture](https://www.amd.com/en/technologies/rdna-2.html) |
| AMD RDNA 3 架构 | Radeon RX 7000 系列与 chiplet GPU | [AMD RDNA 3 Architecture](https://www.amd.com/en/technologies/rdna-3.html) |
| AMD RDNA 4 架构 | Radeon RX 9000 系列与新一代图形/AI 能力 | [AMD RDNA 4 Architecture](https://www.amd.com/en/technologies/rdna-4.html) |

### 框架与推理服务（ROCm 快速安装入口）

> 本节面向 hello-rocm Skill 的快速查阅场景：优先给出框架官方或 AMD ROCm 官方安装入口，并附 AMD ROCm Blog 作为实践案例与版本动态的互相印证。

| 类型 | 项目 | ROCm 快速安装 / 官方说明 | AMD 官方实践参考 | 本项目入口 |
|------|------|--------------------------|------------------|------------|
| 深度学习框架 | PyTorch | [Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html) | [AMD ROCm Blog - PyTorch](https://rocm.blogs.amd.com/search.html?q=PyTorch) | [环境安装](../00-environment/index.md) |
| 深度学习框架 | TensorFlow | [Install TensorFlow for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html) | [AMD ROCm Blog - TensorFlow](https://rocm.blogs.amd.com/search.html?q=TensorFlow) | [环境安装](../00-environment/index.md) |
| 深度学习框架 | JAX | [Install JAX for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html) | [AMD ROCm Blog - JAX](https://rocm.blogs.amd.com/search.html?q=JAX) | [环境安装](../00-environment/index.md) |
| 推理服务 | vLLM | [vLLM AMD ROCm installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#amd-rocm) | [AMD ROCm Blog - vLLM](https://rocm.blogs.amd.com/search.html?q=vLLM) | [vLLM 部署教程](../01-deploy/index.md) |
| 推理服务 | Ollama | [Ollama GPU docs](https://github.com/ollama/ollama/blob/main/docs/gpu.md) | [AMD ROCm Blog - Ollama](https://rocm.blogs.amd.com/search.html?q=Ollama) | [Ollama 部署教程](../01-deploy/index.md) |
| 推理服务 | llama.cpp | [llama.cpp build docs - HIP/ROCm](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) | [AMD ROCm Blog - llama.cpp](https://rocm.blogs.amd.com/search.html?q=llama.cpp) | [llama.cpp 部署教程](../01-deploy/index.md) |
| 推理服务 | LM Studio | [LM Studio GPU docs](https://lmstudio.ai/docs/app/advanced/gpu) | [AMD ROCm Blog - LM Studio](https://rocm.blogs.amd.com/search.html?q=LM%20Studio) | [LM Studio 部署教程](../01-deploy/index.md) |
| 推理运行时 | ONNX Runtime | [Install ONNX Runtime for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/onnxruntime-install.html) | [AMD ROCm Blog - ONNX Runtime](https://rocm.blogs.amd.com/search.html?q=ONNX%20Runtime) | [环境安装](../00-environment/index.md) |

### 库文档

| 库名称 | 用途 | 文档链接 |
|--------|------|----------|
| rocBLAS | 基础线性代数库 | [rocBLAS Docs](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/) |
| MIOpen | 深度学习原语库 | [MIOpen Docs](https://rocm.docs.amd.com/projects/MIOpen/en/latest/) |
| RCCL | 集合通信库 | [RCCL Docs](https://rocm.docs.amd.com/projects/rccl/en/latest/) |
| rocFFT | 快速傅里叶变换 | [rocFFT Docs](https://rocm.docs.amd.com/projects/rocFFT/en/latest/) |
| rocSPARSE | 稀疏矩阵库 | [rocSPARSE Docs](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/) |

## 社区资源

### 教程与博客

- [AMD ROCm Blog](https://rocm.blogs.amd.com/) - AMD 官方技术博客
- [AMD Developer](https://developer.amd.com/) - AMD 开发者资源中心
- [Datawhale](https://github.com/datawhalechina) - 开源学习社区

### 视频教程

> 持续更新中...

### 论坛与社区

| 平台 | 描述 | 链接 |
|------|------|------|
| AMD Community | AMD 官方社区论坛 | [community.amd.com](https://community.amd.com/) |
| GitHub Discussions | ROCm 项目讨论区 | [ROCm Discussions](https://github.com/ROCm/ROCm/discussions) |
| Reddit r/Amd | AMD 相关讨论 | [r/Amd](https://www.reddit.com/r/Amd/) |

## 相关新闻

### 2026

- **2026.03.11** - [ROCm 7.12.0 Preview Release Notes](https://rocm.docs.amd.com/en/7.12.0-preview/about/release-notes.html)
  - 更新 ROCm 7.12.0 预览版发布说明，覆盖 ROCm 组件、安装方式与平台支持变化
  - 兼容性信息以 [ROCm 7.12.0 Compatibility Matrix](https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html?fam=instinct&gpu=mi355x&os=ubuntu&os-version=11_25h2&i=pip) 为准
  - pip 安装索引按 GPU 架构拆分，便于在虚拟环境中选择对应 wheel 源

### 2025

- **2025.12.11** - [ROCm 7.10.0 发布](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)
  - 支持 Windows 平台
  - 支持 pip 安装到 Python 虚拟环境
  - TheRock 项目重构底层架构

> 更多新闻持续更新中...

## 硬件支持

### 支持的 GPU 列表

#### Instinct 系列（数据中心）

| 系列 | 型号 | LLVM Target | ROCm 支持 |
|------|------|-------------|-----------|
| MI350 | MI355X, MI350X | `gfx950` | ✅ |
| MI300 | MI325X, MI300X, MI300A | `gfx942` | ✅ |
| MI200 | MI250X, MI250, MI210 | `gfx90a` | ✅ |
| MI100 | MI100 | `gfx908` | ✅ |

#### Radeon PRO 系列（工作站）

| 系列 | 型号 | LLVM Target | ROCm 支持 |
|------|------|-------------|-----------|
| AI PRO R9000 | R9700, R9600D | `gfx1201` | ✅ |
| PRO W7000 | W7900 Dual Slot, W7900, W7800 48GB, W7800 | `gfx1100` | ✅ |
| PRO W7700 | W7700, V710 | `gfx1101` | ✅ |

#### Radeon RX 系列（消费级）

| 系列 | 型号 | LLVM Target | ROCm 支持 |
|------|------|-------------|-----------|
| RX 9000 | RX 9070 XT, 9070 GRE, 9070 | `gfx1201` | ✅ |
| RX 9000 | RX 9060 XT LP, 9060 XT, 9060 | `gfx1200` | ✅ |
| RX 7000 | RX 7900 XTX, 7900 XT, 7900 GRE | `gfx1100` | ✅ |
| RX 7000 | RX 7800 XT, 7700 XT, 7700 XE, 7700 | `gfx1101` | ✅ |
| RX 7000 | RX 7600 | `gfx1102` | ✅ |

#### Ryzen APU 系列（笔记本/移动端）

| 系列 | 型号 | LLVM Target | ROCm 支持 |
|------|------|-------------|-----------|
| Ryzen AI Max PRO 300 | AI Max+ PRO 395, Max PRO 390/385/380 | `gfx1151` | ✅ |
| Ryzen AI Max 300 | AI Max+ 395, Max 390, Max 385 | `gfx1151` | ✅ |
| Ryzen AI PRO 400 | AI 9 HX PRO 475/470, AI 9 PRO 465, AI 7 PRO 450, AI 5 PRO 440/435 | `gfx1150` | ✅ |
| Ryzen AI 300 | AI 9 HX 375/370, AI 9 365 | `gfx1150` | ✅ |
| Ryzen 200 | 9 270, 7 260/250, 5 240/230/220, 3 210 | `gfx1103` | ✅ |

> 完整支持列表请以 [ROCm 7.12.0 Compatibility Matrix](https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html?fam=instinct&gpu=mi355x&os=ubuntu&os-version=11_25h2&i=pip) 为准。

## 常用工具

### 开发工具

| 工具 | 用途 | 安装命令 |
|------|------|----------|
| hipcc | HIP 编译器 | `sudo apt install hip-dev` |
| rocprof | 性能分析工具 | `sudo apt install rocprofiler` |
| rocgdb | GPU 调试器 | `sudo apt install rocgdb` |
| hipify-clang | CUDA 到 HIP 转换 | `sudo apt install hipify-clang` |

### AI 框架

| 框架 | ROCm 支持 | 安装方式 |
|------|-----------|----------|
| PyTorch | ✅ | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| TensorFlow | ✅ | 参考官方文档 |
| JAX | ✅ | 参考官方文档 |
| ONNX Runtime | ✅ | 参考官方文档 |

## 书籍推荐

> 持续更新中...

## 贡献资源

&emsp;&emsp;如果你有优质的 ROCm 相关资源想要分享，欢迎提交 PR 或 Issue！

### 提交要求

- 资源链接有效且内容优质
- 提供简短的资源描述
- 按照现有分类整理

---

<div align="center">

**欢迎分享更多优质资源！** 🎉

[提交 Issue](https://github.com/datawhalechina/hello-rocm/issues) | [提交 PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
