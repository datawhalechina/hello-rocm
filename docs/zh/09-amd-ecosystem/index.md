# 第 9 章｜拓展视野：AMD 优秀开源项目、前沿行业案例与开发者资源百宝箱

> **本章定位（收官宝藏字典）**：当你完成了前 8 章从零到一的入门与实战后，本章为你系统汇集了 **AMD 官方最具代表性的 AI 与 Agent 开源项目**、**本仓库沉淀的科研与工业级高阶实战案例** 以及 **免费云算力与全系硬件速查资源**，助你在 AMD AI 生态中走得更深更远！

---

## 9.1 🌟 AMD 官方明星开源 AI 与 Agent 项目巡礼

AMD 正在 GitHub（[`github.com/amd`](https://github.com/amd)、[`github.com/ROCm`](https://github.com/ROCm)、[`github.com/AMD-AGI`](https://github.com/AMD-AGI)）快速构建覆盖端侧 AI PC 到数据中心集群的全栈开源工具链，以下项目最值得收藏关注：

### 一、端侧本地智能体与开发者生产力工具

| 开源项目 | 官方仓库链接 | 核心亮点与适用场景 |
|:---|:---|:---|
| **AMD GAIA** | [`amd/gaia`](https://github.com/amd/gaia) | AMD 官方开源的本地隐私 **AI Agent 框架与智能体工厂**，专为 Ryzen AI（NPU + iGPU）与 Radeon 硬件打造，开箱内置代码助手、RAG 问答、邮件摘要与自动评测组件。 |
| **Lemonade SDK** | [`lemonade-sdk/lemonade`](https://github.com/lemonade-sdk/lemonade) | 本课程三大标准服务引擎之一！一站式统一调度 Radeon GPU（`llama.cpp`）与 Ryzen AI XDNA 2 NPU（`FastFlowLM`），自带模型管理与标准 `/v1` 服务接口。 |
| **AMD Skills** | [`amd/skills`](https://github.com/amd/skills) | 将 AMD 官方 ROCm 调优、排错与算子最佳实践打包成标准 `SKILL.md`，一行 `npx skills add amd/skills` 即可赋能你的 `Qoder`、Codex 与 WorkBuddy。 |
| **ROCm CLI** | [`ROCm/rocm-cli`](https://github.com/ROCm/rocm-cli) | ROCm 10 新一代一体化命令行管家，一条命令完成 SDK 安装、硬件体检（`examine`）与模型服务拉起。 |

### 二、自动优化、模型量化与高性能训练/推理底座

| 开源项目 | 官方仓库链接 | 核心亮点与适用场景 |
|:---|:---|:---|
| **Hyperloom** | [`AMD-AGI/Hyperloom`](https://github.com/AMD-AGI/Hyperloom) | **多智能体全自动推理与算子优化引擎**（串联 `Magpie` 压测、`TraceLens` 剖析、`Arbor` 参数搜索与 `GEAK` 自动重写融合算子）。 |
| **TraceLens** | [`AMD-AGI/TraceLens`](https://github.com/AMD-AGI/TraceLens) | 纯 Python 跨框架 GPU 性能剖析报告生成器，自动解析 PyTorch / `vLLM` Trace 揪出性能瓶颈。 |
| **AMD Quark** | [`amd/Quark`](https://github.com/amd/Quark) | AMD 官方全平台训练后量化（PTQ）与压缩工具箱，支持 `INT4 AWQ/GPTQ`、`FP8`、`MXFP4`，通吃 Radeon、Ryzen AI NPU 与 Instinct。 |
| **AITER** | [`ROCm/aiter`](https://github.com/ROCm/aiter) | **AI Tensor Engine for ROCm** —— 面向大模型推理极致打磨的高性能融合算子库（`vLLM` 开启 `VLLM_ROCM_USE_AITER=1` 即可调用）。 |
| **Primus & Primus-Turbo** | [`AMD-AGI/Primus`](https://github.com/AMD-AGI/Primus) | 面向大规模基础模型预训练与后训练的高性能统一训练框架（兼容 Megatron-LM / TorchTitan / JAX）及底层 FP8/MoE 加速库。 |
| **TheRock** | [`ROCm/TheRock`](https://github.com/ROCm/TheRock) | 新一代模块化跨平台 ROCm 构建与分发基座，让 ROCm 同时支持 Linux 与 Windows 并可直接装入 Python 虚拟环境。 |

---

## 9.2 🚀 基于 AMD 平台的前沿科研与行业高阶案例集（本仓库硬核资产）

如果你已经不满足于日常办公和单卡轻量应用，渴望挑战 **机器人具身智能、工业级推荐系统、多卡分布式大模型训练、百亿/千亿模型极限部署或底层 C++ HIP 算子开发**，本仓库为你完整保留并打磨了以下进阶实战专题：

| 进阶方向 | 生态代表项目与复现要点 | 联合生态与进阶深造课程 |
|:---|:---|:---|
| 🤖 **具身智能（Embodied AI）** | **Every Embodied 策略复刻**：在 MuJoCo 桌面抓取环境中基于 ROCm 复刻 ACT、SmolVLA、Pi0 与 Pi0.5 训练与闭环评估 | 🔗 [`datawhalechina/every-embodied`](https://github.com/datawhalechina/every-embodied) |
| 📚 **工业级推荐系统** | **Torch-RecHub 全链路实战**：基于 ROCm + PyTorch 跑通 DeepFM (CTR)、DIN (序列建模)、DSSM (召回)、MMOE (多任务) 与 ONNX 导出 | 🔗 [`datawhalechina/torch-rechub`](https://github.com/datawhalechina/torch-rechub) |
| 🧠 **分布式从零训练大模型** | **Happy-LLM 多卡集群实战**：亲手搭建 Transformer 大模型架构并在 AMD 多卡环境跑通预训练、SFT 与偏好对齐流程 | 🔗 [`datawhalechina/happy-llm`](https://github.com/datawhalechina/happy-llm) |
| ⚡ **旗舰 AI PC 极限部署** | • **Ryzen AI Max+ 395 (Strix Halo 128GB)** 超大统一内存跑通 `Qwen3.6-35B-A3B` 等百亿级 MoE 模型<br/>• **[Strix Arena 性能排行榜](https://strix-arena.web.app/)**（真机推理跑分与可复现配方榜） | 🔗 [`datawhalechina/self-llm`](https://github.com/datawhalechina/self-llm) |
| 🔬 **底层 GPU 架构与算子编程** | 深入 AMD GPU 计算单元（CU / Wavefront / LDS）底层架构，手写高性能 HIP 与 Triton 融合算子及 PyTorch C++ 扩展 | 🔗 **[`datawhalechina/hello-gpu`](https://github.com/datawhalechina/hello-gpu)** |

---

## 9.3 🧰 AMD 开发者算力福利、架构字典与资源百宝箱

### 1. 免费云端与高校算力资源
- ☁️ **[AMD Radeon Cloud 使用与 SSH 远程连接指南](../cloud/amd-radeon-cloud.md)**：手头无卡也能免费体验大显存 AMD GPU。
- 🎓 **[AUP Learning Cloud 端侧算力指南](../cloud/aup-learning-cloud.md)**：AMD 大学计划（AMD University Program）教学与实训算力支持。

### 2. 权威硬件架构字典与官方文档入口
- 🗺️ **[ROCm 软件栈交互式全景图谱](/atlas/rocm/index.html)**：一图看清从底层驱动、编译器到上层 AI 框架的全景拼图。
- 📋 **[硬件体检与 `gfx` 架构对照表](../00-guide/03-hardware-and-cloud.md)** · **[ROCm 10.0.0 环境配置指南](../01-first-model/02-rocm-env-setup.md)**：一秒查清 RDNA 3 (`gfx110X`) / RDNA 3.5 (`gfx115X`) / RDNA 4 (`gfx120X`) / CDNA 3 (`gfx942`) 架构代号与 `pip` 安装源。
- 📰 **[AMD ROCm 官方文档总站](https://rocm.docs.amd.com/)** 与 **[AMD ROCm 技术博客](https://rocm.blogs.amd.com/)**。

### 3. 已验证配置总索引与开源共建邀请
- ⚙️ **全书已验证客户端与引擎配置模板库**：[`src/configs/`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/README.md)（含 WorkBuddy、`Qoder`、Codex 标准配置）与 [Lemonade / NPU 引擎部署排错](../01-first-model/04-lemonade-server.md)。
- 🤝 **加入我们，一起共建更友好的 AMD AI 开源学习生态**！
  - 提交 **[新手试学反馈卡](https://github.com/datawhalechina/hello-rocm/issues/new/choose)** 帮助打磨零卡点教程；
  - 参照 **[内容规范指南](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/CONTENT_GUIDE.md)** 与 **[案例五要素模板](../08-fun-cases/_CASE_TEMPLATE.md)** 提交你的创意案例 PR；
  - 加入 **[Datawhale × hello-rocm 飞书社区讨论](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)**，与志同道合的伙伴组队同行（参见 **[20 天组队学习计划](../learning/20-day-learning-plan.md)**）！
