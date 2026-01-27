<div align=center>
  <img src="./images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*开源 · 社区驱动 · 让 AMD AI 生态更易用*

中文 | [English](./README_en.md)

</div>


&emsp;&emsp;自 **ROCm 7.10.0** (2025年12月11日发布) 以来，ROCm 已支持像 CUDA 一样在 Python 虚拟环境中无缝安装，并正式支持 **Linux 和 Windows** 双系统。这标志着 AMD 在 AI 领域的重大突破——学习者与大模型爱好者在硬件选择上不再局限于 NVIDIA，AMD GPU 正成为一个强有力的竞争选择。

&emsp;&emsp;苏妈在发布会上宣布 ROCm 将保持 **每 6 周一个新版本** 的迭代节奏，并全力转向 AI 领域。前景令人振奋！

&emsp;&emsp;然而，目前全球范围内缺乏系统的 ROCm 大模型推理、部署、训练、微调及 Infra 的学习教程。**hello-rocm** 应运而生，旨在填补这一空白。

&emsp;&emsp;**项目的主要内容就是教程，让更多的学生和未来的从业者了解和熟悉 AMD ROCm 的使用方法！任何人都可以提出 issue 或是提交 PR，共同构建维护这个项目。**

> &emsp;&emsp;***学习建议：本项目的学习建议是，先学习环境配置和部署，然后再学习模型的微调，最后再探索 Infra 算子优化。初学者可以从 LM Studio 或 vLLM 部署开始优先学习。***

### 最新动态

- *2025.12.11:* [*ROCm 7.10.0 Release Notes*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

## 项目意义

&emsp;&emsp;什么是 ROCm？

> ROCm（Radeon Open Compute）是 AMD 推出的开源 GPU 计算平台，旨在为高性能计算和机器学习提供开放的软件栈。它支持 AMD GPU 进行并行计算，是 CUDA 在 AMD 平台上的替代方案。

&emsp;&emsp;百模大战正值火热，开源 LLM 层出不穷。然而，目前大多数大模型教程和开发工具都基于 NVIDIA CUDA 生态。对于想要使用 AMD GPU 的开发者来说，缺乏系统性的学习资源是一个痛点。

&emsp;&emsp;自 ROCm 7.10.0（2025 年 12 月 11 日） 起，AMD 通过 TheRock 项目对 ROCm 底层架构进行了重构，将计算运行时与操作系统解耦，使同一套 ROCm 上层接口可以同时运行在 Linux 与 Windows 上，并支持像 CUDA 一样直接安装到 Python 虚拟环境中使用。这意味着 ROCm 不再是只面向 Linux 的“工程工具”，而是升级为一个真正面向 AI 学习者与开发者的跨平台 GPU 计算平台——无论使用 Windows 还是 Linux，用户都可以更低门槛地使用 AMD GPU 进行训练和推理，大模型与 AI 玩家在硬件选择上不再被 NVIDIA 单一生态所绑定，AMD GPU 正逐步成为一个可以被普通用户真实使用的 AI 计算平台。

&emsp;&emsp;本项目旨在基于核心贡献者的经验，提供 AMD ROCm 平台上大模型部署、微调、训练的完整教程；我们希望充分聚集共创者，一起丰富 AMD AI 生态。

&emsp;&emsp;***我们希望成为 AMD GPU 与普罗大众的桥梁，以自由、平等的开源精神，拥抱更恢弘而辽阔的 AI 世界。***

## 项目受众

&emsp;&emsp;本项目适合以下学习者：


* 手头有一张AMD显卡，想体验一下大模型本地运行;
* 想要使用 AMD GPU 进行大模型开发，但找不到系统教程；
* 希望低成本、高性价比地部署和运行大模型；
* 对 ROCm 生态感兴趣，想要亲自上手实践；

## 项目规划及进展

&emsp;&emsp;本项目拟围绕 ROCm 大模型应用全流程组织，包括环境配置、部署应用、微调训练、算子优化等：


### 项目结构

```
hello-rocm/
├── 01-Deploy/              # ROCm 大模型部署实践
├── 02-Fine-tune/           # ROCm 大模型微调实践
├── 03-AMD-YES/             # AMD 实践案例集合
├── 04-Infra/               # ROCm 算子优化实践
└── 05-References/          # ROCm 优质参考资料
```

### 01. Deploy - ROCm 大模型部署

<p align="center">
  <strong>🚀 ROCm 大模型部署实践</strong><br>
  <em>零基础快速上手 AMD GPU 大模型部署</em><br>
  📖 <strong><a href="./01-Deploy/README.md">Getting Started with ROCm Deploy</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • LM Studio 零基础大模型部署<br>
      • Vllm 零基础大模型部署<br>
    </td>
    <td valign="top" width="50%">
      • SGLang 零基础大模型部署<br>
      • ATOM 零基础大模型部署
    </td>
  </tr>
</table>

### 02. Fine-tune - ROCm 大模型微调

<p align="center">
  <strong>🔧 ROCm 大模型微调实践</strong><br>
  <em>在 AMD GPU 上进行高效模型微调</em><br>
  📖 <strong><a href="./02-Fine-tune/README.md">Getting Started with ROCm Fine-tune</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • 大模型零基础微调教程<br>
      • 大模型单机微调脚本<br>
    </td>
    <td valign="top" width="50%">
      • 大模型多机多卡微调教程
    </td>
  </tr>
</table>

### 03. AMD-YES - AMD 实践案例集合

<p align="center">
  <strong>✨ AMD 实践案例集合</strong><br>
  <em>社区驱动的 AMD GPU 项目实践</em><br>
  📖 <strong><a href="./03-AMD-YES/README.md">Getting Started with ROCm AMD-YES</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • AMchat-高等数学<br>
      • Chat-嬛嬛<br>
      • Tianji-天机<br>
    </td>
    <td valign="top" width="50%">
      • 数字生命<br>
      • happy-llm
    </td>
  </tr>
</table>

### 04. Infra - ROCm 算子优化

<p align="center">
  <strong>⚙️ ROCm 算子优化实践</strong><br>
  <em>CUDA 到 ROCm 的迁移与优化指南</em><br>
  📖 <strong><a href="./04-Infra/README.md">Getting Started with ROCm Infra</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • HIPify 自动化迁移实战<br>
      • BLAS 与 DNN 的无缝切换<br>
    </td>
    <td valign="top" width="50%">
      • NCCL 到 RCCL 的迁移<br>
      • Nsight 到 Rocprof 的映射
    </td>
  </tr>
</table>

### 05. References - ROCm 优质参考资料

<p align="center">
  <strong>📚 ROCm 优质参考资料</strong><br>
  <em>精选的 AMD 官方与社区资源</em><br>
  📖 <strong><a href="./05-References/README.md">ROCm References</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • <a href="https://rocm.docs.amd.com/">ROCm 官方文档</a><br>
      • <a href="https://github.com/amd">AMD GitHub</a><br>
    </td>
    <td valign="top" width="50%">
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCm Release Notes</a><br>
      • 相关新闻
    </td>
  </tr>
</table>

## 贡献指南

&emsp;&emsp;我们欢迎所有形式的贡献！无论是：

* 完善或新增教程
* 修复错误与 Bug
* 分享你的 AMD 项目
* 提出建议与想法

&emsp;&emsp;请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解详情。

&emsp;&emsp;想要深度参与的同学可以联系我们，我们会将你加入到项目的维护者中。

## 致谢

### 核心贡献者


- [宋志学(不要葱姜蒜)-项目负责人](https://github.com/KMnO4-zx) （Datawhale成员）
- [陈榆-项目负责人](https://github.com/lucachen) （内容创作者-谷歌开发者机器学习技术专家）
> 注：欢迎更多贡献者加入！

### 其他

- 如果有任何想法可以联系我们，也欢迎大家多多提出 issue
- 特别感谢以下为教程做出贡献的同学！
- 感谢 AMD University Program 对本项目的支持！！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>

## License

[MIT License](./LICENSE)

---

<div align="center">

**让我们一起构建 AMD AI 的未来！** 💪

Made with ❤️ by the hello-rocm community

</div>
