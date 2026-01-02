# hello-rocm

<div align="center">

**AMD YES! 🚀**

*开源 · 社区驱动 · 让 AMD AI 生态更易用*

</div>


## News

- *2025.12.11:* [*ROCm 7.10.0 Release Notes*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)


## About

自 **ROCm 7.10.0** (2025年12月11日发布) 以来，ROCm 已支持像 CUDA 一样在 Python 虚拟环境中无缝安装，并正式支持 **Linux 和 Windows** 双系统。这标志着 AMD 在 AI 领域的重大突破——学习者与大模型爱好者在硬件选择上不再局限于 NVIDIA，AMD GPU 正成为一个强有力的竞争选择。

苏妈在发布会上宣布 ROCm 将保持 **每 6 周一个新版本** 的迭代节奏，并全力转向 AI 领域。前景令人振奋！

然而，目前全球范围内缺乏系统的 ROCm 大模型推理、部署、训练、微调及 Infra 的学习教程。**hello-rocm** 应运而生，旨在填补这一空白。


## Mission

构建一个开源、社区驱动的 AMD ROCm AI 学习平台，让每个人都能轻松上手 AMD GPU 进行大模型开发。


## Project Structure

```
hello-rocm/
├── 01-Infra/              # ROCm 算子优化
├── 02-Inference/          # ROCm 推理优化
├── 03-Training/           # 模型训练与微调
└── 04-AMD-YES/            # 优秀 AMD 项目收录
```


## Modules

### 01. Infra - ROCm 算子优化
<table align="center">
  <tr>
    <td valign="top" width="50%">
      <b>核心内容</b><br>
      • ROCm 基础架构<br>
      • HIP 编程入门<br>
      • 自定义算子开发<br>
      • 性能分析与优化<br>
      • Composable Kernel (CK) 实践
    </td>
  </tr>
</table>

**入门教程** → [Getting Started with ROCm Infra](./01-Infra/README.md)

### 02. Inference - ROCm 推理优化
<table align="center">
  <tr>
    <td valign="top" width="50%">
      <b>核心内容</b><br>
      • 模型格式转换<br>
      • vLLM / TGI on ROCm<br>
      • 量化技术 (GPTQ, AWQ, GGUF)<br>
      • LM Studio / Ollama 集成<br>
      • 推理性能调优
    </td>
  </tr>
</table>

**入门教程** → [Getting Started with ROCm Inference](./02-Inference/README.md)

### 03. Training - 模型训练与微调
<table align="center">
  <tr>
    <td valign="top" width="50%">
      <b>核心内容</b><br>
      • 单卡/多卡微调 (LoRA, QLoRA)<br>
      • 全量训练最佳实践<br>
      • 分布式训练 (RCCL)<br>
      • 集群部署与通信<br>
      • 训练性能优化
    </td>
  </tr>
</table>

**入门教程** → [Getting Started with ROCm Training](./03-Training/README.md)

### 04. AMD-YES Project - 优秀项目收录
<table align="center">
  <tr>
    <td valign="top" width="50%">
      <b>核心内容</b><br>
      • 社区优秀 ROCm 项目<br>
      • 工具与框架推荐<br>
      • 成功案例分享<br>
      • 贡献者作品展示
    </td>
  </tr>
</table>

**查看项目** → [AMD-YES Project Gallery](./04-AMD-YES/README.md)

## Contributing

我们欢迎所有形式的贡献！无论是：

- 完善或新增教程
- 修复错误与 Bug
- 分享你的 AMD 项目
- 提出建议与想法

请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解详情。


## Resources

- [ROCm 官方文档](https://rocm.docs.amd.com/)
- [AMD GitHub](https://github.com/amd)
- [ROCm Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)


## License

[MIT License](./LICENSE)

---

<div align="center">

**让我们一起构建 AMD AI 的未来！** 💪

Made with ❤️ by the hello-rocm community

</div>
