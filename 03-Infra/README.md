<div align=center>
  <h1>03-Infra</h1>
  <strong>⚙️ ROCm 基础设施与算力编程</strong>
</div>

<div align="center">

*从 AMD AI 硬件全景到 HIP 算子与性能分析*

[返回主页](../README.md)

</div>

## 简介

&emsp;&emsp;本模块面向希望在 **AMD GPU + ROCm** 上建立系统认知的开发者：从 AI 硬件与 ROCm 软件栈全景，到 PyTorch 调用链与 GPU 架构，再到用 **HIP** 手写算子、结合 **rocBLAS / MIOpen** 与性能测量，把「基础设施」讲清楚、练到位。

&emsp;&emsp;内容与仓库内 **第 1～4 章** 连载教程一一对应，默认实验环境为 **Ubuntu 22.04 / 24.04 + ROCm 7.x**，示例设备以 **AMD AI+ MAX395 / Radeon 8060S（gfx1151）** 等为主，读者可按自身显卡与 ROCm 版本对照阅读。目录结构如下：

```
03-Infra/
├── 01-embrace-amd-ai/
│   ├── README.md
│   └── images/
├── 02-decode-ai-accelerator/
│   ├── README.md
│   ├── code/
│   └── images/
├── 03-handwrite-rocm-operator/
│   ├── README.md
│   ├── code/
│   └── images/
├── 04-custom-pytorch-operator/
│   ├── README.md
│   ├── code/
│   └── images/
└── README.md
```

## 教程列表

### 第 1 章：拥抱 AMD AI 算力新时代

&emsp;&emsp;从 Ryzen AI（NPU + GPU）、Radeon 独显到 Instinct 数据中心加速卡，梳理 AMD AI 产品线与典型应用场景；说明 **ROCm** 在栈中的位置，并配合 **PyTorch** 完成 ResNet 训练与 Qwen 等大模型推理等动手实验，建立「能在 AMD 上跑什么」的整体图景。

- **适合人群**：初次接触 AMD AI / ROCm 的开发者与学生
- **难度等级**：⭐⭐
- **预计时间**：2～3 小时

📖 [阅读第 1 章](./01-embrace-amd-ai/README.md)

---

### 第 2 章：解密 AI 加速器——从软件栈到硬件架构

&emsp;&emsp;用 `ldd` 等工具追踪 **PyTorch → HIP → HSA → 驱动 → GPU** 的调用链路，理解 CPU「低延迟」与 GPU「高吞吐」、**SIMT** 执行模型；结合 **CU、LDS、显存带宽** 等概念，读懂 AMD GPU 在 AI 负载下的行为方式。

- **适合人群**：已能跑通模型、希望理解底层栈与硬件的开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：3～4 小时

📖 [阅读第 2 章](./02-decode-ai-accelerator/README.md)

---

### 第 3 章：迈入 ROCm 编程世界——手写一个「PyTorch 算子」

&emsp;&emsp;介绍 **HIP** 与 CUDA 的对应关系、Host/Device 代码结构；从 **手写 Kernel** 复现 Tensor 加法，到使用 **`hipEvent`** 做计时，并初步接触 **rocBLAS** 与 **MIOpen**，完成从 Python 到设备端代码的过渡。

- **适合人群**：具备 C++ 基础、准备编写或阅读自定义算子的开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：2～3 小时

📖 [阅读第 3 章](./03-handwrite-rocm-operator/README.md)

---

### 第 4 章：为 PyTorch 编写自定义 ROCm 算子

&emsp;&emsp;从 PyTorch 的 Python 调用链出发，用 **C++ Extension** 机制将 HIP Kernel 注册为自定义算子；实战 **Fused Swish** 算子、**Grid-Stride Loop** 优化、**Autograd** 自动求导集成，并通过 **内存墙基准测试** 量化带宽瓶颈，完成从「手写 Kernel」到「PyTorch 可调用算子」的完整闭环。

- **适合人群**：希望将自定义 HIP 算子集成到 PyTorch 训练/推理流程的开发者
- **难度等级**：⭐⭐⭐⭐
- **预计时间**：3～4 小时

📖 [阅读第 4 章](./04-custom-pytorch-operator/README.md)

---

## 环境要求

### 硬件要求

- AMD GPU（支持 ROCm 的显卡，如 RX 7000 / 9000 系列、Ryzen AI MAX / AI 300、Instinct MI 系列等）
- 各章实验对显存要求不同：第 1 章涉及 ResNet / 大模型推理时请预留足够显存；第 3、4 章以小型 HIP 程序与矩阵实验为主，一般消费级显卡即可

### 软件要求

- 操作系统：Linux（Ubuntu 22.04+，教程正文以 22.04 / 24.04 为例）
- ROCm 7.10.0 或更高版本（章节内亦可能出现 7.x 表述，以你本机 `rocm-smi` / 发行说明为准）
- 第 1 章：Python 3.10+、PyTorch（ROCm 构建版）
- 第 3、4 章：建议安装 **ROCm HIP 开发组件**（如 `hipcc`、`rocm-dev` 等，随发行版包名可能略有差异）、CMake 3.16+、GCC 9+ 或 Clang 12+

## 常见问题

<details>
<summary>Q: 如何确认我的 AMD GPU 是否支持 ROCm？</summary>

请参考 [ROCm 官方支持列表](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) 查看支持的 GPU 型号与发行版组合。

</details>

<details>
<summary>Q: 第 3、4 章与第 1、2 章是什么关系？</summary>

第 1、2 章侧重**全景与栈/硬件认知**；第 3、4 章侧重**HIP 编程与算子级实践**。两章标题相近：可先读第 3 章建立最小闭环，再读第 4 章做映射与性能深化；若时间有限，也可按个人基础选读其中一章。

</details>

<details>
<summary>Q: 编译 HIP 程序时提示找不到头文件或链接库？</summary>

1. 确认已安装对应版本的 ROCm 开发包，且 `hipcc` 在 `PATH` 中  
2. 检查 `HIP_PATH`、`ROCM_PATH` 等环境变量是否与安装路径一致  
3. 参考 [HIP 安装与入门](https://rocm.docs.amd.com/projects/HIP/en/latest/) 核对依赖

</details>

## 参考资源

- [ROCm 官方文档](https://rocm.docs.amd.com/)
- [HIP 编程指南](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [rocBLAS 文档](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
- [MIOpen 文档](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [PyTorch for ROCm](https://pytorch.org/get-started/locally/)

---

<div align="center">

**欢迎贡献更多 Infra 与算子实践内容！** 🎉

[提交 Issue](https://github.com/datawhalechina/hello-rocm/issues) | [提交 PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
