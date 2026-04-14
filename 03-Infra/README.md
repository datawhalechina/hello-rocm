<div align=center>
  <h1>03-Infra</h1>
  <strong>⚙️ ROCm 算子优化实践</strong>
</div>

<div align="center">

*CUDA 到 ROCm 的迁移与优化指南*

[返回主页](../README.md)

</div>

## 简介

&emsp;&emsp;本模块专注于 ROCm 基础设施和算子优化，帮助开发者将 CUDA 代码迁移到 ROCm 平台，并掌握 AMD GPU 上的性能优化技巧。

&emsp;&emsp;随着 AMD GPU 在 AI 领域的崛起，越来越多的开发者需要将现有的 CUDA 代码迁移到 ROCm 平台。本模块提供系统的迁移指南和优化实践，帮助你充分发挥 AMD GPU 的性能潜力。

## 教程列表

### HIPify 自动化迁移实战

&emsp;&emsp;HIPify 是 AMD 提供的自动化工具，可以将 CUDA 代码自动转换为 HIP 代码（ROCm 的编程接口）。本教程将指导你如何使用 HIPify 完成代码迁移。

- **适合人群**：有 CUDA 开发经验、需要迁移代码的开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：2 小时

**核心内容：**
- HIPify 工具安装与配置
- 自动转换 CUDA 代码
- 手动调整与兼容性处理
- 迁移后的验证与测试

📖 [开始学习 HIPify 迁移教程](./HIPify/README.md)

---

### BLAS 与 DNN 的无缝切换

&emsp;&emsp;本教程介绍如何将 cuBLAS/cuDNN 代码迁移到 rocBLAS/MIOpen，实现基础数学库和深度学习库的无缝切换。

- **适合人群**：使用底层数学库的开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：1.5 小时

**核心内容：**

| CUDA | ROCm | 说明 |
|------|------|------|
| cuBLAS | rocBLAS | 基础线性代数库 |
| cuDNN | MIOpen | 深度学习原语库 |
| cuFFT | rocFFT | 快速傅里叶变换 |
| cuSPARSE | rocSPARSE | 稀疏矩阵库 |

- API 对照与转换
- 性能调优技巧
- 常见问题解决

📖 [开始学习 BLAS/DNN 迁移教程](./BLAS-DNN/README.md)

---

### NCCL 到 RCCL 的迁移

&emsp;&emsp;RCCL（ROCm Communication Collectives Library）是 AMD 版本的集合通信库，用于多 GPU 分布式训练。本教程介绍如何从 NCCL 迁移到 RCCL。

- **适合人群**：进行多 GPU 分布式开发的开发者
- **难度等级**：⭐⭐⭐⭐
- **预计时间**：2 小时

**核心内容：**
- RCCL 安装与环境配置
- NCCL API 到 RCCL 的映射
- 多节点通信配置
- 分布式训练集成

📖 [开始学习 RCCL 迁移教程](./RCCL/README.md)

---

### Nsight 到 Rocprof 的映射

&emsp;&emsp;性能分析是优化的基础。本教程介绍如何使用 ROCm 的性能分析工具替代 NVIDIA Nsight 系列工具。

- **适合人群**：需要进行性能调优的开发者
- **难度等级**：⭐⭐⭐
- **预计时间**：1.5 小时

**核心内容：**

| NVIDIA 工具 | ROCm 工具 | 用途 |
|-------------|-----------|------|
| Nsight Systems | rocprof | 系统级性能分析 |
| Nsight Compute | rocprof --stats | 内核级性能分析 |
| nvprof | rocprof | 命令行分析工具 |
| Nsight Graphics | ROCm Debugger | 图形调试 |

- 性能数据采集
- 热点分析与优化
- 可视化工具使用

📖 [开始学习 Rocprof 使用教程](./Rocprof/README.md)

---

## 环境要求

### 硬件要求

- AMD GPU（支持 ROCm）
- 建议使用 MI 系列或 RX 7000 系列

### 软件要求

- 操作系统：Linux (Ubuntu 22.04+)
- ROCm 7.10.0 或更高版本
- CMake 3.16+
- GCC 9+ 或 Clang 12+

## 快速开始

```bash
# 1. 安装 ROCm 开发工具
sudo apt install rocm-dev hip-dev

# 2. 验证 HIP 环境
hipcc --version

# 3. 编译简单的 HIP 程序
hipcc hello_hip.cpp -o hello_hip
./hello_hip
```

## CUDA 与 HIP 快速对照

```cpp
// CUDA 代码
cudaMalloc(&d_ptr, size);
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_ptr);
cudaFree(d_ptr);

// HIP 代码（几乎相同！）
hipMalloc(&d_ptr, size);
hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice);
hipLaunchKernelGGL(kernel, grid, block, 0, 0, d_ptr);
hipFree(d_ptr);
```

## 常见问题

<details>
<summary>Q: HIPify 转换后的代码能否在 NVIDIA GPU 上运行？</summary>

是的！HIP 代码具有跨平台特性，可以通过条件编译同时支持 AMD 和 NVIDIA GPU。

</details>

<details>
<summary>Q: ROCm 支持哪些 CUDA 版本的 API？</summary>

ROCm 的 HIP 接口兼容大部分 CUDA Runtime API，具体兼容性请参考 [HIP API 文档](https://rocm.docs.amd.com/projects/HIP/en/latest/)。

</details>

<details>
<summary>Q: 迁移后性能下降怎么办？</summary>

1. 使用 rocprof 进行性能分析
2. 检查内存访问模式是否适合 AMD GPU 架构
3. 调整 workgroup size 和 wavefront 配置
4. 查阅 AMD GPU 架构优化指南

</details>

## 参考资源

- [ROCm 官方文档](https://rocm.docs.amd.com/)
- [HIP 编程指南](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [rocBLAS 文档](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
- [MIOpen 文档](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [RCCL 文档](https://rocm.docs.amd.com/projects/rccl/en/latest/)

---

<div align="center">

**欢迎贡献更多 Infra 教程！** 🎉

[提交 Issue](https://github.com/datawhalechina/hello-rocm/issues) | [提交 PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
