<div align=center>
  <h1>04-References</h1>
  <strong>📚 ROCm 优质参考资料</strong>
</div>

<div align="center">

*精选的 AMD 官方与社区资源*

[返回主页](../README.md)

</div>

## 简介

&emsp;&emsp;本模块收集整理了 ROCm 和 AMD GPU 相关的优质学习资源，包括官方文档、社区教程、技术博客和相关新闻。帮助你快速找到所需的参考资料。

## 官方资源

### AMD 官方文档

| 资源 | 描述 | 链接 |
|------|------|------|
| ROCm 文档 | ROCm 平台官方文档 | [rocm.docs.amd.com](https://rocm.docs.amd.com/) |
| ROCm Release Notes | 版本发布说明 | [Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) |
| HIP 编程指南 | HIP API 和编程指南 | [HIP Docs](https://rocm.docs.amd.com/projects/HIP/en/latest/) |
| AMD GitHub | AMD 开源项目仓库 | [github.com/amd](https://github.com/amd) |
| ROCm GitHub | ROCm 项目仓库 | [github.com/ROCm](https://github.com/ROCm) |

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

### 2025

- **2025.12.11** - [ROCm 7.10.0 发布](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)
  - 支持 Windows 平台
  - 支持 pip 安装到 Python 虚拟环境
  - TheRock 项目重构底层架构

> 更多新闻持续更新中...

## 硬件支持

### 支持的 GPU 列表

#### 消费级显卡

| 系列 | 型号 | ROCm 支持 |
|------|------|-----------|
| RX 7000 | RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT | ✅ |
| RX 6000 | RX 6900 XT, RX 6800 XT, RX 6800, RX 6700 XT | ✅ |

#### 专业/数据中心显卡

| 系列 | 型号 | ROCm 支持 |
|------|------|-----------|
| MI300 | MI300X, MI300A | ✅ |
| MI250 | MI250X, MI250 | ✅ |
| MI200 | MI210 | ✅ |
| MI100 | MI100 | ✅ |

> 完整支持列表请参考 [ROCm 系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

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
