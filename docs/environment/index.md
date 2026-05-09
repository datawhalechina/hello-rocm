<div align=center>
  <h1>00-Environment</h1>
  <div align='center'>

  [![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

  </div>
  <strong>🛠️ ROCm 基础环境安装与配置</strong>
</div>

<div align="center">

*统一环境基线 · ROCm 7.12.0 · 所有后续章节的前置依赖*

[返回主页](../README.md) | [English](./README_en.md)

</div>

## 简介

&emsp;&emsp;本章节是整个 **hello-rocm** 项目的环境基线参考。统一以 **ROCm 7.12.0**（Technology Preview，2026-03-26 发布）为目标版本，覆盖 Windows 和 Ubuntu 双平台的安装、校验与卸载流程。

&emsp;&emsp;后续所有章节（01-Deploy、02-Fine-tune 等）的环境准备均以本章为基准。如需使用其他 ROCm 版本或其他 GPU 架构，请参考 [GPU 架构对照表](./rocm-gpu-architecture-table.md) 进行对应替换。

> 💡 **平台建议**：Windows 已支持 ROCm 体验与推理验证，但 ROCm 生态工具链（如 rocminfo、amd-smi、多卡支持、容器化部署等）在 **Ubuntu** 上支持更完整。**建议使用 Ubuntu 24.04 作为主力开发环境**，Windows 可作为快速体验或轻量推理使用。

> ⚠️ **ROCm 7.12.0 为 Technology Preview 版本**，不适合生产环境。生产环境请使用 [ROCm 7.2 production stream](https://rocm.docs.amd.com/en/latest/)。

> ⚠️ **Windows 用户必读**：安装前务必确认你的 **Adrenalin Driver 版本** 和 **Windows 版本** 符合要求（见下方版本信息表），否则 ROCm 将无法正常运行。

---

## 版本信息

| 项目 | 要求 | 下载链接 |
|:---|:---|:---|
| ROCm | 7.12.0 (Technology Preview) | [官方安装页](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html) |
| PyTorch | 2.10.0 / 2.9.1 | 通过 uv 安装（见下文） |
| Python | 3.11 / 3.12 / 3.13 | 由 uv 自动管理 |
| **Windows 版本** | **11 25H2** | — |
| **Adrenalin Driver (Windows)** | **26.3.1** | [**⬇️ 下载 Adrenalin 26.3.1**](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-3-1.html#Downloads) |
| **Visual Studio 2022 (Windows)** | **Community，勾选「使用 C++ 的桌面开发」** | [**⬇️ 下载 VS 2022**](https://visualstudio.microsoft.com/zh-hans/downloads/) |
| Ubuntu | 24.04.3 (HWE kernel 6.14 for Ryzen APU) | [Ubuntu Downloads](https://ubuntu.com/download/desktop) |

---

## 目录

- [GPU 架构对照表（独立文件）](./rocm-gpu-architecture-table.md)
- [一、Windows 安装](#一windows-11-安装)
- [二、Ubuntu 安装](#二ubuntu-2404-安装)
- [三、校验安装](#三校验安装)
- [四、卸载 ROCm](#四卸载-rocm)
- [五、切换其他 GPU 架构](#五切换其他-gpu-架构)

---

## 一、Windows 11 安装

> 以 **Ryzen AI Max+ 395（gfx1151）** 为例。
>
> 📖 官方文档：[Install ROCm on Windows](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html?fam=ryzen&gpu=max-pro-395&os=windows&os-version=11_25h2&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/pytorch.html?fam=ryzen&gpu=max-pro-395&os=windows)

### 1.1 前置条件检查

| ✅ 检查项 | 要求 |
|:---|:---|
| **Windows 版本** | **必须 Windows 11 25H2**（设置 → 系统 → 关于 查看） |
| **Adrenalin 驱动** | **必须 26.3.1**（[⬇️ 下载](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-3-1.html#Downloads)） |
| **Visual Studio 2022**（可选） | Community 版即可，安装时勾选「使用 C++ 的桌面开发」（[⬇️ 下载](https://visualstudio.microsoft.com/zh-hans/downloads/)）。AMD Quark 等需要编译自定义算子时必需 |

![Visual Studio 安装勾选「使用 C++ 的桌面开发」](./images/visual_studil_c++_desktop_installer.png)

### 1.2 卸载冲突软件

- 控制面板 → 程序 → 卸载程序 → 移除所有 **HIP SDK** 相关项

### 1.3 关闭 Windows 安全功能

以下功能会干扰 ROCm 运行，**必须关闭**：

- **WDAG**：控制面板 → 程序和功能 → 启用或关闭 Windows 功能 → 取消勾选 "Microsoft Defender Application Guard"
- **SAC**：设置 → 隐私和安全 → Windows 安全中心 → 应用和浏览器控制 → 智能应用控制设置 → **关闭**

### 1.4 安装 uv（Python 包管理器）

本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境和依赖，替代传统的 pip + venv 流程。uv 由 Rust 编写，速度提升 10-100 倍。

```powershell
# Windows 安装 uv（PowerShell）
irm https://astral.sh/uv/install.ps1 | iex

# 或使用 winget
# winget install astral-sh.uv

# 验证安装
uv --version
```

> 📖 更多安装方式参考：[uv 入门教程](https://www.runoob.com/python3/uv-tutorial.html)

### 1.5 安装 ROCm + PyTorch

```powershell
# 安装 Python 3.12（uv 内置版本管理，无需单独安装 Python）
uv python install 3.12

# 创建虚拟环境
uv venv --python 3.12
.venv\Scripts\activate

# 安装 ROCm 运行时 + 库（gfx1151 = Ryzen AI Max+ 395/390/385）
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]"

# 安装 PyTorch
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio

# 安装其他项目依赖（如有 requirements.txt）
uv pip install -r requirements.txt
```

> ⚠️ 不要将 ROCm DLL 复制到 System32，否则会引起冲突。
>
> 💡 上述 `gfx1151` 对应 **Ryzen AI Max 系列**（395/390/385）。其他 GPU 请替换 `--index-url`：
>
> | 你的 GPU | 替换为 |
> |:---|:---|
> | Ryzen AI PRO 400 系列 (AI 9 HX PRO 475 等) | `https://repo.amd.com/rocm/whl/gfx1150/` |
> | Radeon RX 9070 XT / 9060 XT | `https://repo.amd.com/rocm/whl/gfx120X-all/` |
> | Radeon RX 7900 XTX / 7800 XT | `https://repo.amd.com/rocm/whl/gfx110X-all/` |
> | Instinct MI300X / MI325X | `https://repo.amd.com/rocm/whl/gfx94X-dcgpu/` |
>
> 完整对照请查阅 [GPU 架构对照表](./rocm-gpu-architecture-table.md) 或 [官方兼容性矩阵](https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html)。

> 🚀 **国内加速提示**：对于非 ROCm 的普通 PyPI 包，可配置镜像源加速下载：
> ```bash
> # 全局配置清华镜像（仅影响未指定 --index-url 的包）
> uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 二、Ubuntu 24.04 安装

> 以 **Ryzen AI Max+ PRO 395（gfx1151）** 为例。
>
> 📖 官方文档：[Install ROCm on Ubuntu](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html?fam=ryzen&gpu=max-pro-395&os=ubuntu&os-version=24.04&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/pytorch.html?fam=ryzen&gpu=max-pro-395&os=linux)

### 2.1 安装 uv 与依赖

```bash
sudo apt install -y libatomic1 libquadmath0

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证
uv --version
```

### 2.2 安装 ROCm + PyTorch（uv 方式，推荐）

```bash
# 安装 Python 3.12
uv python install 3.12

# 创建虚拟环境
uv venv --python 3.12
source .venv/bin/activate

# 安装 ROCm（gfx1151 = Ryzen AI Max+ 395/390/385）
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]"

# 安装 PyTorch
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio

# 安装其他项目依赖（如有 requirements.txt）
uv pip install -r requirements.txt
```

> 💡 其他 GPU 替换 `--index-url` 即可，对照表见 [Windows 1.5 节](#15-安装-rocm--pytorch) 或 [GPU 架构对照表](./rocm-gpu-architecture-table.md)。

### 2.3 备选：一键安装脚本

如果你希望自动完成内核、驱动、ROCm 全套安装，可使用本项目提供的安装脚本：

```bash
sudo apt update
sudo apt install -y curl git

git clone -b unified-installer https://github.com/amdjiahangpan/rocm-install-script.git
cd rocm-install-script
chmod +x install.sh
sudo ./install.sh
```

> 📖 脚本详情及参数说明：[rocm-install-script (unified-installer 分支)](https://github.com/amdjiahangpan/rocm-install-script/tree/unified-installer)

### 2.4 配置 GPU 访问权限（Linux）

> 💡 此步可在安装后任意时间执行，重启生效即可。

```bash
sudo usermod -a -G render,video "$LOGNAME"
# 重启或重新登录后生效
```

---

## 三、校验安装

### 3.1 PyTorch 检测（Windows / Linux 通用）

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

期望输出：

```
PyTorch: 2.10.0+rocm7.12.0
ROCm available: True
Device: AMD Radeon Graphics
```

> 💡 ROCm 通过 HIP 兼容 CUDA API，`torch.cuda.is_available()` 返回 `True` 是正常的。

### 3.2 简单计算测试

```python
import torch
x = torch.randn(3, 3, device='cuda')
y = torch.randn(3, 3, device='cuda')
print(x @ y)
```

### 3.3 Linux 专用工具

```bash
rocminfo | grep -E "Name:|Marketing Name:"
rocm-smi          # 或 amd-smi monitor
hipinfo           # pip 安装方式可用
```

### 3.4 常见问题

| 现象 | 原因 | 解决 |
|:---|:---|:---|
| `torch.cuda.is_available()` = `False` | 驱动版本不匹配 | Windows：确认 [Adrenalin 26.3.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-3-1.html#Downloads)；Linux：确认 inbox kernel |
| `No GPU detected` (Linux) | 未加入 render/video 组 | `sudo usermod -a -G render,video $LOGNAME` + 重启 |
| DLL 加载错误 (Windows) | SAC/WDAG 未关闭 | 见 [1.3 节](#13-关闭-windows-安全功能) |

---

## 四、卸载 ROCm

### Windows

直接删除项目中的 `.venv` 文件夹即可（资源管理器中右键删除，或在 CMD 中执行）：

```cmd
rmdir /s /q .venv
```

如需卸载 Adrenalin 驱动：控制面板 → 程序 → 卸载程序 → AMD Software

### Ubuntu

```bash
rm -rf .venv
```

---

## 五、切换其他 GPU 架构

只需将安装命令中的 `--index-url` 或 apt 包名替换为对应值：

| GPU 示例 | LLVM Target | pip index URL |
|:---|:---|:---|
| MI355X / MI350X | gfx950 | `https://repo.amd.com/rocm/whl/gfx950-dcgpu/`  |
| MI300X / MI325X | gfx942 | `https://repo.amd.com/rocm/whl/gfx94X-dcgpu/` |
| RX 9070 XT | gfx1201 | `https://repo.amd.com/rocm/whl/gfx120X-all/`  |
| RX 7900 XTX | gfx1100 | `https://repo.amd.com/rocm/whl/gfx110X-all/`  |
| Ryzen AI Max 395 | gfx1151 | `https://repo.amd.com/rocm/whl/gfx1151/`  |
| Ryzen AI PRO 400 | gfx1150 | `https://repo.amd.com/rocm/whl/gfx1150/`  |

完整对照表见 [GPU 架构对照表](./rocm-gpu-architecture-table.md)。

---

> 📖 完整官方文档：
> - [ROCm 7.12.0 Release Notes](https://rocm.docs.amd.com/en/7.12.0-preview/about/release-notes.html)
> - [Compatibility Matrix](https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html)
> - [Install ROCm](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html)
> - [Install PyTorch](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/pytorch.html)
