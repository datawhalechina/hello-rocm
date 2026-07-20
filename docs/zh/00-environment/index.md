<div align=center>
  <h1>00-Environment</h1>
  <div align='center'>

  [![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

  </div>
  <strong>🛠️ ROCm 基础环境安装与配置</strong>
</div>

<div align="center">

*统一环境基线 · ROCm 7.14.0（TheRock）· 所有后续章节的前置依赖*

[返回主页](/zh/) | [English](/00-environment/)

</div>

## 简介

&emsp;&emsp;本章节是整个 **hello-rocm** 项目的环境基线参考。统一以 **ROCm 7.14.0**（ROCm Core SDK，2026-07-15 发布）为目标版本，覆盖 Windows 和 Ubuntu 双平台的安装、校验与卸载流程。

&emsp;&emsp;后续所有章节（01-Deploy、02-Fine-tune 等）的环境准备均以本章为基准。如需使用其他 ROCm 版本或其他 GPU 架构，请参考 [GPU 架构对照表](/zh/00-environment/rocm-gpu-architecture-table) 进行对应替换。

> 🚀 **里程碑版本：ROCm 正式转向 [TheRock](https://github.com/ROCm/TheRock)**。7.14.0 是自 7.10.0 引入 Windows / pip 支持以来最重要的架构转折，标志着 ROCm 从"单体大包"走向"模块化生态"：
> - **精简内核**：Core SDK 只保留必要的运行时与开发组件；
> - **按需扩展**：面向 AI、数据科学、HPC 的可选领域 SDK；
> - **模块化安装**：只装工作流所需组件，减小体积、加快创新。
>
> 对**本项目的 pip / uv 安装流程无影响**（wheel 仍从 `repo.amd.com/rocm/whl/` 分发）。若你走 Linux 的 **apt / dnf 系统包**安装，则需注意包名与安装目录的变化（见 [2.5 节 · apt 安装（TheRock）](#25-备选apt-安装therock)）。详见 [TheRock 迁移指南](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html)。

> 💡 **平台建议**：Windows 已支持 ROCm 体验与推理验证，但 ROCm 生态工具链（如 rocminfo、amd-smi、多卡支持、容器化部署等）在 **Ubuntu** 上支持更完整。**建议使用 Ubuntu 24.04 作为主力开发环境**，Windows 可作为快速体验或轻量推理使用。

> ⚠️ **Windows 用户必读**：安装前务必确认你的 **Adrenalin Driver 版本** 和 **Windows 版本** 符合要求（见下方版本信息表），否则 ROCm 将无法正常运行。

---

## 版本信息

| 项目 | 要求 | 下载链接 |
|:---|:---|:---|
| ROCm | 7.14.0 (ROCm Core SDK / TheRock) | [官方安装页](https://rocm.docs.amd.com/en/latest/install/rocm.html) |
| PyTorch | 2.12.0 | 通过 uv 安装（见下文） |
| Python | 3.11 / 3.12 / 3.13 / 3.14 | 由 uv 自动管理 |
| **Windows 版本** | **11 25H2** | — |
| **Adrenalin Driver (Windows)** | **26.5.1** | [**⬇️ 下载 Adrenalin 26.5.1**](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-5-1.html#Downloads) |
| **Visual Studio 2022 (Windows)** | **Community，勾选「使用 C++ 的桌面开发」** | [**⬇️ 下载 VS 2022**](https://visualstudio.microsoft.com/zh-hans/downloads/) |
| Ubuntu | 24.04.4 (GA kernel 6.8) / 26.04 (GA kernel 7.0) | [Ubuntu Downloads](https://ubuntu.com/download/desktop) |

> ⚠️ **Ryzen APU 用户注意（Ubuntu 24.04）**：Ryzen APU（gfx1150 / 1151 / 1152 / 1153 / 1103）在 Ubuntu 24.04 上需要 **OEM 内核 6.14**：`sudo apt install linux-oem-24.04c`，安装后重启。

### AI 生态兼容性

ROCm 7.14.0 为主流深度学习框架和推理引擎提供了优化支持（较 7.13.0 全面升级）：

| 框架 / 引擎 | 支持版本 | 说明 |
|:---|:---|:---|
| PyTorch | 2.12.0 | Profiler 后端切换为 rocprofiler-sdk（替代 roctracer） |
| JAX | 0.10.0 | — |
| vLLM | 0.23.0 | 提供推理就绪镜像与包 |
| SGLang | 0.5.13 | — |
| TensorFlow | 2.21 | — |

> 💡 上述版本替代了 7.13.0 时期的 PyTorch 2.9.1 / JAX 0.8.2 / vLLM 0.19.1 / SGLang 0.5.9。vLLM 镜像按 GPU 架构分发，详见 [vLLM inference and serving](https://rocm.docs.amd.com/en/latest/ai-inference/vllm.html)。

---

## 目录

- [GPU 架构对照表（独立文件）](/zh/00-environment/rocm-gpu-architecture-table)
- [一、Windows 安装](#一windows-11-安装)
- [二、Ubuntu 安装](#二ubuntu-2404-安装)
  - [2.5 备选：apt 安装（TheRock）](#25-备选apt-安装therock)
- [三、校验安装](#三校验安装)
- [四、卸载 ROCm](#四卸载-rocm)
- [五、切换其他 GPU 架构](#五切换其他-gpu-架构)

---

## 一、Windows 11 安装

> 以 **Ryzen AI Max+ 395（gfx1151）** 为例。
>
> 📖 官方文档：[Install ROCm on Windows](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&gpu=max-395&os=windows&windows-ver=11&gfx=gfx1151&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.12.0&i=pip&gpu=max-395&gfx=gfx1151)

### 1.1 前置条件检查

| ✅ 检查项 | 要求 |
|:---|:---|
| **Windows 版本** | **必须 Windows 11 25H2**（设置 → 系统 → 关于 查看） |
| **Adrenalin 驱动** | **必须 26.5.1**（[⬇️ 下载](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-5-1.html#Downloads)） |
| **Visual Studio 2022**（可选） | Community 版即可，安装时勾选「使用 C++ 的桌面开发」（[⬇️ 下载](https://visualstudio.microsoft.com/zh-hans/downloads/)）。AMD Quark 等需要编译自定义算子时必需 |

<div align='center'>
    <img src="../../public/images/00-environment/visual_studil_c++_desktop_installer.png" alt="Visual Studio 安装勾选「使用 C++ 的桌面开发」" width="90%">
</div>

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
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,device-gfx1151]==7.14.0"

# 安装 PyTorch
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision[device-gfx1151]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0"

# 安装其他项目依赖（如有 requirements.txt）
uv pip install -r requirements.txt
```

> ⚠️ 不要将 ROCm DLL 复制到 System32，否则会引起冲突。
>
> 💡 **7.14.0 新语法**：wheel 已统一为多架构源 `whl-multi-arch/`，通过 `[device-gfxXXXX]` extras 指定你的 GPU 架构（不再用分架构 `--index-url`）。上述 `gfx1151` 对应 **Ryzen AI Max 系列**（395/390/385）。其他 GPU 只需替换 extras 中的架构标签：
>
> | 你的 GPU | device extras 标签 |
> |:---|:---|
> | Ryzen AI 9 HX (PRO) 475 / 375 等 | `device-gfx1150` |
> | Ryzen AI 7 (PRO) 450 / 350 等 | `device-gfx1152` |
> | Ryzen AI 7 445 / AI 5 435（7.14.0 新增） | `device-gfx1153` |
> | Radeon RX 9070 XT / 9070 GRE / AI PRO R9700S | `device-gfx1201` |
> | Radeon RX 9060 XT / 9060 XT LP / 9060 | `device-gfx1200` |
> | Radeon RX 7900 XTX / PRO W7900 | `device-gfx1100` |
> | Instinct MI300X / MI325X | `device-gfx942` |
> | 全部架构（体积大，兼容性最广） | `device-all` |
>
> 例如 gfx1150：`"rocm[libraries,device-gfx1150]==7.14.0"` 与 `"torch[device-gfx1150]==2.12.0+rocm7.14.0"`。
>
> 完整对照请查阅 [GPU 架构对照表](/zh/00-environment/rocm-gpu-architecture-table) 或 [官方兼容性矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)。

> 🚀 **国内加速提示**：对于非 ROCm 的普通 PyPI 包，可配置镜像源加速下载：
> ```bash
> # 全局配置清华镜像（仅影响未指定 --index-url 的包）
> uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 二、Ubuntu 24.04 安装

> 以 **Ryzen AI Max+ PRO 395（gfx1151）** 为例。
>
> 📖 官方文档：[Install ROCm on Ubuntu](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&gpu=max-395&os=ubuntu&os-version=24.04&gfx=gfx1151&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=linux&pytorch-ver=2.12.0&i=pip&gpu=max-395&gfx=gfx1151)

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

# 安装 ROCm 运行时 + 库（gfx1151 = Ryzen AI Max+ 395/390/385）
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,device-gfx1151]==7.14.0"

# 安装 PyTorch
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision[device-gfx1151]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0"

# 安装其他项目依赖（如有 requirements.txt）
uv pip install -r requirements.txt
```

> 💡 其他 GPU 只需替换 extras 中的架构标签（如 `device-gfx1150`、`device-gfx942`、`device-all`），对照表见 [Windows 1.5 节](#15-安装-rocm--pytorch) 或 [GPU 架构对照表](/zh/00-environment/rocm-gpu-architecture-table)。

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

### 2.5 备选：apt 安装（TheRock）

> 💡 如果你不用 pip / uv，而是想通过 **系统包管理器**（apt）做**系统级安装**，7.14.0 引入了 TheRock 打包体系，包名与安装目录发生变化，需特别留意。

| 变化项 | ROCm Core SDK 7.14.0 | ROCm Legacy（7.2 及之前） |
|:---|:---|:---|
| 安装目录 | `/opt/rocm/core` | `/opt/rocm/` |
| 包名前缀 | `amdrocm-*`（如 `amdrocm-blas`） | `rocm-*` / `roc*` / `hip*` |
| 共享扩展目录 | `/opt/rocm/extras-7/` | 无 |
| 包合并 | hipBLAS + rocBLAS → `amdrocm-blas` 等 | 分散为独立包 |

```bash
sudo apt update
sudo apt install sudo wget gpg
# 依据官方安装页添加 amdrocm 仓库后：

# 安装全部 GPU 架构（体积较大，兼容性最广）
sudo apt install amdrocm-core-sdk7.14

# 或按具体架构安装（体积更小，需已知你的 GPU 架构，例：gfx110x）
sudo apt install amdrocm-core-sdk7.14-gfx110x
```

> ✅ **兼容性说明**：7.14.0 与 **ROCm 7.2 legacy 保持 ABI/API 兼容，无需重新编译**。apt 安装时 `amdrocm` 元包会通过 `update-alternatives` 为 `/opt/rocm/bin`、`/opt/rocm/lib` 等提供向后兼容软链接；tarball 安装则需自行更新 `PATH` / `LD_LIBRARY_PATH` / `ROCM_PATH` 指向 `/opt/rocm/core`。
>
> ⚠️ **注意**：`amd-smi` 已替代已被移除的 `rocm-smi`；ASAN 包在 7.14.0 暂不提供，计划后续版本补齐。
>
> 📖 完整包名对照与迁移细节见 [TheRock 迁移指南](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html) 与 [官方安装页](https://rocm.docs.amd.com/en/latest/install/rocm.html)。

---

## 三、校验安装

### 3.1 PyTorch 检测（Windows / Linux 通用）

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

期望输出：

```
PyTorch: 2.12.0+rocm7.14.0
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
amd-smi monitor   # ROCm SMI 已在 7.14.0 移除，统一使用 amd-smi
hipinfo           # pip 安装方式可用
```

### 3.4 常见问题

| 现象 | 原因 | 解决 |
|:---|:---|:---|
| `torch.cuda.is_available()` = `False` | 驱动版本不匹配 | Windows：确认 [Adrenalin 26.5.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-5-1.html#Downloads)；Linux：确认 inbox / OEM kernel（Ryzen APU 需 `linux-oem-24.04c`） |
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

7.14.0 起，wheel 统一从 `https://repo.amd.com/rocm/whl-multi-arch/` 分发，通过 `[device-gfxXXXX]` extras 指定架构。只需将安装命令中的架构标签替换为对应值：

| GPU 示例 | LLVM Target | device extras 标签 |
|:---|:---|:---|
| MI355X / MI350X / MI350P | gfx950 | `device-gfx950` |
| MI300X / MI325X | gfx942 | `device-gfx942` |
| RX 9070 XT / 9070 GRE / AI PRO R9700S | gfx1201 | `device-gfx1201` |
| RX 9060 XT / 9060 XT LP / 9060 | gfx1200 | `device-gfx1200` |
| RX 7900 XTX / PRO W7900 | gfx1100 | `device-gfx1100` |
| Radeon PRO W6800 / V620 | gfx1030 | `device-gfx1030` |
| Ryzen AI Max 395 | gfx1151 | `device-gfx1151` |
| Ryzen AI PRO 400 / AI 9 HX 475 | gfx1150 | `device-gfx1150` |
| Ryzen AI 200 PRO / AI 7 350 | gfx1152 | `device-gfx1152` |
| Ryzen AI 7 445 / AI 5 435（7.14.0 新增） | gfx1153 | `device-gfx1153` |
| 全部架构 | — | `device-all` |

例如切换到 gfx942（MI300X）：

```bash
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,device-gfx942]==7.14.0"
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx942]==2.12.0+rocm7.14.0" "torchvision[device-gfx942]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0"
```

> 💡 若走 apt 系统包安装，则改用架构专属元包（如 `amdrocm-core-sdk7.14-gfx110x`），详见 [2.5 节](#25-备选apt-安装therock)。

完整对照表见 [GPU 架构对照表](/zh/00-environment/rocm-gpu-architecture-table)。

---

> 📖 完整官方文档：
> - [ROCm 7.14.0 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
> - [TheRock 迁移指南](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html)
> - [Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
> - [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html)
> - [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
> - [Install JAX](https://rocm.docs.amd.com/en/latest/frameworks/jax/install.html)
> - [vLLM Inference and Serving](https://rocm.docs.amd.com/en/latest/ai-inference/vllm.html)
> - [ComfyUI Image Generation](https://rocm.docs.amd.com/en/latest/ai-inference/comfyui.html)
> - [xDiT Diffusion Inference](https://rocm.docs.amd.com/en/latest/ai-inference/xdit.html)
> - [Inference Optimization](https://rocm.docs.amd.com/en/latest/ai-inference/optimization.html)
