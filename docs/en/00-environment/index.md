<div align=center>
  <h1>00-Environment</h1>
  <div align='center'>

  [![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

  </div>
  <strong>🛠️ ROCm Environment Setup</strong>
</div>

<div align="center">

*Unified environment baseline · ROCm 7.13.0 · Prerequisite for all subsequent chapters*

[Back to Home](/) | [中文](./)

</div>

## Introduction

&emsp;&emsp;This chapter serves as the environment baseline for the entire **hello-rocm** project. It targets **ROCm 7.13.0** (Technology Preview, released 2026-05-15) and covers installation, verification, and uninstallation on both Windows and Ubuntu.

&emsp;&emsp;All subsequent chapters (01-Deploy, 02-Fine-tune, etc.) depend on this setup. To use a different ROCm version or GPU architecture, refer to the [GPU Architecture Reference Table](/00-environment/rocm-gpu-architecture-table) for substitutions.

> 💡 **Platform recommendation**: Windows supports ROCm for quick inference and experimentation, but the full ROCm toolchain (rocminfo, amd-smi, multi-GPU, containerized deployment, etc.) is best supported on **Ubuntu**. **We recommend Ubuntu 24.04 as the primary development environment**; Windows works well for lightweight inference and quick testing.

> ⚠️ **ROCm 7.13.0 is a Technology Preview release** — not suitable for production. For production use, see [ROCm 7.2 production stream](https://rocm.docs.amd.com/en/latest/).

> ⚠️ **Windows users must read**: Before installation, verify that your **Adrenalin Driver version** and **Windows version** meet the requirements (see version table below), or ROCm will not function.

---

## Version Requirements

| Item | Requirement | Download |
|:---|:---|:---|
| ROCm | 7.13.0 (Technology Preview) | [Official install page](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html) |
| PyTorch | 2.11.0 / 2.10.0 / 2.9.1 | Via uv (see below) |
| Python | 3.10 / 3.11 / 3.12 / 3.13 / 3.14 | Managed by uv |
| **Windows Version** | **11 25H2** | — |
| **Adrenalin Driver (Windows)** | **26.5.1** | [**⬇️ Download Adrenalin 26.5.1**](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-5-1.html#Downloads) |
| **Visual Studio 2022 (Windows)** | **Community, select "Desktop development with C++"** | [**⬇️ Download VS 2022**](https://visualstudio.microsoft.com/downloads/) |
| Ubuntu | 24.04.4 (GA kernel 6.8) / 26.04 (GA kernel 7.0) | [Ubuntu Downloads](https://ubuntu.com/download/desktop) |

### AI Ecosystem Compatibility

ROCm 7.13.0 provides optimized support for popular deep learning frameworks and AI inference engines:

| Framework / Engine | Supported Versions | Python Versions |
|:---|:---|:---|
| PyTorch | 2.11.0 / 2.10.0 / 2.9.1 | 3.14 / 3.13 / 3.12 / 3.11 / 3.10 |
| JAX | — | — |
| vLLM | 0.19.1 | 3.13 |

> 💡 vLLM 0.19.1 Docker images ship with PyTorch 2.10 + Python 3.13, distributed per GPU architecture (gfx942, gfx950, gfx1100-1201, gfx1150-1152). See [vLLM inference and serving](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/vllm.html).

---

## Table of Contents

- [GPU Architecture Reference Table (separate file)](/00-environment/rocm-gpu-architecture-table)
- [1. Windows Installation](#1-windows-11-installation)
- [2. Ubuntu Installation](#2-ubuntu-2404-installation)
- [3. Verify Installation](#3-verify-installation)
- [4. Uninstall ROCm](#4-uninstall-rocm)
- [5. Switching GPU Architectures](#5-switching-gpu-architectures)

---

## 1. Windows 11 Installation

> Example: **Ryzen AI Max+ 395 (gfx1151)**
>
> 📖 Official docs: [Install ROCm on Windows](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html?fam=ryzen&gpu=max-pro-395&os=windows&os-version=11_25h2&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/en/7.13.0-preview/rocm-for-ai/pytorch.html?fam=ryzen&gpu=max-pro-395&os=windows)

### 1.1 Prerequisites Check

| ✅ Check | Requirement |
|:---|:---|
| **Windows Version** | **Must be Windows 11 25H2** (Settings → System → About) |
| **Adrenalin Driver** | **Must be 26.5.1** ([⬇️ Download](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-5-1.html#Downloads)) |
| **Visual Studio 2022** (Optional) | Community edition, select "Desktop development with C++" ([⬇️ Download](https://visualstudio.microsoft.com/downloads/)). Required for AMD Quark or custom op compilation |

<div align='center'>
    <img src="../../public/images/00-environment/visual_studil_c++_desktop_installer.png" alt="Visual Studio installer — select Desktop development with C++" width="90%">
</div>

### 1.2 Remove Conflicting Software

- Control Panel → Programs → Uninstall a program → Remove all **HIP SDK** entries

### 1.3 Disable Windows Security Features

The following features interfere with ROCm and **must be disabled**:

- **WDAG**: Control Panel → Programs and Features → Turn Windows features on or off → Uncheck "Microsoft Defender Application Guard"
- **SAC**: Settings → Privacy & Security → Windows Security → App & browser control → Smart App Control settings → **Off**

### 1.4 Install uv (Python Package Manager)

This project uses [uv](https://docs.astral.sh/uv/) to manage Python environments and dependencies, replacing the traditional pip + venv workflow. uv is written in Rust and is 10-100x faster.

```powershell
# Windows install (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Or via winget
# winget install astral-sh.uv

# Verify
uv --version
```

> 📖 More install methods: [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)

### 1.5 Install ROCm + PyTorch

```powershell
# Install Python 3.12 (uv has built-in version management)
uv python install 3.12

# Create virtual environment
uv venv --python 3.12
.venv\Scripts\activate

# Install ROCm runtime + libraries (gfx1151 = Ryzen AI Max+ 395/390/385)
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]"

# Install PyTorch
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio

# Install other project dependencies (if requirements.txt exists)
uv pip install -r requirements.txt
```

> ⚠️ Do NOT copy ROCm DLLs to System32 — this causes conflicts.
>
> 💡 The `gfx1151` above corresponds to **Ryzen AI Max series** (395/390/385). For other GPUs, replace `--index-url`:
>
> | Your GPU | Replace with |
> |:---|:---|
> | Ryzen AI PRO 400 series (AI 9 HX PRO 475 etc.) | `https://repo.amd.com/rocm/whl/gfx1150/` |
> | Ryzen AI 200 PRO / AI 7 350 etc. (gfx1152) | `https://repo.amd.com/rocm/whl/gfx1152/` |
> | Radeon RX 9070 XT / 9070 GRE / AI PRO R9700S | `https://repo.amd.com/rocm/whl/gfx120X-all/` |
> | Radeon RX 9060 XT / 9060 XT LP / 9060 | `https://repo.amd.com/rocm/whl/gfx120X-all/` |
> | Radeon RX 7900 XTX / 7800 XT | `https://repo.amd.com/rocm/whl/gfx110X-all/` |
> | Instinct MI300X / MI325X | `https://repo.amd.com/rocm/whl/gfx94X-dcgpu/` |
>
> Full reference: [GPU Architecture Table](/00-environment/rocm-gpu-architecture-table) or [Official Compatibility Matrix](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html).

---

## 2. Ubuntu 24.04 Installation

> Example: **Ryzen AI Max+ PRO 395 (gfx1151)**
>
> 📖 Official docs: [Install ROCm on Ubuntu](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html?fam=ryzen&gpu=max-pro-395&os=ubuntu&os-version=24.04&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/en/7.13.0-preview/rocm-for-ai/pytorch.html?fam=ryzen&gpu=max-pro-395&os=linux)

### 2.1 Install uv and Dependencies

```bash
sudo apt install -y libatomic1 libquadmath0

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

### 2.2 Install ROCm + PyTorch (uv, recommended)

```bash
# Install Python 3.12
uv python install 3.12

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install ROCm (gfx1151 = Ryzen AI Max+ 395/390/385)
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]"

# Install PyTorch
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio

# Install other project dependencies (if requirements.txt exists)
uv pip install -r requirements.txt
```

> 💡 For other GPUs, replace `--index-url` — see [Section 1.5](#15-install-rocm--pytorch) or [GPU Architecture Table](/00-environment/rocm-gpu-architecture-table).

### 2.3 Alternative: One-Click Install Script

For a fully automated installation (kernel, driver, ROCm), use the project's install script:

```bash
git clone -b unified-installer https://github.com/amdjiahangpan/rocm-install-script.git
cd rocm-install-script
chmod +x install.sh
sudo ./install.sh
```

> 📖 Script details and options: [rocm-install-script (unified-installer branch)](https://github.com/amdjiahangpan/rocm-install-script/tree/unified-installer)

### 2.4 Configure GPU Access Permissions (Linux)

> 💡 This step can be done anytime after installation; takes effect after reboot.

```bash
sudo usermod -a -G render,video "$LOGNAME"
# Log out and back in, or reboot
```

---

## 3. Verify Installation

### 3.1 PyTorch Check (Windows / Linux)

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:

```
PyTorch: 2.11.0+rocm7.13.0
ROCm available: True
Device: AMD Radeon Graphics
```

> 💡 ROCm uses HIP to provide CUDA API compatibility, so `torch.cuda.is_available()` returning `True` is expected behavior.

### 3.2 Simple Computation Test

```python
import torch
x = torch.randn(3, 3, device='cuda')
y = torch.randn(3, 3, device='cuda')
print(x @ y)
```

### 3.3 Linux-only Tools

```bash
rocminfo | grep -E "Name:|Marketing Name:"
rocm-smi          # or: amd-smi monitor
hipinfo           # available with pip installation
```

### 3.4 Troubleshooting

| Symptom | Cause | Solution |
|:---|:---|:---|
| `torch.cuda.is_available()` = `False` | Driver version mismatch | Windows: confirm [Adrenalin 26.5.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-5-1.html#Downloads); Linux: confirm inbox kernel |
| `No GPU detected` (Linux) | Not in render/video group | `sudo usermod -a -G render,video $LOGNAME` + reboot |
| DLL load error (Windows) | SAC/WDAG not disabled | See [Section 1.3](#13-disable-windows-security-features) |

---

## 4. Uninstall ROCm

### Windows

Simply delete the `.venv` folder (via File Explorer, or in CMD):

```cmd
rmdir /s /q .venv
```

To uninstall Adrenalin driver: Control Panel → Programs → Uninstall a program → AMD Software

### Ubuntu

```bash
rm -rf .venv
```

---

## 5. Switching GPU Architectures

Simply replace the `--index-url` or apt package name with the corresponding value:

| GPU Example | LLVM Target | pip index URL |
|:---|:---|:---|
| MI355X / MI350X / MI350P | gfx950 | `https://repo.amd.com/rocm/whl/gfx950-dcgpu/` |
| MI300X / MI325X | gfx942 | `https://repo.amd.com/rocm/whl/gfx94X-dcgpu/` | 
| RX 9070 XT / 9070 GRE / AI PRO R9700S | gfx1201 | `https://repo.amd.com/rocm/whl/gfx120X-all/` |
| RX 9060 XT / 9060 XT LP / 9060 | gfx1200 | `https://repo.amd.com/rocm/whl/gfx120X-all/` |
| RX 7900 XTX / PRO W7900 | gfx1100 | `https://repo.amd.com/rocm/whl/gfx110X-all/` | 
| Radeon PRO W6800 / V620 | gfx1030 | `https://repo.amd.com/rocm/whl/gfx1030/` |
| Ryzen AI Max 395 | gfx1151 | `https://repo.amd.com/rocm/whl/gfx1151/` |
| Ryzen AI PRO 400 / AI 9 HX 475 | gfx1150 | `https://repo.amd.com/rocm/whl/gfx1150/` | 
| Ryzen AI 200 PRO / AI 7 350 | gfx1152 | `https://repo.amd.com/rocm/whl/gfx1152/` | 

Full reference: [GPU Architecture Table](/00-environment/rocm-gpu-architecture-table)

---

> 📖 Official documentation:
> - [ROCm 7.13.0 Release Notes](https://rocm.docs.amd.com/en/7.13.0-preview/about/release-notes.html)
> - [Compatibility Matrix](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html)
> - [Install ROCm](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html)
> - [Install PyTorch](https://rocm.docs.amd.com/en/7.13.0-preview/frameworks/pytorch/install.html)
> - [Install JAX](https://rocm.docs.amd.com/en/7.13.0-preview/frameworks/jax/install.html)
> - [vLLM Inference and Serving](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/vllm.html)
> - [ComfyUI Image Generation](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/comfyui.html)
> - [xDiT Diffusion Inference](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/xdit.html)
> - [Inference Optimization](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/optimization.html)
