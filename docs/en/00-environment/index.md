<div align=center>
  <h1>00-Environment</h1>
  <div align='center'>

  [![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

  </div>
  <strong>🛠️ ROCm Environment Setup</strong>
</div>

<div align="center">

*Unified environment baseline · ROCm 10.0.0 (TheRock + ROCm.AI) · Prerequisite for all subsequent chapters*

[Back to Home](/) | [中文](./)

</div>

## Introduction

&emsp;&emsp;This chapter serves as the environment baseline for the entire **hello-rocm** project. It targets **ROCm 10.0.0** (ROCm Core SDK, released 2026-08-26) and covers installation, verification, and uninstallation on both Windows and Ubuntu.

&emsp;&emsp;All subsequent chapters (01-Deploy, 02-Fine-tune, etc.) depend on this setup. To use a different ROCm version or GPU architecture, refer to the [GPU Architecture Reference Table](/00-environment/rocm-gpu-architecture-table) for substitutions. For 10.0.0 vs 7.14.0, plus AMD Skills / Hyperloom / ROCm CLI, see the [ROCm 10.0.0 release notes](/00-environment/rocm-10-0-0-release-notes).

> 🚀 **Major release: ROCm 10.0.0 is built on [TheRock](https://github.com/ROCm/TheRock) and ships ROCm.AI for the first time**. 7.14.0 finished the move from a monolithic bundle to a modular Core SDK. 10.0.0 folds install, validation, serving, and optimization into three new entry points:
> - **AMD Skills**: official AMD optimization knowledge inside Claude / Cursor / Codex;
> - **Hyperloom**: an open-source auto-optimizer that profiles, finds bottlenecks, rewrites kernels, and tunes parameters;
> - **ROCm CLI**: install, verify, deploy, and manage from one command surface.
>
> The pip / uv flow still works, but the **wheel index moved from `repo.amd.com/rocm/whl-multi-arch/` to `https://stable.repo.amd.com/rocm/whl-next/`**. Linux apt / dnf repos now live on [stable.repo.amd.com](https://stable.repo.amd.com). See the [TheRock transition guide](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html) and the [official latest docs](https://rocm.docs.amd.com/en/latest/).

> 💡 **Platform recommendation**: Windows supports ROCm for quick inference and experimentation, but the full ROCm toolchain (rocminfo, amd-smi, multi-GPU, containerized deployment, etc.) is best supported on **Ubuntu**. **We recommend Ubuntu 24.04 as the primary development environment**; Windows works well for lightweight inference and quick testing.

> ⚠️ **Windows users must read**: Before installation, verify that your **Adrenalin Driver version** and **Windows version** meet the requirements (see version table below), or ROCm will not function.

---

## Version Requirements

| Item | Requirement | Download |
|:---|:---|:---|
| ROCm | 10.0.0 (ROCm Core SDK / TheRock) | [Official install page](https://rocm.docs.amd.com/en/latest/install/rocm.html) |
| PyTorch | 2.13.0 | Via uv (see below) |
| Python | 3.11 / 3.12 / 3.13 / 3.14 | Managed by uv |
| **Windows Version** | **11 25H2** | — |
| **Adrenalin Driver (Windows)** | **26.8.1** | [**⬇️ Download Adrenalin 26.8.1**](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads) |
| **Visual Studio 2022 (Windows)** | **Community, select "Desktop development with C++"** | [**⬇️ Download VS 2022**](https://visualstudio.microsoft.com/downloads/) |
| Ubuntu | 24.04.4 (GA kernel 6.8) / 26.04 (GA kernel 7.0) | [Ubuntu Downloads](https://ubuntu.com/download/desktop) |

> ⚠️ **Ryzen APU users note (Ubuntu 24.04)**: Ryzen APUs (gfx1150 / 1151 / 1152 / 1153 / 1103) require the **OEM kernel 6.14** on Ubuntu 24.04: `sudo apt install linux-oem-24.04c`, then reboot.

### AI Ecosystem Compatibility

ROCm 10.0.0 provides optimized support for popular deep learning frameworks and AI inference engines (a full upgrade over 7.14.0):

| Framework / Engine | Supported Version | Notes |
|:---|:---|:---|
| PyTorch | 2.13.0 | Also 2.12.0 / 2.11.0; Windows is validated on 2.13.0 |
| JAX | 0.11.0 | Also 0.10.2 / 0.10.0 |
| vLLM | 0.27.0 | Official images cover gfx1151 and other client / APU targets |
| SGLang | 0.5.15 | Instinct and selected Radeon GPUs |
| TensorFlow | 2.21 | Also 2.20 / 2.19.1 |

> 💡 These versions replace the 7.14.0-era PyTorch 2.12.0 / JAX 0.10.0 / vLLM 0.23.0 / SGLang 0.5.13. See [vLLM inference](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html). Full delta: [ROCm 10.0.0 release notes](/00-environment/rocm-10-0-0-release-notes).

---

## Table of Contents

- [ROCm 10.0.0 Release Notes](/00-environment/rocm-10-0-0-release-notes)
- [GPU Architecture Reference Table (separate file)](/00-environment/rocm-gpu-architecture-table)
- [1. Windows Installation](#1-windows-11-installation)
- [2. Ubuntu Installation](#2-ubuntu-2404-installation)
  - [2.5 Alternative: apt Install (TheRock)](#25-alternative-apt-install-therock)
  - [2.6 Alternative: ROCm CLI](#26-alternative-rocm-cli)
- [3. Verify Installation](#3-verify-installation)
- [4. Uninstall ROCm](#4-uninstall-rocm)
- [5. Switching GPU Architectures](#5-switching-gpu-architectures)

---

## 1. Windows 11 Installation

> Example: **Ryzen AI Max+ 395 (gfx1151)**
>
> 📖 Official docs: [Install ROCm on Windows](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&gpu=max-395&os=windows&windows-ver=11&gfx=gfx1151&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.13.0&i=pip&gpu=max-395&gfx=gfx1151)

### 1.1 Prerequisites Check

| ✅ Check | Requirement |
|:---|:---|
| **Windows Version** | **Must be Windows 11 25H2** (Settings → System → About) |
| **Adrenalin Driver** | **Must be 26.8.1** ([⬇️ Download](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads)) |
| **Visual Studio 2022** (Optional) | Community edition, select "Desktop development with C++" ([⬇️ Download](https://visualstudio.microsoft.com/downloads/)). Required for AMD Quark or custom op compilation |

<div align='center'>
    <img src="../../public/images/00-environment/visual_studil_c++_desktop_installer.png" alt="Visual Studio installer — select Desktop development with C++" width="90%">
</div>

### 1.2 Remove Conflicting Software

- Control Panel → Programs → Uninstall a program → Remove all **HIP SDK** entries (HIP SDK is retired in 10.0.0; Windows and Linux now share the ROCm Core SDK)

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

# Install PyTorch (the wheel already includes the ROCm runtime; gfx1151 = Ryzen AI Max+ 395/390/385)
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1151]==2.13.0+rocm10.0.0" "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"

# Install other project dependencies (if requirements.txt exists)
uv pip install -r requirements.txt
```

> ⚠️ Do NOT copy ROCm DLLs to System32 — this causes conflicts.
>
> 💡 **10.0.0 index**: wheels come from `https://stable.repo.amd.com/rocm/whl-next/`, and you still select the GPU via the `[device-gfxXXXX]` extra. The `gfx1151` above is the **Ryzen AI Max series** (395/390/385). For other GPUs, swap the architecture tag:
>
> | Your GPU | device extras tag |
> |:---|:---|
> | Ryzen AI 9 HX (PRO) 475 / 375 etc. | `device-gfx1150` |
> | Ryzen AI 7 (PRO) 450 / 350 etc. | `device-gfx1152` |
> | Ryzen AI 7 445 / AI 5 435 | `device-gfx1153` |
> | Radeon RX 9070 XT / 9070 GRE / AI PRO R9700S | `device-gfx1201` |
> | Radeon RX 9060 XT / 9060 XT LP / 9060 / **RX 9050** (new in 10.0.0) | `device-gfx1200` |
> | Radeon RX 7900 XTX / PRO W7900 | `device-gfx1100` |
> | Instinct MI300X / MI325X | `device-gfx942` |
> | All architectures (larger, broadest compatibility) | `device-all` |
>
> For example, gfx1150: `"torch[device-gfx1150]==2.13.0+rocm10.0.0"`.
>
> Full reference: [GPU Architecture Table](/00-environment/rocm-gpu-architecture-table) or [Official Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

---

## 2. Ubuntu 24.04 Installation

> Example: **Ryzen AI Max+ PRO 395 (gfx1151)**
>
> 📖 Official docs: [Install ROCm on Ubuntu](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&gpu=max-395&os=ubuntu&os-version=24.04&gfx=gfx1151&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=linux&pytorch-ver=2.13.0&i=pip&gpu=max-395&gfx=gfx1151)

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

# Install PyTorch (the wheel already includes the ROCm runtime; gfx1151 = Ryzen AI Max+ 395/390/385)
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1151]==2.13.0+rocm10.0.0" "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"

# Install other project dependencies (if requirements.txt exists)
uv pip install -r requirements.txt
```

> 💡 For other GPUs, just swap the architecture tag in the extra (e.g. `device-gfx1150`, `device-gfx942`, `device-all`) — see [Section 1.5](#15-install-rocm--pytorch) or the [GPU Architecture Table](/00-environment/rocm-gpu-architecture-table).

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

### 2.5 Alternative: apt Install (TheRock)

> 💡 If you don't use pip / uv and prefer a **system-wide install** via the **system package manager** (apt), TheRock packaging continues from 7.14.0 into 10.0.0: package names stay `amdrocm-*`, and the repo moves to `stable.repo.amd.com`.

| Change | ROCm Core SDK 10.0.0 | ROCm Legacy (7.2 and earlier) |
|:---|:---|:---|
| Install directory | `/opt/rocm/core` | `/opt/rocm/` |
| Package prefix | `amdrocm-*` (e.g. `amdrocm-blas`) | `rocm-*` / `roc*` / `hip*` |
| Repository | `https://stable.repo.amd.com/rocm/core/packages/` | Older `repo.amd.com` layout |
| Library lookup | Installed packages embed **RPATH** (before `LD_LIBRARY_PATH`) | RUNPATH / manual `LD_LIBRARY_PATH` |

```bash
sudo apt update
sudo apt install sudo wget gpg
# Add the 10.0.0 stable repo (Ubuntu 24.04 example):
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources <<'EOF'
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
EOF

sudo apt update
# Use the meta-package name shown on the official install page
```

> ✅ **Compatibility**: the Core SDK keeps an ABI/API compatibility path with ROCm 7.2 legacy. With apt, the `amdrocm` meta package still configures `update-alternatives` for `/opt/rocm/bin` and `/opt/rocm/lib`. For tarball installs, point `PATH` / `LD_LIBRARY_PATH` / `ROCM_PATH` at `/opt/rocm/core`.
>
> ⚠️ **Note**: keep using `amd-smi` (`rocm-smi` was removed in 7.14.0). **ASAN packages ship with 10.0.0**. Installed DEB / RPM / runfile packages now use RPATH, so a multi-version machine is less likely to pick up the wrong libraries from `LD_LIBRARY_PATH`.
>
> 📖 Official selector: [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html).

### 2.6 Alternative: ROCm CLI

If you do not want to copy pip extras by hand, use the ROCm.AI [ROCm CLI](https://github.com/ROCm/rocm-cli) (Technology Preview):

```bash
curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh
rocm examine          # GPU / driver / runtime
rocm install sdk      # TheRock wheels into a CLI-managed environment
rocm serve qwen       # local OpenAI-compatible server
```

Windows:

```powershell
irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
```

> Minimum Linux is Ubuntu 24.04. If a 7.14.0 hand-built `.venv` is already present, `rocm examine` reports it as unmanaged and `rocm install sdk` creates a managed runtime beside it. Details: [ROCm 10.0.0 release notes](/00-environment/rocm-10-0-0-release-notes).

---

## 3. Verify Installation

### 3.1 PyTorch Check (Windows / Linux)

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:

```
PyTorch: 2.13.0+rocm10.0.0
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
amd-smi monitor   # ROCm SMI was removed in 7.14.0 — use amd-smi
hipinfo           # available with pip installation
```

### 3.4 Troubleshooting

| Symptom | Cause | Solution |
|:---|:---|:---|
| `torch.cuda.is_available()` = `False` | Driver version mismatch | Windows: confirm [Adrenalin 26.8.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads); Linux: confirm inbox / OEM kernel (Ryzen APUs need `linux-oem-24.04c`) |
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

Since 10.0.0, wheels are served from `https://stable.repo.amd.com/rocm/whl-next/`, and you still select the architecture via the `[device-gfxXXXX]` extra. Replace the architecture tag in the install command:

| GPU Example | LLVM Target | device extras tag |
|:---|:---|:---|
| MI355X / MI350X / MI350P | gfx950 | `device-gfx950` |
| MI300X / MI325X | gfx942 | `device-gfx942` |
| RX 9070 XT / 9070 GRE / AI PRO R9700S | gfx1201 | `device-gfx1201` |
| RX 9060 XT / 9060 XT LP / 9060 / RX 9050 (new in 10.0.0) | gfx1200 | `device-gfx1200` |
| RX 7900 XTX / PRO W7900 | gfx1100 | `device-gfx1100` |
| Radeon PRO W6800 / V620 | gfx1030 | `device-gfx1030` |
| Ryzen AI Max 395 | gfx1151 | `device-gfx1151` |
| Ryzen AI PRO 400 / AI 9 HX 475 | gfx1150 | `device-gfx1150` |
| Ryzen AI 200 PRO / AI 7 350 | gfx1152 | `device-gfx1152` |
| Ryzen AI 7 445 / AI 5 435 | gfx1153 | `device-gfx1153` |
| All architectures | — | `device-all` |

For example, to switch to gfx942 (MI300X):

```bash
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx942]==2.13.0+rocm10.0.0" "torchvision[device-gfx942]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
```

> 💡 For the apt path, add the `stable.repo.amd.com` repo from [Section 2.5](#25-alternative-apt-install-therock), then use the meta-package name shown on the [official install page](https://rocm.docs.amd.com/en/latest/install/rocm.html).

Full reference: [GPU Architecture Table](/00-environment/rocm-gpu-architecture-table)

---

> 📖 Official documentation:
> - [ROCm 10.0.0 docs home](https://rocm.docs.amd.com/en/latest/)
> - [ROCm 10.0.0 release notes (this project)](/00-environment/rocm-10-0-0-release-notes)
> - [ROCm 10.0.0 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
> - [TheRock Transition Guide](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html)
> - [Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
> - [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html)
> - [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
> - [vLLM Inference](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)
> - [AMD Skills](https://github.com/amd/skills)
> - [ROCm CLI](https://github.com/ROCm/rocm-cli)
> - [Hyperloom](https://rocm.docs.amd.com/projects/hyperloom/en/latest/)
