<div align=center>
  <h1>00-Environment</h1>
  <div align='center'>

  [![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

  </div>
  <strong>🛠️ ROCm 基础环境安装与配置</strong>
</div>

<div align="center">

*统一环境基线 · ROCm 10.0.0（TheRock + ROCm.AI）· 所有后续章节的前置依赖*

[返回主页](/zh/) | [English](/00-environment/)

</div>

## 简介

&emsp;&emsp;本章节是整个 **hello-rocm** 项目的环境基线参考。统一以 **ROCm 10.0.0**（ROCm Core SDK，2026-08-26 发布）为目标版本，覆盖 Windows 和 Ubuntu 双平台的安装、校验与卸载流程。

&emsp;&emsp;后续所有章节（01-Deploy、02-Fine-tune 等）的环境准备均以本章为基准。如需使用其他 ROCm 版本或其他 GPU 架构，请参考 [GPU 架构对照表](/zh/00-environment/rocm-gpu-architecture-table) 进行对应替换。10.0.0 相对 7.14.0 的差异、以及 AMD Skills / Hyperloom / ROCm CLI，见 [ROCm 10.0.0 版本说明](/zh/00-environment/rocm-10-0-0-release-notes)。

> 🚀 **大版本：ROCm 10.0.0 建立在 [TheRock](https://github.com/ROCm/TheRock) 之上，并首次带上 ROCm.AI**。7.14.0 完成了从「单体大包」到模块化 Core SDK 的转折；10.0.0 把安装、验证、服务和优化收成三个新入口：
> - **AMD Skills**：把 AMD 官方优化知识做进 Claude / Cursor / Codex；
> - **Hyperloom**：开源自动优化引擎，自己做剖析、找瓶颈、改内核、调参数；
> - **ROCm CLI**：装环境、验证、部署、管理，一条命令走完。
>
> pip / uv 流程仍然可用，但 **wheel 索引已从 `repo.amd.com/rocm/whl-multi-arch/` 换成 `https://stable.repo.amd.com/rocm/whl-next/`**。Linux apt / dnf 仓库也收到 [stable.repo.amd.com](https://stable.repo.amd.com)。详见 [TheRock 迁移指南](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html) 与 [官方文档 latest](https://rocm.docs.amd.com/en/latest/)。

> 💡 **平台建议**：Windows 已支持 ROCm 体验与推理验证，但 ROCm 生态工具链（如 rocminfo、amd-smi、多卡支持、容器化部署等）在 **Ubuntu** 上支持更完整。**建议使用 Ubuntu 24.04 作为主力开发环境**，Windows 可作为快速体验或轻量推理使用。

> ⚠️ **Windows 用户必读**：安装前务必确认你的 **Adrenalin Driver 版本** 和 **Windows 版本** 符合要求（见下方版本信息表），否则 ROCm 将无法正常运行。

---

## 版本信息

| 项目 | 要求 | 下载链接 |
|:---|:---|:---|
| ROCm | 10.0.0 (ROCm Core SDK / TheRock) | [官方安装页](https://rocm.docs.amd.com/en/latest/install/rocm.html) |
| PyTorch | 2.13.0 | 通过 uv 安装（见下文） |
| Python | 3.11 / 3.12 / 3.13 / 3.14 | 由 uv 自动管理 |
| **Windows 版本** | **11 25H2** | — |
| **Adrenalin Driver (Windows)** | **26.8.1** | [**⬇️ 下载 Adrenalin 26.8.1**](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads) |
| **Visual Studio 2022 (Windows)** | **Community，勾选「使用 C++ 的桌面开发」** | [**⬇️ 下载 VS 2022**](https://visualstudio.microsoft.com/zh-hans/downloads/) |
| Ubuntu | 24.04.4 (GA kernel 6.8) / 26.04 (GA kernel 7.0) | [Ubuntu Downloads](https://ubuntu.com/download/desktop) |

> ⚠️ **Ryzen APU 用户注意（Ubuntu 24.04）**：Ryzen APU（gfx1150 / 1151 / 1152 / 1153 / 1103）在 Ubuntu 24.04 上需要 **OEM 内核 6.14**：`sudo apt install linux-oem-24.04c`，安装后重启。

### AI 生态兼容性

ROCm 10.0.0 为主流深度学习框架和推理引擎提供了优化支持（较 7.14.0 全面升级）：

| 框架 / 引擎 | 支持版本 | 说明 |
|:---|:---|:---|
| PyTorch | 2.13.0 | 另支持 2.12.0 / 2.11.0；Windows 验证版本为 2.13.0 |
| JAX | 0.11.0 | 另支持 0.10.2 / 0.10.0 |
| vLLM | 0.27.0 | 官方镜像按发行版分发；gfx1151 等消费级 / APU 已覆盖 |
| SGLang | 0.5.15 | Instinct / 部分 Radeon |
| TensorFlow | 2.21 | 另支持 2.20 / 2.19.1 |

> 💡 上述版本替代了 7.14.0 时期的 PyTorch 2.12.0 / JAX 0.10.0 / vLLM 0.23.0 / SGLang 0.5.13。vLLM 镜像见 [vLLM inference](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)。完整差异见 [ROCm 10.0.0 版本说明](/zh/00-environment/rocm-10-0-0-release-notes)。

---

## 目录

- [ROCm 10.0.0 版本说明](/zh/00-environment/rocm-10-0-0-release-notes)
- [GPU 架构对照表（独立文件）](/zh/00-environment/rocm-gpu-architecture-table)
- [一、Windows 安装](#一windows-11-安装)
- [二、Ubuntu 安装](#二ubuntu-2404-安装)
  - [2.5 备选：apt 安装（TheRock）](#25-备选apt-安装therock)
  - [2.6 备选：ROCm CLI 一条命令](#26-备选rocm-cli-一条命令)
- [三、校验安装](#三校验安装)
- [四、卸载 ROCm](#四卸载-rocm)
- [五、切换其他 GPU 架构](#五切换其他-gpu-架构)

---

## 一、Windows 11 安装

> 以 **Ryzen AI Max+ 395（gfx1151）** 为例。
>
> 📖 官方文档：[Install ROCm on Windows](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&gpu=max-395&os=windows&windows-ver=11&gfx=gfx1151&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.13.0&i=pip&gpu=max-395&gfx=gfx1151)

### 1.1 前置条件检查

| ✅ 检查项 | 要求 |
|:---|:---|
| **Windows 版本** | **必须 Windows 11 25H2**（设置 → 系统 → 关于 查看） |
| **Adrenalin 驱动** | **必须 26.8.1**（[⬇️ 下载](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads)） |
| **Visual Studio 2022**（可选） | Community 版即可，安装时勾选「使用 C++ 的桌面开发」（[⬇️ 下载](https://visualstudio.microsoft.com/zh-hans/downloads/)）。AMD Quark 等需要编译自定义算子时必需 |

<div align='center'>
    <img src="../../public/images/00-environment/visual_studil_c++_desktop_installer.png" alt="Visual Studio 安装勾选「使用 C++ 的桌面开发」" width="90%">
</div>

### 1.2 卸载冲突软件

- 控制面板 → 程序 → 卸载程序 → 移除所有 **HIP SDK** 相关项（10.0.0 起 HIP SDK 已退役，Windows 与 Linux 共用 ROCm Core SDK）

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

# 安装 PyTorch（wheel 已带 ROCm 运行时；gfx1151 = Ryzen AI Max+ 395/390/385）
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1151]==2.13.0+rocm10.0.0" "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"

# 安装其他项目依赖（如有 requirements.txt）
uv pip install -r requirements.txt
```

> ⚠️ 不要将 ROCm DLL 复制到 System32，否则会引起冲突。
>
> 💡 **10.0.0 索引**：wheel 从 `https://stable.repo.amd.com/rocm/whl-next/` 分发，仍通过 `[device-gfxXXXX]` extras 指定 GPU 架构。上述 `gfx1151` 对应 **Ryzen AI Max 系列**（395/390/385）。其他 GPU 只需替换 extras 中的架构标签：
>
> | 你的 GPU | device extras 标签 |
> |:---|:---|
> | Ryzen AI 9 HX (PRO) 475 / 375 等 | `device-gfx1150` |
> | Ryzen AI 7 (PRO) 450 / 350 等 | `device-gfx1152` |
> | Ryzen AI 7 445 / AI 5 435 | `device-gfx1153` |
> | Radeon RX 9070 XT / 9070 GRE / AI PRO R9700S | `device-gfx1201` |
> | Radeon RX 9060 XT / 9060 XT LP / 9060 / **RX 9050**（10.0.0 新增） | `device-gfx1200` |
> | Radeon RX 7900 XTX / PRO W7900 | `device-gfx1100` |
> | Instinct MI300X / MI325X | `device-gfx942` |
> | 全部架构（体积大，兼容性最广） | `device-all` |
>
> 例如 gfx1150：`"torch[device-gfx1150]==2.13.0+rocm10.0.0"`。
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
> 📖 官方文档：[Install ROCm on Ubuntu](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&gpu=max-395&os=ubuntu&os-version=24.04&gfx=gfx1151&i=pip) | [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=linux&pytorch-ver=2.13.0&i=pip&gpu=max-395&gfx=gfx1151)

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

# 安装 PyTorch（wheel 已带 ROCm 运行时；gfx1151 = Ryzen AI Max+ 395/390/385）
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1151]==2.13.0+rocm10.0.0" "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"

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

> 💡 如果你不用 pip / uv，而是想通过 **系统包管理器**（apt）做**系统级安装**，TheRock 打包体系从 7.14.0 延续到 10.0.0：包名仍是 `amdrocm-*`，仓库则收到 `stable.repo.amd.com`。

| 变化项 | ROCm Core SDK 10.0.0 | ROCm Legacy（7.2 及之前） |
|:---|:---|:---|
| 安装目录 | `/opt/rocm/core` | `/opt/rocm/` |
| 包名前缀 | `amdrocm-*`（如 `amdrocm-blas`） | `rocm-*` / `roc*` / `hip*` |
| 仓库 | `https://stable.repo.amd.com/rocm/core/packages/` | 旧 `repo.amd.com` 分发 |
| 库查找 | 已安装包嵌入 **RPATH**（优先于 `LD_LIBRARY_PATH`） | RUNPATH / 手工 `LD_LIBRARY_PATH` |

```bash
sudo apt update
sudo apt install sudo wget gpg
# 添加 10.0.0 稳定仓库（Ubuntu 24.04 示例）：
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
# 具体元包名以官方安装页当前选项为准
```

> ✅ **兼容性说明**：Core SDK 与 ROCm 7.2 legacy 保持 ABI/API 兼容路径；apt 安装时 `amdrocm` 元包仍通过 `update-alternatives` 提供 `/opt/rocm/bin`、`/opt/rocm/lib` 软链接。tarball 安装请把 `PATH` / `LD_LIBRARY_PATH` / `ROCM_PATH` 指到 `/opt/rocm/core`。
>
> ⚠️ **注意**：继续使用 `amd-smi`（`rocm-smi` 已在 7.14.0 移除）。**ASAN 包已在 10.0.0 随标准包提供**。已安装的 DEB / RPM / runfile 改为 RPATH，一台机器多版本时更不容易被旧 `LD_LIBRARY_PATH` 带跑。
>
> 📖 完整安装选择器见 [官方安装页](https://rocm.docs.amd.com/en/latest/install/rocm.html)。

### 2.6 备选：ROCm CLI 一条命令

不想手抄 pip extras 时，可以用 ROCm.AI 的 [ROCm CLI](https://github.com/ROCm/rocm-cli)（Technology Preview）：

```bash
curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh
rocm examine          # 看 GPU / 驱动 / 运行时
rocm install sdk      # 把 TheRock wheel 装进 CLI 托管环境
rocm serve qwen       # 拉起本地 OpenAI 兼容服务
```

Windows：

```powershell
irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
```

> 最低 Linux 为 Ubuntu 24.04。机器上如果已经有 7.14.0 的手建 `.venv`，`rocm examine` 会标成 unmanaged，再执行 `rocm install sdk` 会在旁边建托管运行时，不会直接覆盖。详见 [ROCm 10.0.0 版本说明](/zh/00-environment/rocm-10-0-0-release-notes)。

---

## 三、校验安装

### 3.1 PyTorch 检测（Windows / Linux 通用）

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

期望输出：

```
PyTorch: 2.13.0+rocm10.0.0
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
amd-smi monitor   # ROCm SMI 已在 7.14.0 移除，继续使用 amd-smi
hipinfo           # pip 安装方式可用
```

### 3.4 常见问题

| 现象 | 原因 | 解决 |
|:---|:---|:---|
| `torch.cuda.is_available()` = `False` | 驱动版本不匹配 | Windows：确认 [Adrenalin 26.8.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads)；Linux：确认 inbox / OEM kernel（Ryzen APU 需 `linux-oem-24.04c`） |
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

10.0.0 起，wheel 从 `https://stable.repo.amd.com/rocm/whl-next/` 分发，仍通过 `[device-gfxXXXX]` extras 指定架构。只需将安装命令中的架构标签替换为对应值：

| GPU 示例 | LLVM Target | device extras 标签 |
|:---|:---|:---|
| MI355X / MI350X / MI350P | gfx950 | `device-gfx950` |
| MI300X / MI325X | gfx942 | `device-gfx942` |
| RX 9070 XT / 9070 GRE / AI PRO R9700S | gfx1201 | `device-gfx1201` |
| RX 9060 XT / 9060 XT LP / 9060 / RX 9050（10.0.0 新增） | gfx1200 | `device-gfx1200` |
| RX 7900 XTX / PRO W7900 | gfx1100 | `device-gfx1100` |
| Radeon PRO W6800 / V620 | gfx1030 | `device-gfx1030` |
| Ryzen AI Max 395 | gfx1151 | `device-gfx1151` |
| Ryzen AI PRO 400 / AI 9 HX 475 | gfx1150 | `device-gfx1150` |
| Ryzen AI 200 PRO / AI 7 350 | gfx1152 | `device-gfx1152` |
| Ryzen AI 7 445 / AI 5 435 | gfx1153 | `device-gfx1153` |
| 全部架构 | — | `device-all` |

例如切换到 gfx942（MI300X）：

```bash
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx942]==2.13.0+rocm10.0.0" "torchvision[device-gfx942]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
```

> 💡 若走 apt 系统包安装，按 [2.5 节](#25-备选apt-安装therock) 添加 `stable.repo.amd.com` 仓库后，以 [官方安装页](https://rocm.docs.amd.com/en/latest/install/rocm.html) 当前元包名为准。

完整对照表见 [GPU 架构对照表](/zh/00-environment/rocm-gpu-architecture-table)。

---

> 📖 完整官方文档：
> - [ROCm 10.0.0 文档首页](https://rocm.docs.amd.com/en/latest/)
> - [ROCm 10.0.0 版本说明（本项目）](/zh/00-environment/rocm-10-0-0-release-notes)
> - [ROCm 10.0.0 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
> - [TheRock 迁移指南](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html)
> - [Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
> - [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html)
> - [Install PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
> - [vLLM Inference](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)
> - [AMD Skills](https://github.com/amd/skills)
> - [ROCm CLI](https://github.com/ROCm/rocm-cli)
> - [Hyperloom](https://rocm.docs.amd.com/projects/hyperloom/en/latest/)
