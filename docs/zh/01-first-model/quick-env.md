# 1.2 ROCm 基础环境与推理运行环境完整配置（Windows 11 / Ubuntu 24.04 / 云算力）

在正式拉起四大主体模型（`Gemma 4 E4B`、`Qwen3.5-9B`、`Gemma 4 12B`、`Qwen3.6-35B-A3B`）之前，本节手把手带你一次性配齐 **ROCm 10.0.0（TheRock 模块化架构）+ `uv` + PyTorch 2.13.0 + 三大推理引擎前置环境**，并通过标准体检脚本确认 GPU 100% 就绪！

> 📖 如需了解 ROCm 10.0.0 底层打包变更、卸载细节或 XDNA2 NPU 深度内核参数，也可随时对照本仓库专项档案：[00-Environment 详细参考](../00-environment/index.md) · [ROCm 10.0.0 版本说明](../00-environment/rocm-10-0-0-release-notes.md) · [完整 GPU 架构对照表](../00-environment/rocm-gpu-architecture-table.md)。

---

## 一、环境版本基线与你的显卡架构标签速查

自 ROCm 转向 **TheRock** 模块化体系后，Python 虚拟环境可以直接通过 `uv pip` 按显卡架构标签（`[device-gfxXXXX]`）一键装入 ROCm 运行时与 PyTorch，无需像早年那样手动折腾复杂的系统级依赖！

### 1.1 推荐软硬件基线表

| 组件项目 | 标准要求版本 | 关键避坑提示 |
|:---|:---|:---|
| **ROCm 基线** | **ROCm 10.0.0**（Core SDK / TheRock） | 新版 wheel 官方源统一为 `https://stable.repo.amd.com/rocm/whl-next/` |
| **Python 与包管理** | **Python 3.12** + **`uv`** | `uv` 比传统 `pip` 快 10～100 倍，且内置 Python 版本自动下载管理 |
| **PyTorch 版本** | **`2.13.0+rocm10.0.0`** | 后续第 5 章中文 LoRA 微调与第 6 章 `AMD Quark` 量化均依赖此环境 |
| **Windows 11 要求** | **Windows 11 25H2** + **Adrenalin 驱动 `26.8.1`** | 驱动版本过低会导致 GPU 无法识别（[官方驱动下载说明](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads)） |
| **Ubuntu 要求** | **Ubuntu 24.04 LTS**（或 26.04） | **Ryzen APU 用户必看**：Ubuntu 24.04 需升级 OEM 6.14 内核（见下文 3.1 节） |

### 1.2 10 秒查出你的 GPU `device-gfxXXXX` 安装标签

安装 PyTorch ROCm wheel 时只需根据你的设备型号替换对应的架构标签：

| 你的 AMD 显卡 / 处理器系列 | 对应的安装标签 (`extras`) | 硬件档位参考 |
|:---|:---|:---|
| **Radeon RX 7600 系列** | `device-gfx1102` | 🟢 入门档 |
| **Radeon RX 7800 XT / 7700 XT** | `device-gfx1101` | 🔵 甜点档 |
| **Radeon RX 7900 XTX / 7900 XT / GRE / PRO W7900** | `device-gfx1100` | 🔵 甜点档 |
| **Radeon RX 9060 XT / 9060 / RX 9050**（RDNA 4） | `device-gfx1200` | 🟢 入门 / 🔵 甜点档 |
| **Radeon RX 9070 XT / 9070 GRE / AI PRO R9700**（RDNA 4） | `device-gfx1201` | 🔵 甜点档 |
| **Ryzen AI 9 HX 370 / 375 / 475**（Radeon 890M / 880M） | `device-gfx1150` | 🟢 入门档 |
| **Ryzen AI 7 350 / 450**（Radeon 860M） | `device-gfx1152` | 🟢 入门档 |
| **Ryzen AI Max+ 395 / Max 390 / 385**（Radeon 8060S / 8050S） | **`device-gfx1151`** | 🟣 旗舰档 |
| **Instinct MI300X / MI325X**（云算力常见） | `device-gfx942` | ☁️ 云端兜底档 |
| *拿不准型号时的万能兜底（体积较大、兼容最广）* | `device-all` | 全系通用 |

---

## 二、通道 A：Windows 11 本地环境完整配置（独显台式机 / 锐龙 AI 本）

### Step 1：前置排雷（90% 的 Windows 报错都出在这里！）

1. **升级系统与显卡驱动**：确认系统为 **Windows 11 25H2**（设置 → 系统 → 关于），并安装 **AMD Software: Adrenalin Edition 26.8.1**（或更新版本）。
2. **卸载旧版退役组件**：打开「控制面板 → 卸载程序」，如果以前装过旧版 **AMD HIP SDK**，请将其卸载干净（ROCm 10 起已统一由 Core SDK 接管，共存会引发 DLL 冲突；也切勿手动复制 DLL 到 `System32`）。
3. **关闭两项会拦截 ROCm 计算核的 Windows 安全开关**：
   - 关闭 **WDAG**：控制面板 → 程序和功能 → 启用或关闭 Windows 功能 → 取消勾选 `Microsoft Defender Application Guard`；
   - 关闭 **SAC（智能应用控制）**：设置 → 隐私和安全 → Windows 安全中心 → 应用和浏览器控制 → 智能应用控制设置 → 设为 **关闭**。
4. *（可选，第 6 章跑自定义编译算子时需用）*：安装 [Visual Studio 2022 Community](https://visualstudio.microsoft.com/zh-hans/downloads/) 并勾选「使用 C++ 的桌面开发」。

### Step 2：安装 `uv` 并创建非 C 盘工作目录与 Python 3.12 虚拟环境

打开 **PowerShell** 依次执行：

```powershell
# 1. 一行安装极速包管理器 uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version

# 2. 在非系统盘建立模型仓与沙盒工作目录（避免大权重塞爆 C 盘！）
mkdir D:\AI_Models
mkdir D:\AI_Workspace
cd D:\AI_Workspace

# 3. 自动下载 Python 3.12 并创建激活虚拟环境
uv python install 3.12
uv venv --python 3.12 .venv
.venv\Scripts\activate

# 4. 配置国内清华 PyPI 镜像（加速后续普通 Python 依赖包下载）
uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step 3：装入 ROCm 10.0.0 + PyTorch 2.13.0（请将 `device-gfx1151` 替换为你上面查到的标签！）

```powershell
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ `
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" `
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" `
  "torchaudio==2.11.0.2+rocm10.0.0"
```

> 💡 **懒人备选（ROCm CLI 一条命令托管）**：如果你不想自己查 `gfx` 标签，也可以直接安装官方新推出的 [ROCm CLI](https://github.com/ROCm/rocm-cli)：
> ```powershell
> irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
> rocm examine
> rocm install sdk
> ```

---

## 三、通道 B：Ubuntu 24.04 本地环境完整配置（主力开发强烈推荐！）

Ubuntu 24.04 拥有最完整的 ROCm 工具链支持（特别是 `vLLM` 容器、`lemonade-server` 系统服务与第 6 章 `AITER` / `Hyperloom` 算子加速）。

### Step 1：清理旧残留、内核升级与 GPU 设备权限配置

```bash
# 1. （仅针对装过旧版 ROCm 6.x/7.x 的机器）清理旧包避免路径串台
sudo apt remove 'rocm*' 'amdrocm*' -y && sudo apt autoremove -y

# 2. 【Ryzen AI / APU 笔记本或迷你主机必做】安装 OEM 6.14 内核（独显台式机可跳过）
sudo apt update
sudo apt install -y linux-oem-24.04c

# 3. 安装基础运行库并将当前用户加入 render, video 硬件访问组
sudo apt install -y libatomic1 libquadmath0 build-essential git curl wget jq pciutils
sudo usermod -a -G render,video "$LOGNAME"

# 若安装了新内核或刚修改用户组，请重启一次使权限与 /dev/kfd 生效
# sudo reboot
```

重启后验证底层 GPU 设备节点正常暴露：

```bash
ls -l /dev/kfd /dev/dri
```

### Step 2：使用 `uv` 安装 Python 3.12 与 ROCm 10 PyTorch

```bash
# 1. 安装 uv 并创建工作区与虚拟环境
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 2>/dev/null || true

mkdir -p ~/ai_models ~/ai_workspace
cd ~/ai_workspace
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. 配置清华源加速普通包，并从 AMD 官方源安装对应架构的 PyTorch 2.13.0
# （请将下方 device-gfx1151 替换为你显卡的标签，如 RX 7900 填 device-gfx1100，RX 9070 填 device-gfx1201）
uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" \
  "torchaudio==2.11.0.2+rocm10.0.0"
```

> 💡 **Ubuntu 一键托管备选**：同样支持 `curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh && rocm examine && rocm install sdk`。

### Step 3：按需准备三大标准推理引擎的专属前置组件

我们在下一节（1.3～1.5）提供了三大标准推理引擎，你可以根据打算使用的引擎完成前置安装：

1. **若准备用 `llama-server`（最轻量通用）**：无需额外系统服务，下一节直接下载预编译压缩包解压即跑。
2. **若准备用 `lemonade-server`（Ryzen AI / 端侧统一内存强烈推荐）**：
   ```bash
   sudo add-apt-repository -y ppa:lemonade-team/stable
   sudo apt update
   sudo apt install -y lemonade-server
   lemonade --version
   # 安装通用 Vulkan GGUF 后端（兼容性最稳）
   lemonade backends install llamacpp:vulkan
   lemonade status
   ```
   *(注：若你的 Ryzen AI 电脑还想单独跑原生 XDNA2 NPU 路径，可补充阅读 [FastFlowLM NPU 驱动与 IOMMU 配置](../00-environment/fastflowlm.md))*。
3. **若准备用 `vLLM`（16G+ 独显或云端高并发首选）**：确保已安装 Docker 并将当前用户加入 `docker` 组（`sudo usermod -aG docker $USER`），提前拉取官方 ROCm 验证镜像：
   ```bash
   docker pull vllm/vllm-openai-rocm:latest
   # 或拉取 ROCm 10.0.0 固定版本验证镜像：
   # docker pull rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
   ```

---

## 四、通道 C：手头暂无 AMD 显卡？免费云算力 + SSH 端口隧道无缝配置

如果你目前手头只有轻薄办公本或 Mac，完全不影响跟学全书：
1. 先按 **[AMD Radeon Cloud 使用指南](../cloud/amd-radeon-cloud.md)**（或 **[AUP Learning Cloud 指南](../cloud/aup-learning-cloud.md)**）开通免费云端工作区（云镜像通常已预装好 Ubuntu + ROCm 驱动，打开终端按上面 Ubuntu Step 2 建个 `uv` 环境即可）。
2. **🌟 本地桌面工具直连云卡秘籍（SSH 端口转发）**：在云端启动模型服务后，在你自己笔记本终端执行：
   ```bash
   ssh -L 8080:127.0.0.1:8080 -L 8000:127.0.0.1:8000 -L 13305:127.0.0.1:13305 <云主机用户名>@<云主机IP>
   ```
   这样你本机桌面安装的 **WorkBuddy** 和 **`Qoder`** 依然直接连 `http://127.0.0.1:8080/v1`，就能直接调用云端大显存 AMD GPU 处理你本机的 PPT 和 Excel 文件！

---

## 五、🎯 环境配置终极三连验收（必跑体检！）

无论你使用的是 Windows 11、Ubuntu 还是云平台，激活 `.venv` 虚拟环境后，运行下面这段 **10 秒体检代码**：

```bash
python - <<'PY'
import torch

print("1. PyTorch 版本:", torch.__version__)
print("2. ROCm GPU 是否可用:", torch.cuda.is_available())
assert torch.cuda.is_available(), "❌ 未检测到可用 GPU，请检查下方常见排错表！"

print("3. 识别到的设备名称:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x.T
print("4. BF16 矩阵乘冒烟计算成功! 输出均值:", float(y.mean()))
PY
```

> 💡 **为什么代码里写的是 `torch.cuda.is_available()`？**  
> 因为 ROCm 通过 HIP 层直接兼容了 PyTorch 的 `cuda` 接口语义！这样市面上绝大多数开源大模型代码（如 `LLaMA-Factory`、`Transformers`、`vLLM`）一行代码不用改就能直接在你的 AMD 显卡上跑起来！

### ❓ 新手环境配置高频排错速查

| 报错现象 | 99% 的原因 | 一行解决指令 / 方案 |
|:---|:---|:---|
| `torch.cuda.is_available()` 返回 `False`（Ubuntu） | 当前用户未加入 `render,video` 组，或 Ryzen APU 未装 6.14 OEM 内核 | 执行 `groups` 检查是否有 `render video`；缺失则执行 `sudo usermod -a -G render,video $LOGNAME` 并重启 |
| `torch.cuda.is_available()` 返回 `False`（Windows） | 显卡驱动低于 `26.8.1`，或未关闭 WDAG / SAC，或装成了 CPU 版 `torch` | 核对 Adrenalin 26.8.1 驱动，确认 `uv pip install` 带上了 `--index-url https://stable.repo.amd.com/rocm/whl-next/` |
| `HIP error: invalid device function` | 安装的 `[device-gfxXXXX]` 标签与你真实的显卡代号不匹配 | 对照本页 1.2 节表格换回正确的 `device-gfxXXXX` 重装，或改装通用包 `torch[device-all]` |

---

看到 `4. BF16 矩阵乘冒烟计算成功!` 后，你的 AMD AI 地基已经稳如泰山！马上进入 **[1.3～1.5 三大标准服务引擎极速部署（拉起四大主体模型）](./three-engines-deploy.md)**！
