# 1.2 保姆级基础环境与 ROCm 10.0.0 配置指南（Windows 11 / Ubuntu 24.04 / 云算力）

> **本节目标**：根据你的操作系统（Windows 11、Ubuntu 24.04 或免费云算力），用最清爽、不污染系统、不塞爆 C 盘的方式配好 **AMD 显卡驱动、魔搭高速下载工具与 ROCm 10.0.0 + PyTorch 2.13 虚拟环境**，并跑通 4 步 GPU 矩阵乘体检脚本！

很多新手一听到“配深度学习环境”就头皮发麻，生怕把电脑搞蓝屏或者把 C 盘塞红。  
别担心！受益于 AMD 全新的模块化跨平台构建底座 **[`TheRock`](https://github.com/ROCm/TheRock)** 与 **ROCm 10.0.0**，现在安装 ROCm PyTorch 已经不需要像几年前那样折腾几十行的系统级环境变量——**一条 `uv pip install` 命令就能把对应显卡的 ROCm 运行库干净利落地装进一个独立的文件夹里**！

---

## 🚦 起步前先选路：「极速轻量档」还是「全栈满血档」？

为了不让你多装任何暂时用不上的巨型软件包，我们为你提供了两条完全兼容的递进通道：

| 环境档位 | 需要安装什么？ | 耗时与磁盘占用 | 能支撑本书哪些章节？ |
|:---|:---|:---|:---|
| **🏃 通道一：极速轻量档**<br/>（着急今天就玩上 PPT 的零基础同学） | 仅需 **最新 AMD 显卡驱动 + `uv`（含 `modelscope` 下载器）** | **~5 分钟**（不到 500 MB） | 直接畅玩 **1.3 `llama-server`**、**1.4 `lemonade-server`**、**第 2 章 WorkBuddy 大师 PPT / 自动填表 / `Qoder`**、**第 3～4 章知识库与 MCP**！ |
| **🚀 通道二：ROCm 10 全栈满血档**<br/>（强烈推荐一步到位！） | 在通道一基础上，一条命令装入 **`PyTorch 2.13 + ROCm 10.0.0`** | **~12 分钟**（约 6～9 GB） | **全书第 0～9 章 100% 通吃**！解锁 **1.5 `vLLM`**、**第 5 章中文 LoRA 微调** 与 **第 6 章 AMD Quark 量化及算子加速**！ |

*(建议所有硬盘空间大于 40 GB 的同学直接跟着下面走完 **通道一 + 通道二**，一劳永逸！)*

---

## 🛡️ 铁律前置：先把模型缓存改到非系统盘（拯救 C 盘！）

一颗大模型动辄 3 GB ～ 25 GB。如果不提前指定缓存目录，默认全都会偷偷往 `C:\Users\你的用户名\.cache` 里塞，不出三天 C 盘就会亮红灯！请先花 30 秒创建纯英文的工作目录与模型仓库：

### 🪟 Windows 11（打开 PowerShell 执行）

假设你的大容量硬盘是 `D:` 盘（如果只有 C 盘，改成 `C:\ai_workspace` 即可，**切记路径不要有中文和空格**）：

```powershell
# 1. 创建纯英文工作区与模型存放目录
New-Item -ItemType Directory -Force -Path "D:\ai_workspace\models", "D:\ai_workspace\venvs"

# 2. 永久设置魔搭 ModelScope 与 HuggingFace 缓存路径到 D 盘（写入当前用户环境变量）
[Environment]::SetEnvironmentVariable("MODELSCOPE_CACHE", "D:\ai_workspace\models\modelscope", "User")
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\ai_workspace\models\huggingface", "User")

# 让当前 PowerShell 窗口立刻生效
$env:MODELSCOPE_CACHE = "D:\ai_workspace\models\modelscope"
$env:HF_HOME = "D:\ai_workspace\models\huggingface"
```

### 🐧 Ubuntu 24.04（打开 Bash 终端执行）

```bash
# 1. 创建工作区与模型缓存目录（若有挂载大机械盘/固态盘如 /data，可替换为 /data/ai_workspace）
mkdir -p ~/ai_workspace/models/modelscope ~/ai_workspace/models/huggingface

# 2. 写入 ~/.bashrc 永久生效（同时开启国内 HuggingFace 镜像兜底）
cat << 'EOF' >> ~/.bashrc
export MODELSCOPE_CACHE="$HOME/ai_workspace/models/modelscope"
export HF_HOME="$HOME/ai_workspace/models/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
EOF
source ~/.bashrc
```

---

## 分支 A：🪟 Windows 11 环境完整配置步骤

### Step 1：系统版本与显卡驱动检查（最关键一步！）

1. **检查 Windows 版本**：按 `Win + R` 输入 `winver` 回车，推荐使用 **Windows 11 24H2 / 25H2**；
2. **安装配套 AMD 显卡驱动**：前往 [AMD 官方驱动下载页](https://www.amd.com/zh-cn/support/download/drivers.html) 安装最新 **AMD Software: Adrenalin Edition（ROCm 10 完整适配推荐 `26.8.1` 或更新版本）**，安装完成后重启电脑；
3. **两个避坑小细节**：
   - 如果你以前手动装过旧版独立的 `AMD HIP SDK 5.x/6.x`，建议在「设置 → 应用 → 已安装的应用」里把它卸载干净，避免旧版 DLL 抢占路径；
   - 如果后续导入 PyTorch wheel 时遇到 Windows 安全中心拦截未签名动态库报错，可在「Windows 安全中心 → 应用和浏览器控制 → 智能应用控制（SAC）」与「Windows Defender 应用程序防护（WDAG）」中按官方指引关闭拦截。

### Step 2：安装现代 Python 极速包管家 `uv` 并创建虚拟环境

传统 `conda` 又慢又臃肿，本课程统一采用每秒解析上百个包的现代神器 **[`uv`](https://github.com/astral-sh/uv)**（即使你电脑里原本没装 Python，`uv` 也会自动帮你下载干净隔离的 Python 3.12！）：

```powershell
# 1. 一键安装 uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 重新打开一个 PowerShell 窗口确保 uv 命令可用，进入工作区创建 Python 3.12 虚拟环境
cd D:\ai_workspace
uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1

# 2. 先用清华镜像装好魔搭高速下载器与 OpenAI 客户端（至此「通道一：极速轻量档」已就绪！）
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope openai
```

### Step 3（通道二全栈必装）：一条命令装入 ROCm 10.0.0 + PyTorch 2.13

请把下面命令里的 **`gfx1100`** 替换成你在 [第 0.3 节](../00-guide/03-hardware-and-cloud.md) 查到的自己显卡代号（例如 RX 9070 XT 填 `gfx1201`、RX 7800 XT 填 `gfx1101`、Ryzen AI Max+ 395 填 `gfx1151`、RX 7900 GRE/XTX 填 `gfx1100`）：

```powershell
# 注意：将两处 device-gfx1100 替换为你自己的显卡 gfx 代号！
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ `
  "torch[device-gfx1100]==2.13.0+rocm10.0.0" `
  "torchvision[device-gfx1100]==0.28.0+rocm10.0.0" `
  "torchaudio==2.11.0.2+rocm10.0.0"
```

---

## 分支 B：🐧 Ubuntu 24.04 LTS 环境完整配置步骤（原生满血性能！）

Linux 是 ROCm 发挥 100% 满血实力（尤其是 `vLLM`、`AITER` 融合算子与多卡训练）的大本营。

### Step 1：内核、用户组权限与基础驱动就绪

```bash
# 1. 将当前用户加入显卡计算权限组（漏掉这步会导致 PyTorch 找不到显卡！）
sudo usermod -a -G render,video "$LOGNAME"

# 2. 【仅针对 Ryzen AI 300 / Max+ 395 (gfx1150 / gfx1151) 锐龙 APU 用户】
# Ubuntu 24.04 默认老内核可能认不全最新 RDNA 3.5 核显与 NPU，强烈建议升级官方 OEM 6.14+ 内核：
# sudo apt update && sudo apt install -y linux-oem-24.04c

# 3. 确认 /dev/kfd 与 /dev/dri 设备节点存在（若新装系统尚未装 amdgpu 内核驱动，请参考官方 rocm.docs.amd.com 安装 amdgpu-dkms 并重启）
ls -l /dev/kfd /dev/dri/render*
```

> [!WARNING]
> 执行完 `sudo usermod -a -G render,video "$LOGNAME"` 后，请执行 `newgrp render`（或者注销重登一次终端），用 `groups` 命令确认输出中包含 `render` 和 `video`！

### Step 2：安装 `uv`、创建 Python 3.12 环境并装入 ROCm 10.0.0 PyTorch

同样记得把下面命令中的 **`gfx1151`** 替换为你自己的显卡代号（如 `gfx1100`、`gfx1101`、`gfx1201`、`gfx942`）：

```bash
# 1. 安装极速包管理工具 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

# 2. 创建并激活独立虚拟环境
cd ~/ai_workspace
uv venv .venv --python 3.12
source .venv/bin/activate

# 3. 安装魔搭下载器与 OpenAI SDK（清华源秒装）
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope openai

# 4. 按你的显卡架构安装 ROCm 10.0.0 PyTorch（将 gfx1151 替换为你自己的代号！）
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" \
  "torchaudio==2.11.0.2+rocm10.0.0"
```

---

## 分支 C：☁️ 免费云算力（AMD Radeon Cloud）3 分钟直通步骤

如果你使用 **[AMD Radeon Cloud](../cloud/amd-radeon-cloud.md)**：
1. 登录控制台，直接选择 **`ROCm + PyTorch`** 官方预置工作区模板启动实例（里面已经帮你预装好了驱动、ROCm 与 PyTorch！）；
2. 打开浏览器里的 JupyterLab Terminal（或通过本地 `ssh radeon-cloud` 登录），只需补装 `modelscope` 即可：
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope openai
   ```
3. 在你**本地自己电脑**上开一个终端，保持 SSH 端口隧道常驻（这样第 2 章本机桌面上的 WorkBuddy 就能直接调用云端 GPU）：
   ```bash
   ssh -N -L 8000:127.0.0.1:8000 -L 8080:127.0.0.1:8080 radeon-cloud
   ```

---

## ✅ 终极验收：运行 4 步「GPU 矩阵乘体检脚本」

在已激活 `.venv` 的终端（Windows PowerShell 或 Linux Bash 通用）中，直接复制运行下面这段体检命令。**注意：在 PyTorch 里，AMD ROCm 完美复用了 `torch.cuda` 接口命名，所以老代码一行不用改就能跑在 AMD 显卡上！**

```bash
python -c "
import time, torch
print('1. PyTorch 版本:', torch.__version__, '| HIP 运行时:', torch.version.hip)
assert torch.cuda.is_available(), '❌ 未检测到可用 GPU！请查看下方排错清单'
print('2. 成功识别显卡:', torch.cuda.get_device_name(0))
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f'3. 可用显存容量: {vram_gb:.1f} GB')
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.perf_counter()
c = a @ b
torch.cuda.synchronize()
print(f'4. ✅ BF16 4096x4096 矩阵乘冒烟计算成功！耗时 {(time.perf_counter()-t0)*1000:.2f} ms，输出形状 {tuple(c.shape)}')
"
```

🎉 **当你看到第 4 行输出 `✅ BF16 4096x4096 矩阵乘冒烟计算成功！` 时，恭喜你！你已经跨过了深度学习入门最难的一道门槛！**

---

## 🚑 新手高频环境报错 30 秒急救箱

| 报错现象 | 99% 的真实原因 | 一招解决命令 / 操作 |
|:---|:---|:---|
| `torch.cuda.is_available()` 返回 `False`（Linux） | 当前终端还没刷新 `render,video` 用户组权限，或系统同时有核显且未识别 | 运行 `groups` 检查是否有 `render`；若无，执行 `sudo usermod -a -G render,video "$LOGNAME"` 后**重启系统**。 |
| `torch.cuda.is_available()` 返回 `False`（Windows） | 驱动版本过老，或 `device-gfxXXXX` 填错了显卡型号 | 升级 Adrenalin 驱动至 `26.8.1+`，核对 [0.3 节表格](../00-guide/03-hardware-and-cloud.md) 重装匹配的 `torch[device-gfxXXXX]`。 |
| 网上老帖子教我设 `HSA_OVERRIDE_GFX_VERSION=11.0.0`，能设吗？ | 那是 ROCm 5/6 时代没有原生包时的临时强行伪装方案，极易引起算子崩溃（Segmentation Fault） | **在 ROCm 10.0.0 下强烈不建议乱设！** 请优先使用官方原生匹配的 `device-gfxXXXX` wheel 包。 |
| `uv pip install` 下载 wheel 断流超时 | 海外软件源偶发网络抖动 | 直接重跑同一条 `uv pip install` 命令（`uv` 支持断点缓存续传），普通包加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。 |

---

## 👉 下一步：任选一款引擎点亮你的第一颗大模型！

地基已经打得固若金汤！接下来在 **1.3 / 1.4 / 1.5** 三节中，请按你的电脑特点**先挑最顺手的一款引擎跑通即可**：
- 🪟 **Windows / Linux 独立显卡与通用电脑** 👉 首推 **[1.3 标准引擎一：`llama-server` 轻量全能部署实战（端口 `8080`）](./03-llama-server.md)**
- 💻 **Ryzen AI 锐龙笔记本 / Max+ 395 统一内存主机** 👉 首推 **[1.4 标准引擎二：`lemonade-server` 原生双擎部署实战（端口 `13305`）](./04-lemonade-server.md)**
- 🐧 **Ubuntu 大显存 / AMD Radeon Cloud 云算力** 👉 亦可直接体验 **[1.5 标准引擎三：`vLLM` 工业级高并发部署实战（端口 `8000`）](./05-vllm-server.md)**
