# 0.3 3 分钟硬件体检、四大主体模型选型与免费云算力指南

> **本节目标**：花 3 分钟查清你手头电脑的显卡型号、显存/内存容量与 `gfx` 架构代号，对号入座选定陪伴你跑完全书的 **主体大模型**；如果手头暂时没有合适硬件，一键申领 **100 小时免费云端 AMD 大显存算力**。

在下载任何软件之前，我们先做一次像“量体裁衣”一样简单的硬件体检——**选对和自己显卡显存严丝合缝的模型，能避开新手 90% 的卡顿与显存溢出（OOM）报错！**

---

## 第一步：30 秒查清你的 AMD 硬件家底

请根据你当前使用的操作系统，任选下面一种方法查看：

### 🪟 如果你使用 Windows 11（零命令行最简法）

1. 同时按下键盘上的 **`Ctrl + Shift + Esc`** 打开「任务管理器」；
2. 点击左侧的 **「性能（Performance）」** 图标（像心电图的那个）；
3. 点击左下角的 **「GPU 0」或「GPU 1」**，重点记下右上角和底部的 **3 个数字**：
   - **显卡名称**：例如 `AMD Radeon RX 7800 XT`、`AMD Radeon RX 9070 XT` 或 `AMD Radeon 890M / 8060S Graphics`；
   - **专用 GPU 内存（Dedicated GPU memory）**：这就是独立显卡的黄金硬显存（例如 `8.0 GB`、`16.0 GB`、`24.0 GB`）；
   - **如果是锐龙核显笔记本 / 迷你主机（APU）**：请同时点击左侧 **「内存（Memory）」** 看总物理内存（例如 `32.0 GB` 或 `128 GB`），并留意任务管理器里是否有 **「NPU 0（AMD IPU / XDNA 2）」** 图标！

如果你想在 PowerShell 里一条命令看清所有显卡与显存信息，也可以打开终端复制运行：

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, @{Name="AdapterRAM_GB";Expression={[math]::Round($_.AdapterRAM/1GB, 1)}}, DriverVersion
```

### 🐧 如果你使用 Ubuntu / Linux

打开终端，复制运行以下两条命令：

```bash
# 1. 查看显卡物理型号与系统总内存
lspci | grep -iE "vga|display|3d" && free -h

# 2. 如果已安装过驱动/ROCm，直接查看显卡详细显存与 gfx 架构代号
amd-smi static 2>/dev/null || rocminfo | grep -E "Marketing Name|Name:\s+gfx" | head -n 6
```

---

## 第二步：对号入座！课程四大算力档位 × 四大主体模型选型表

查清显存（或统一内存）后，请直接在下表中找到你的专属档位。`hello-rocm 2.0` 全书精选了综合中文能力、多模态、工具调用（Tool Calling）与本地推理速度最强的 **4 款旗舰开源主体模型**，保证每个档位都能丝滑通关：

| 你的硬件档位 | 代表性 AMD 显卡 / 处理器型号 | 课程推荐主体模型（闭眼选它！） | 推荐量化规格与预计显存占用 | 核心能力与通关体验 |
|:---|:---|:---|:---|:---|
| **🟢 入门档 / 轻薄本档**<br/>（8G～12G 独显，或 16G～32G 锐龙核显 & NPU 本） | • 独显：RX 7600 / 7600 XT / 7700 XT / RX 9060 XT<br/>• 移动/APU：Ryzen AI 7/9 350 / HX 370 / 8845HS | **`Gemma 4 E4B`**<br/>（`google/gemma-4-E4B-it`） | `Q4_K_M`（GGUF）或 FLM NPU 格式<br/>仅占 **~3.2 GB ～ 4.8 GB**（含 32K 上下文） | **小钢炮天花板**！有效约 4.5B 参数（含多模态达 ~8B 级表现），原生支持图文音理解、128K 长窗口与精准函数调用，8G 显卡和轻薄本跑出 **50～90+ tok/s** 飞一般手感！ |
| **🔵 甜点主力档**<br/>（16G～24G 独立显卡） | • RDNA 3：RX 7800 XT (16G) / 7900 GRE (16G) / 7900 XT (20G) / 7900 XTX (24G)<br/>• RDNA 4：RX 9070 / RX 9070 XT (16G)<br/>• 工作站：Radeon PRO W7800 / W7900 | **双主力自由搭档**：<br/>① **`Qwen3.5-9B`**（`Qwen/Qwen3.5-9B`）<br/>② **`Gemma 4 12B`**（`google/gemma-4-12B-it`） | • `Q4_K_M` 约 **6.5 GB ～ 9.2 GB**<br/>• `Q8_0` / `FP8` 约 **10.5 GB ～ 14.5 GB** | • **`Qwen3.5-9B`**：中文公文写作、复杂报销单抽取、长通知 JSON 格式化与中文 Agent 首选；<br/>• **`Gemma 4 12B`**：跨文件代码重构（`Qoder`）、复杂英文文献阅读与严谨逻辑推理利器！ |
| **🟣 旗舰 AI PC 档**<br/>（64G～128G 统一内存工作站/迷你主机） | • **Ryzen AI Max+ 395**（Strix Halo，Radeon 8060S 40 CU 集显 + 最高 96GB 动态划拨显存） | **`Qwen3.6-35B-A3B`**<br/>（`Qwen/Qwen3.6-35B-A3B`） | `Q4_K_M` 约 **~22 GB ～ 26 GB**<br/>`Q8_0` / `BF16` 约 **38 GB ～ 72 GB** | **桌面级降维打击**！总参数高达 **350 亿（35B）**，但采用稀疏混合专家（MoE）架构，**每生成一个词仅激活 30 亿（3B）参数**——用 3B 小模型的闪电速度，白嫖媲美云端大模型的深度规划与长文推理实力！ |
| **☁️ 免费云算力档**<br/>（手头暂无 AMD 卡 / 苹果 Mac / 老旧轻薄本） | • **AMD Radeon Cloud**（MI300X / Radeon PRO 大显存实例）<br/>• **AUP Learning Cloud** | 全系通吃！可直接体验 **`Qwen3.5-9B`**、**`Gemma 4 12B`** 乃至 **`Qwen3.6-35B-A3B`** | 支持全精度 **`BF16`** 与 **`FP8`** 高速推理及第 5 章完整 LoRA 微调 | 注册即领 **100 小时中国区免费云端算力**，通过一行 SSH 隧道命令映射回本地，你电脑上的 WorkBuddy 和 `Qoder` 照样丝滑调用！ |

> [!IMPORTANT]
> **给所有同学的一条省心黄金建议**：  
> 就算你的电脑是 24G 大显存甚至 128G 顶配，在第 1 章**第一次跑通流程时**，我们也强烈推荐先花 2 分钟下载体积最小（仅 ~2.9 GB）的 **`Gemma 4 E4B`（`Q4_K_M`）** 验证全链路畅通，再切换到 `Qwen3.5-9B`、`Gemma 4 12B` 或 `Qwen3.6-35B-A3B` 主力服役！

---

## 第三步：10 秒速查你的 GPU `gfx` 架构代号（第 1 章装环境必用）

在 AMD ROCm 软件栈里，每一代显卡都有一个简洁的内部身份证号，叫做 **`gfx` Target 编号**（比如 `gfx1100`、`gfx1201`）。第 1.2 节用 `uv pip` 一键安装 PyTorch ROCm 10 时会用到它，现在先顺手记在备忘录里：

| 产品家族与架构代际 | 常见显卡 / 处理器具体型号 | 对应的 `gfx` 代号 | ROCm 10.0.0 支持说明 |
|:---|:---|:---|:---|
| **RDNA 4 桌面独显** | Radeon RX 9070 XT / RX 9070 | **`gfx1201`** | 原生支持 Win11 & Ubuntu PyTorch `device-gfx1201` 与 FP8 加速 |
| **RDNA 4 主流独显** | Radeon RX 9060 XT | **`gfx1200`** | 原生支持 Win11 & Ubuntu PyTorch `device-gfx1200` |
| **RDNA 3.5 旗舰统一内存 APU** | **Ryzen AI Max+ 395 / Max 390**（Strix Halo，Radeon 8060S / 8050S） | **`gfx1151`** | 旗舰 AI PC 神器！原生支持 `device-gfx1151` + XDNA 2 NPU |
| **RDNA 3.5 移动锐龙 AI 300** | Ryzen AI 9 HX 370 / HX 365（Radeon 890M / 880M） | **`gfx1150`** | 支持 `device-gfx1150` + `lemonade-server` GPU/NPU 双擎 |
| **RDNA 3 旗舰桌面/工作站独显** | Radeon RX 7900 XTX / 7900 XT / 7900 GRE / PRO W7900 / W7800 | **`gfx1100`** | 社区保有量极大的主力甜点/旗舰卡，`device-gfx1100` |
| **RDNA 3 中高端独显** | Radeon RX 7800 XT / RX 7700 XT / PRO W7700 | **`gfx1101`** | 16G 高性价比主力卡，对应 `device-gfx1101` |
| **RDNA 3 入门独显** | Radeon RX 7600 XT (16G) / RX 7600 (8G) | **`gfx1102`** | 入门高性价比卡，对应 `device-gfx1102` |
| **CDNA 3 数据中心计算卡** | Instinct MI300X / MI300A / MI325X（云平台常见） | **`gfx942`** | AMD Radeon Cloud 主力云端大显存加速器，对应 `device-gfx942` |

*(注：如果你的型号是更早期的 RX 6000 系列如 RX 6800/6900 XT `gfx1030`，或者 Ryzen 7040/8040 核显 `gfx1103`，在第 1.3 节和 1.4 节同样可以通过跨平台 Vulkan 后端零门槛跑起 `llama-server` 和 `lemonade-server`！完整全系字典见 [第 9.3 节](../09-amd-ecosystem/index.md)。)*

---

## 第四步：手头没有 AMD 显卡？5 分钟免费申领云端大显存算力

如果你目前用的是轻薄办公本、MacBook，或者想在第 5、6 章体验全精度大模型微调与百亿模型推理，本仓库已为你准备好两条官方免费算力通道：

1. **通道 A（全社会开发者与学生通用）：[AMD Radeon Cloud 免费 100 小时算力申领指南](../cloud/amd-radeon-cloud.md)**
   - **纯中文网页免翻墙**，微信扫码注册并完善信息即可兑换 **100 小时免费云端 GPU 时长**（后续参与魔搭激励计划最高可扩展至 1000+ 小时）；
   - 开箱预装好 ROCm + PyTorch 环境，省去本地装驱动步骤！
2. **通道 B（高校师生实训专享）：[AUP Learning Cloud 高校算力指南](../cloud/aup-learning-cloud.md)**
   - 面向高校课程实验与组队学习营同学提供的云端实训环境。

### 💡 云端算力怎么接我本地电脑的 WorkBuddy 和 `Qoder`？（SSH 隧道一招鲜）

很多用云算力的同学担心：*“模型跑在云端服务器上，我自己电脑桌面上的 WorkBuddy 怎么连过去做 PPT 呢？”*  
其实只需要在你自己电脑的终端里运行一行 **SSH 本地端口转发（Port Forwarding）** 命令（详细公钥配置见 [AMD Radeon Cloud SSH 章节](../cloud/amd-radeon-cloud.md)）：

```bash
# 把云端实例里的 8000 端口（vLLM）或 8080 端口（llama-server）安全映射到你本机的同一端口！
ssh -N -L 8000:127.0.0.1:8000 -L 8080:127.0.0.1:8080 radeon-cloud
```

保持这个终端窗口不关，云端强大的大显存显卡就像直接插进了你的笔记本主板一样，本地所有软件照填 `http://127.0.0.1:8000/v1` 即可零差别运行！

---

## 🔍 本节打卡自检清单

在进入下一节前，请确认你心里已经有了这 3 个答案：
- [x] 我已经看清了自己的显卡型号与可用显存 / 内存大小（或决定使用 AMD Radeon Cloud）；
- [x] 我已经选定了第一颗要下载的主体模型（例如 🟢 `Gemma 4 E4B`、🔵 `Qwen3.5-9B` / `Gemma 4 12B` 或 🟣 `Qwen3.6-35B-A3B`）；
- [x] 我记下了自己显卡对应的 `gfx` 代号（如 `gfx1100` / `gfx1151` / `gfx1201`）。

👉 接下来，用最后 **5 分钟** 看清全书与 Datawhale 六大王牌课程的衔接全景图，并把你的 24 小时专属助教 **`hello-rocm Skill`** 装进电脑：**[0.4 课程全景地图、跨课程引用体系与 `hello-rocm Skill` 伴学安装](./04-roadmap-and-skill.md)**！
