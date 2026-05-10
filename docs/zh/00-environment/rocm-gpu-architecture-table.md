## AMD GPU / APU 架构对照表（ROCm 7.12.0）

> 快速查询你的 GPU 对应的 LLVM target，用于安装命令中的架构参数。

---

### Instinct 系列（数据中心）

| 设备系列 | 具体型号 | LLVM Target | 架构 |
|:---|:---|:---|:---|
| MI350 Series | MI355X, MI350X | `gfx950` | CDNA 4 |
| MI300 Series | MI325X, MI300X, MI300A | `gfx942` | CDNA 3 |
| MI200 Series | MI250X, MI250, MI210 | `gfx90a` | CDNA 2 |
| MI100 Series | MI100 | `gfx908` | CDNA |

---

### Radeon PRO 系列（工作站）

| 设备系列 | 具体型号 | LLVM Target | 架构 |
|:---|:---|:---|:---|
| AI PRO R9000 | R9700, R9600D | `gfx1201` | RDNA 4 |
| PRO W7000 | W7900 Dual Slot, W7900, W7800 48GB, W7800 | `gfx1100` | RDNA 3 |
| PRO W7700 | W7700, V710 | `gfx1101` | RDNA 3 |

---

### Radeon RX 系列（消费级）

| 设备系列 | 具体型号 | LLVM Target | 架构 |
|:---|:---|:---|:---|
| RX 9000 | RX 9070 XT, 9070 GRE, 9070 | `gfx1201` | RDNA 4 |
| RX 9000  | RX 9060 XT LP, 9060 XT, 9060 | `gfx1200` | RDNA 4 |
| RX 7000  | RX 7900 XTX, 7900 XT, 7900 GRE | `gfx1100` | RDNA 3 |
| RX 7000 | RX 7800 XT, 7700 XT, 7700 XE, 7700 | `gfx1101` | RDNA 3 |
| RX 7000  | RX 7600 | `gfx1102` | RDNA 3 |

---

### Ryzen APU 系列（笔记本/移动端）

| 设备系列 | 具体型号 | LLVM Target | 架构 | iGPU 型号 |
|:---|:---|:---|:---|:---|
| **AI Max PRO 300** | AI Max+ PRO 395, Max PRO 390/385/380 | `gfx1151` | RDNA 3.5 | Radeon 8060S |
| **AI Max 300** | AI Max+ 395, Max 390, Max 385 | `gfx1151` | RDNA 3.5 | Radeon 8060S / 8050S |
| **AI PRO 400** | AI 9 HX PRO 475/470, AI 9 PRO 465, AI 7 PRO 450, AI 5 PRO 440/435 | `gfx1150` | RDNA 3.5 | Radeon 890M / 880M / 860M |
| **AI 300** | AI 9 HX 375/370, AI 9 365 | `gfx1150` | RDNA 3.5 | Radeon 890M / 880M |
| **Ryzen 200** | 9 270, 7 260/250, 5 240/230/220, 3 210 | `gfx1103` | RDNA 3 | Radeon 780M / 760M / 740M |

---

### pip 安装索引 URL 速查

| LLVM Target | pip index URL |
|:---|:---|
| `gfx950` | `https://repo.amd.com/rocm/whl/gfx950-dcgpu/` |
| `gfx942` (gfx94X) | `https://repo.amd.com/rocm/whl/gfx94X-dcgpu/` |
| `gfx90a` | `https://repo.amd.com/rocm/whl/gfx90a/` |
| `gfx908` | `https://repo.amd.com/rocm/whl/gfx908/` |
| `gfx1201` / `gfx1200` | `https://repo.amd.com/rocm/whl/gfx120X-all/` |
| `gfx1100` / `gfx1101` / `gfx1102` | `https://repo.amd.com/rocm/whl/gfx110X-all/` |
| `gfx1151` | `https://repo.amd.com/rocm/whl/gfx1151/` |
| `gfx1150` | `https://repo.amd.com/rocm/whl/gfx1150/` |
| `gfx1103` | `https://repo.amd.com/rocm/whl/gfx1103/` |

> 💡 切换 GPU 时只需把安装命令中的 `--index-url` 替换为上表对应 URL 即可。

---

### 操作系统支持概览

| GPU 类别 | Linux | Windows |
|:---|:---|:---|
| Instinct (CDNA) | Ubuntu 22.04/24.04, RHEL, Debian, SLES | ❌ |
| Radeon PRO / RX | Ubuntu 22.04/24.04, RHEL | ✅ Windows 11 25H2 |
| Ryzen APU (AI Max/300/200) | Ubuntu 24.04 (HWE kernel 6.14) | ✅ Windows 11 25H2 |

---

> 📖 完整兼容性矩阵请参考：[ROCm 7.12.0 Compatibility Matrix](https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html)
