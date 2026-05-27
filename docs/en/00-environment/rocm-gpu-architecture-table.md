## AMD GPU / APU Architecture Reference Table (ROCm 7.13.0)

> Quickly look up the LLVM target for your GPU to use as the architecture parameter in installation commands.

---

### Instinct Series (Data Center)

| Device Series | Specific Models | LLVM Target | Architecture |
|:---|:---|:---|:---|
| MI350 Series | MI355X, MI350X, MI350P | `gfx950` | CDNA 4 |
| MI300 Series | MI325X, MI300X, MI300A | `gfx942` | CDNA 3 |
| MI200 Series | MI250X, MI250, MI210 | `gfx90a` | CDNA 2 |
| MI100 Series | MI100 | `gfx908` | CDNA |

---

### Radeon PRO Series (Workstation)

| Device Series | Specific Models | LLVM Target | Architecture |
|:---|:---|:---|:---|
| AI PRO R9000 | R9700S, R9700, R9600D | `gfx1201` | RDNA 4 |
| PRO W7000 | W7900 Dual Slot, W7900, W7800 48GB, W7800 | `gfx1100` | RDNA 3 |
| PRO W7700 | W7700, V710 | `gfx1101` | RDNA 3 |

---

### Radeon RX Series (Consumer)

| Device Series | Specific Models | LLVM Target | Architecture |
|:---|:---|:---|:---|
| RX 9000 | RX 9070 XT, 9070 GRE, 9070 | `gfx1201` | RDNA 4 |
| RX 9000  | RX 9060 XT LP, 9060 XT, 9060 | `gfx1200` | RDNA 4 |
| RX 7000  | RX 7900 XTX, 7900 XT, 7900 GRE | `gfx1100` | RDNA 3 |
| RX 7000 | RX 7800 XT, 7700 XT, 7700 XE, 7700 | `gfx1101` | RDNA 3 |
| RX 7000  | RX 7600 | `gfx1102` | RDNA 3 |

---

### Ryzen APU Series (Laptop / Mobile)

| Device Series | Specific Models | LLVM Target | Architecture | iGPU Model |
|:---|:---|:---|:---|:---|
| **AI Max PRO 300** | AI Max+ PRO 395, Max PRO 390/385/380 | `gfx1151` | RDNA 3.5 | Radeon 8060S |
| **AI Max 300** | AI Max+ 395, AI Max+ 392, AI Max+ 388, Max 390, Max 385 | `gfx1151` | RDNA 3.5 | Radeon 8060S / 8050S |
| **AI PRO 400** | AI 9 HX PRO 475/470, AI 9 PRO 465, AI 7 PRO 450, AI 5 PRO 440 | `gfx1150` / `gfx1152` | RDNA 3.5 | Radeon 890M / 880M / 860M |
| **AI 400** | AI 9 HX 475/470, AI 9 465, AI 7 450 | `gfx1150` / `gfx1152` | RDNA 3.5 | Radeon 890M / 880M / 860M |
| **AI 300** | AI 9 HX 375/370, AI 9 365, AI 7 350/345, AI 5 340/330 | `gfx1150` / `gfx1152` | RDNA 3.5 | Radeon 890M / 880M |
| **Ryzen 200** | 9 270, 7 260/250, 5 240/230/220, 3 210 and PRO series | `gfx1103` | RDNA 3 | Radeon 780M / 760M / 740M |

---

### pip Index URL Quick Reference

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

> 💡 When switching GPUs, simply replace the `--index-url` in your installation command with the corresponding URL from the table above.

---

### Operating System Support Overview

| GPU Category | Linux | Windows | WSL |
|:---|:---|:---|:---|
| Instinct (CDNA) | Ubuntu 22.04/24.04/26.04, RHEL 9/10, Debian 12/13, SLES 15/16 | ❌ | ❌ |
| Radeon AI PRO R9700S, R9600D (gfx1201) | Ubuntu 22.04/24.04/26.04, RHEL 9/10 | ✅ Windows 11 25H2 | ✅ |
| Radeon AI PRO R9700 (gfx1201) | Ubuntu 22.04/24.04/26.04, RHEL 9/10 | ✅ Windows 11 25H2 | ❌ |
| Radeon RX 9070 (gfx1201) | Ubuntu 22.04/24.04/26.04, RHEL 9/10 | ✅ Windows 11 25H2 | ✅ |
| Radeon RX 9070 XT, 9070 GRE (gfx1201) | Ubuntu 22.04/24.04/26.04, RHEL 9/10 | ✅ Windows 11 25H2 | ❌ |
| Radeon RX 9060 (gfx1200) | Ubuntu 22.04/24.04/26.04 | ✅ Windows 11 25H2 | ✅ |
| Radeon RX 9060 XT LP, 9060 XT (gfx1200) | Ubuntu 22.04/24.04/26.04, RHEL 9/10 | ✅ Windows 11 25H2 | ❌ |
| Radeon PRO W7900 / W7800 / RX 7900 (gfx1100) | Ubuntu 22.04/24.04/26.04, RHEL 9/10 | ✅ Windows 11 25H2 | ❌ |
| Radeon RX 7800/7700/7600 (gfx1101/1102) | Ubuntu 22.04/24.04/26.04, RHEL 9/10 | ✅ Windows 11 25H2 | ❌ |
| Ryzen AI Max+ PRO 395, Max+ 395 (gfx1151) | Ubuntu 24.04/26.04, RHEL | ✅ Windows 11 25H2 | ✅ |
| Ryzen AI Max other models (gfx1151) | Ubuntu 24.04/26.04, RHEL | ✅ Windows 11 25H2 | ❌ |
| Ryzen AI 9 HX (PRO) 475, 375 (gfx1150) | Ubuntu 24.04/26.04, RHEL | ✅ Windows 11 25H2 | ✅ |
| Ryzen AI other models (gfx1150/gfx1152) | Ubuntu 24.04/26.04, RHEL | ✅ Windows 11 25H2 | ❌ |
| Ryzen 200 (gfx1103) | Ubuntu 24.04/26.04 | ✅ Windows 11 25H2 | ❌ |

---

> 📖 For the full compatibility matrix, see: [ROCm 7.13.0 Compatibility Matrix](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html)