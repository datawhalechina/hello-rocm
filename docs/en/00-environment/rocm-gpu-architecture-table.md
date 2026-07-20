## AMD GPU / APU Architecture Reference Table (ROCm 7.14.0)

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
| **AI Max PRO 400** | AI Max+ PRO 495, Max PRO 490/485 | `gfx1151` | RDNA 3.5 | Radeon 8060S |
| **AI Max PRO 300** | AI Max+ PRO 395, Max PRO 390/385/380 | `gfx1151` | RDNA 3.5 | Radeon 8060S |
| **AI Max 300** | AI Max+ 395, AI Max+ 392, AI Max+ 388, Max 390, Max 385 | `gfx1151` | RDNA 3.5 | Radeon 8060S / 8050S |
| **AI PRO 400** | AI 9 HX PRO 475/470, AI 9 PRO 465, AI 7 PRO 450, AI 5 PRO 440 | `gfx1150` / `gfx1152` | RDNA 3.5 | Radeon 890M / 880M / 860M |
| **AI 400** | AI 9 HX 475/470, AI 9 465, AI 7 450 | `gfx1150` / `gfx1152` | RDNA 3.5 | Radeon 890M / 880M / 860M |
| **AI 400 (gfx1153, new in 7.14.0)** | AI 7 445, AI 5 435/430, AI 5 PRO 435 | `gfx1153` | RDNA 3.5 | Radeon 860M / 840M |
| **AI 300** | AI 9 HX 375/370, AI 9 365, AI 7 350/345, AI 5 340/330 | `gfx1150` / `gfx1152` | RDNA 3.5 | Radeon 890M / 880M |
| **Ryzen 200** | 9 270, 7 260/250, 5 240/230/220, 3 210 and PRO series | `gfx1103` | RDNA 3 | Radeon 780M / 760M / 740M |

---

### pip Install (device extras Quick Reference)

> Since 7.14.0, wheels are served from a single multi-arch index `https://repo.amd.com/rocm/whl-multi-arch/`, and you select the architecture via the `[device-gfxXXXX]` extra (no more per-arch `--index-url`).

| LLVM Target | device extras tag |
|:---|:---|
| `gfx950` | `device-gfx950` |
| `gfx942` | `device-gfx942` |
| `gfx90a` | `device-gfx90a` |
| `gfx908` | `device-gfx908` |
| `gfx1201` | `device-gfx1201` |
| `gfx1200` | `device-gfx1200` |
| `gfx1100` / `gfx1101` / `gfx1102` | `device-gfx1100` / `device-gfx1101` / `device-gfx1102` |
| `gfx1030` | `device-gfx1030` |
| `gfx1151` | `device-gfx1151` |
| `gfx1150` | `device-gfx1150` |
| `gfx1152` | `device-gfx1152` |
| `gfx1153` (new in 7.14.0) | `device-gfx1153` |
| `gfx1103` | `device-gfx1103` |
| All architectures | `device-all` |

Install example (gfx1151):

```bash
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,device-gfx1151]==7.14.0"
uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision[device-gfx1151]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0"
```

> 💡 When switching GPUs, simply replace the `device-gfxXXXX` tag in the extra with the corresponding value from the table above.

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

> 📖 For the full compatibility matrix, see: [ROCm 7.14.0 Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)