## Qwen3.6-35B-A3B 模型介绍

Qwen3.6-35B-A3B 是通义的 MoE 文本模型（约 **35B 总参 / 3B 激活**）。本模块两条硬件路径：Lemonade / llama.cpp 走 **GGUF + iGPU**，FastFlowLM 走 **NPU2 + XDNA2**。不要把 GGUF 塞给 `flm`，也不要把 `model.q4nx` 塞给 llama.cpp。

> 对照：同系列更早的 GPU 教程见 [Qwen3](/zh/01-deploy/qwen3/llamacpp-rocm7-deploy) · [Qwen3.5](/zh/01-deploy/qwen3.5/llamacpp-rocm7-deploy)。Lemonade 目录若写成 `Qwen3.5-35B-A3B-GGUF`，那是另一代，不要当 3.6 用。以 `lemonade list` 里带 **3.6** 的 id 为准。

---

### 路径对照

| 路径 | 格式 | 典型体积 | 运行时 |
|:---|:---|:---|:---|
| **GPU / Lemonade** | GGUF（目录 `Qwen3.6-35B-A3B-GGUF`，多为 UD-Q4） | 约 20–23 GiB | Lemonade `llamacpp:vulkan`（或 `rocm`）@ **13305** |
| **NPU / FastFlowLM** | NPU2 | 常驻约 **29 GiB** | `flm serve qwen3.6-moe:35b-a3b` @ **8219** |

比 Gemma 4 E4B（约 9 GiB）更吃统一内存。先停其它大模型再开。NPU 短问并发 1/2/4/8 已在验证机通过，单路长生成大约 **14 tok/s**。

---

### 部署入口

- [FastFlowLM NPU](./fastflowlm-npu-deploy.md)
- [Lemonade GPU](./lemonade-gpu-deploy.md)
- [Lemonade NPU](./lemonade-npu-deploy.md)

环境：[FastFlowLM](/zh/00-environment/fastflowlm.md) · [Lemonade](/zh/00-environment/lemonade.md) · [XDNA2](/zh/00-environment/xdna2-npu.md)
