## Qwen3.6-35B-A3B

Qwen3.6-35B-A3B is a Qwen MoE text model (~**35B total / 3B active**). This chapter has two hardware paths: Lemonade / llama.cpp on **GGUF + iGPU**, FastFlowLM on **NPU2 + XDNA2**. Do not feed GGUF to `flm`, or `model.q4nx` to llama.cpp.

> GPU tutorials for earlier Qwen generations: [Qwen3](/01-deploy/qwen3/llamacpp-rocm7-deploy) · [Qwen3.5](/01-deploy/qwen3.5/llamacpp-rocm7-deploy). A Lemonade id like `Qwen3.5-35B-A3B-GGUF` is a different generation — use an id that contains **3.6**.

| Path | Format | Typical size | Runtime |
|:---|:---|:---|:---|
| **GPU / Lemonade** | GGUF (`Qwen3.6-35B-A3B-GGUF`, often UD-Q4) | ~20–23 GiB | Lemonade `llamacpp:vulkan` (or `rocm`) @ **13305** |
| **NPU / FastFlowLM** | NPU2 | ~**29 GiB** resident | `flm serve qwen3.6-moe:35b-a3b` @ **8219** |

This is hungrier than Gemma 4 E4B (~9 GiB). Stop other large models first. On the validation host, NPU short-prompt concurrency 1/2/4/8 passed; long-form decode was about **14 tok/s**.

Deploy: [FastFlowLM NPU](./fastflowlm-npu-deploy.md) · [Lemonade GPU](./lemonade-gpu-deploy.md) · [Lemonade NPU](./lemonade-npu-deploy.md)
