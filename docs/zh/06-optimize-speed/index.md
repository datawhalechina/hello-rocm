# 第 6 章｜速度不够怎么办？AMD 量化对比、零 C++ 算子加速与 `Hyperloom` 自动优化

> **本章目标**：**全程不写一行 C++ 代码！** 选定一个模型用 **AMD 官方 `Quark`** 跑通量化前与量化后的直观速度/体积对比，一键唤醒 ROCm 高性能融合算子（`@torch.compile`、`hipBLASLt`、`AITER`），并体验用 **`AMD-AGI/Hyperloom`** 全自动优化推理！

---

## 6.1 给慢助手做 AI 体检：看懂 `TTFT` / `TPS` 与 `TraceLens` 自动诊断

觉得本地模型“慢”时，其实藏着两种截然不同的慢法，治法也完全不同：

| 核心体检指标 | 中文大白话 | 你什么时候最难受？ | 底层卡在哪儿？ |
|:---|:---|:---|:---|
| **`TTFT`**<br/>(Time To First Token) | **首字等待延迟**：按下回车到蹦出第 1 个字等了几秒 | 往 WorkBuddy / `Qoder` 塞了一整篇长论文或上万字工具定义，干等 10 秒没动静 | **算力密集（Compute-Bound）**：需要开启融合算子（`FlashAttention` / `AITER`）与 **Prefix Caching 前缀缓存** |
| **`TPS`**<br/>(Tokens Per Second) | **每秒吐字速度**：蹦出第 1 个字后，往外飙字快不快 | 模型一个字一个字慢吞吞往外挤，看着着急 | **显存带宽密集（Memory-Bound）**：每生成一个字都要把整个模型权重搬运一遍，**做权重量化压缩体积立竿见影**！ |

- **一键定位热点算子**：除了终端执行 `watch -n 1 amd-smi` 观察显存与功耗外，AMD 官方开源了纯 Python 性能剖析工具 **[`AMD-AGI/TraceLens`](https://github.com/AMD-AGI/TraceLens)**，可将 PyTorch / `vLLM` Trace 直接汇总成清晰的算子耗时排行 Excel/Markdown 报表，一眼看清时间花在了矩阵乘（GEMM）还是注意力层上。

---

## 6.2 AMD 官方量化直观实战：用 `AMD Quark` 跑同一模型的量化前后对比

想亲眼验证“量化到底能省多少空间、提速多少倍”？我们无需复杂校准流程，直接**从课程三大主体模型中选定一个基准模型（如 `Qwen3.5-9B` 或 `Gemma 4 12B`），用 AMD 官方量化工具箱 [`AMD Quark`](https://github.com/amd/Quark) 跑一遍量化前（`BF16`）与量化后（`INT4` / `FP8`）的同台实测对比**！

### Step 1：安装 `amd-quark` 并执行单模型量化

本仓库已为你封装好开箱即用的量化与对比报告生成脚本 **[`src/main-assistant/quant_bench/quark_before_after_bench.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/quant_bench/quark_before_after_bench.py)**：

```bash
uv pip install amd-quark transformers accelerate

# 对主体模型 Qwen3.5-9B 执行对称 INT4 分组量化并导出
python3 src/main-assistant/quant_bench/quark_before_after_bench.py \
  --quantize-model Qwen/Qwen3.5-9B \
  --output-dir ./quark_qwen3.5_9b_int4
```

### Step 2：量化前 vs 量化后速度、体积与显存对比实测

分别用 `vLLM`（或 `llama-server`）加载原版 `BF16` 模型与量化后模型，记录同一组中文提示词下的实测指标，并运行报告生成命令：

```bash
python3 src/main-assistant/quant_bench/quark_before_after_bench.py --demo-report
```

你将得到一张可直接放进第 7 章简历评估报告的清晰对比表（以 `Qwen3.5-9B` 在 24G Radeon 独显实测量级为例）：

| 对比维度 | 量化前基线 (`BF16`) | `AMD Quark` 量化后 (`INT4`) | 优化提升幅度 |
|:---|:---:|:---:|:---:|
| **权重文件体积** | 18.10 GB | **5.45 GB** | 体积缩减 **69.9%** |
| **峰值显存占用** | 20.80 GB | **7.20 GB** | 显存节省 **65.4%** |
| **首字延迟 (TTFT)** | 620 ms | **275 ms** | 等待缩短 **55.6%** |
| **生成吞吐速度** | 24.5 tok/s | **58.2 tok/s** | 提速 **2.38x** |

---

## 6.3 零 C++ 算子加速：`@torch.compile` 自动融合 + 一键唤醒 `hipBLASLt` / `AITER`

不写一行 C++/HIP 代码，普通用户怎么享受 AMD 工程师打磨好的底层高性能算子？只需打开下面 **3 个隐藏加速开关**！

### 开关 1：纯 Python 加一行 `@torch.compile` 唤醒 ROCm Triton 算子融合

在 PyTorch 脚本中写自定义前后处理或轻量模型时，只需加上一行装饰器，ROCm 底层的 **Triton 编译器** 就会自动把多次零碎的显存读写融合成一个极速算子：

```python
import torch

@torch.compile
def fused_rmsnorm_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + 1e-6) * weight
    return torch.nn.functional.silu(x_norm)
```

### 开关 2：`llama-server` 开启 `hipBLASLt` + FlashAttention + 对称 KV Cache

启动 `llama-server` 时加上这组环境变量与参数，长文档阅读首字速度立竿见影（注意 `-ctk` 与 `-ctv` 保持对称 `q8_0` 避免回退慢路径）：

```bash
ROCBLAS_USE_HIPBLASLT=1 ./llama-server \
  -m ~/ai_models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf \
  -ngl 99 --ctx-size 32768 --jinja \
  -fa on -ctk q8_0 -ctv q8_0 --no-mmap
```

### 开关 3：`vLLM` 一键唤醒 AMD 王牌融合算子库 `AITER`

启动 ROCm 版 `vLLM` 时，加上环境变量 **`VLLM_ROCM_USE_AITER=1`**，即可一键激活 **[`ROCm/aiter`（AI Tensor Engine for ROCm）](https://github.com/ROCm/aiter)** 的汇编/Triton 级融合注意力与矩阵算子，长上下文吞吐最高可翻 **2 倍以上**！

---

👉 接下来看如何让 WorkBuddy 与 `Qoder` 实现“秒回”，以及全书最酷的 **用 AI 全自动优化 AI**：**[6.4 & 6.5 Prefix Caching 秒回与 `AMD-AGI/Hyperloom` 自动寻优](./agent-cache-and-hyperloom.md)**！
