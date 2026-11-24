# 6.2 AMD 官方量化实战：用 `AMD Quark` 压缩单个模型并对比前后速度与大小

> **本节目标**：聚焦最干净直观的单模型对比实验 —— 选用 **同一颗模型**（例如第 5 章刚用过的 **`Qwen3.5-9B`** 或轻量级 **`Gemma 4 E4B` / `Qwen3-4B`**），用 **15 行 Python 调用 AMD 官方量化工具箱 [`amd/Quark`](https://github.com/amd/Quark)** 完成对称 `INT4`（或 `FP8`）权重量化，亲手测出 **量化前 vs 量化后在文件体积（GB）、峰值显存（GB）、首字延迟（TTFT）与生成吐字速度（`tok/s`）上的震撼反差**，并牢记 **对称 KV Cache 避坑法则**！

---

## 一、为什么单模型量化能让“体积砍掉 70%，吐字速度反而快一倍”？

在 `6.1` 节我们刚刚学过：大模型逐词吐字（Decode 阶段）卡在 **显存带宽墙（Memory-Bound）** 上。  
拿 **`Qwen3.5-9B`** 来算一笔直观的物理账：
- **量化前（`BF16` 原版）**：权重文件足足有 **`18.1 GB`**。你的显卡每蹦出 1 个汉字，就得把这 18.1 GB 数据从显存搬进计算核心过一遍。就算一张甜点独显的显存带宽有 `500 GB/s`，光搬数据每秒最多也就蹦二十几个字，而且 `20.8 GB` 的总显存占用直接把 16G 显卡拒之门外！
- **经 `AMD Quark` 压缩后（`INT4 Per-Group` 对称量化）**：权重文件瞬间瘦身至 **`~5.45 GB`（体积立减 70%！）**。现在每蹦出 1 个字只需搬运 5.45 GB 数据——不仅 16G 显卡瞬间空出一大半显存呼吸空间，**吐字速度（`tok/s`）更是一举翻了 2 倍以上**！

---

## 二、⚠️ 先记一条价值千金的避坑法则：为什么一定要选「对称量化（Symmetric）」？

不管你用 `AMD Quark` 压缩权重，还是在 `llama-server` / `vLLM` 里开启 KV Cache 量化省显存，都会看到一个开关：`symmetric=True`（对称量化）vs `symmetric=False`（非对称量化）。

> [!WARNING]
> **对称量化（Symmetric Quantization）避坑口诀**：
> 1. **权重量化优先开 `symmetric=True, group_size=128`**：免去零点偏移（Zero-Point）额外计算开销，直接兼容 `vLLM` 与 ROCm 高速反量化融合内核，速度最快且数值最稳；
> 2. **KV Cache 草稿本切忌盲目开非对称 4-bit！** 在长上下文（`32K`）中文抽取和代码重构任务中，注意力分数的长尾分布对 KV 误差极敏感——若需压缩 KV Cache，强烈推荐使用 **对称 8-bit（如 `FP8` 或 `llama-server` 的 `-ctk q8_0 -ctv q8_0`）**，既省一半草稿本显存，又绝不掉智商！

---

## 三、三步跑完「同一模型量化前 vs 量化后」同台实测对比！

本仓库已为你编写好经过自动化单测验证（含除零异常防御）的完整对比工具：**[`src/main-assistant/quant_bench/quark_before_after_bench.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/quant_bench/quark_before_after_bench.py)**。

### Step 1：安装 `amd-quark` 并验证测速报告生成器单测全绿

```bash
cd ~/ai_workspace/hello-rocm
source ~/ai_workspace/.venv/bin/activate

# 1. 安装 AMD 官方量化库 amd-quark
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple amd-quark transformers accelerate

# 2. 先跑一遍本仓库单元测试，确认前后指标对比与边界保护 100% 正常
PYTHONPATH=. python tests/test_course_v2.py TestQuarkBeforeAfterBench -v
```

---

### Step 2：读懂并用 15 行核心代码量化你的模型！

打开 `src/main-assistant/quant_bench/quark_before_after_bench.py` 第 50～77 行，看看用 **AMD Quark** 把一颗 `BF16` 模型压缩成对称 `INT4`（每 128 个权重共享一组缩放因子）有多么清爽直观：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver
import torch

# 1. 加载待压缩的单个原版模型（可填第 5 章合并导出的微调模型，或魔搭本地目录）
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 2. 定义对称 Per-Group INT4 权重量化规格（group_size=128 兼顾精度与速度）
int4_spec = QuantizationSpec(
    dtype=Dtype.int4,
    qscheme=QSchemeType.per_group,
    observer_cls=PerGroupMinMaxObserver,
    symmetric=True,
    group_size=128,
    is_dynamic=False,
)

# 3. 执行一键量化冻结并保存为标准 HuggingFace 目录（vLLM 可直接加载！）
quantizer = ModelQuantizer(Config(global_quant_config=QuantizationConfig(weight=int4_spec)))
quant_model = quantizer.quantize_model(model)
quantizer.freeze(quant_model)
quant_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

在终端直接指定你的本地模型目录执行一键压缩（8G 显存同学可用轻巧的 `Qwen3-4B` 或 `Gemma 4 E4B` 跑对比；16G/24G/云算力同学直接拿主力 `Qwen3.5-9B` 跑对比！）：

```bash
python src/main-assistant/quant_bench/quark_before_after_bench.py \
  --quantize-model ~/ai_workspace/models/Qwen3.5-9B \
  --output-dir ~/ai_workspace/models/Qwen3.5-9B-Quark-INT4
```

*(💡 零等待快捷对标技巧：如果你的机器只有 8G 显存装不下量化前的 9B `BF16` 原版，你也可以在 `llama-server` 上对同一个模型分别下载 `Q8_0 / FP16` 与 `Q4_K_M` 跑完全相同的测试题，采集真实体积、显存和 `tok/s` 填入报告生成器！)*

---

### Step 3：分别测出量化前与量化后的 4 个硬指标，一键生成 Markdown 报告表！

1. **查看硬盘体积（`size_gb`）**：运行 `du -sh <量化前目录> <量化后目录>`；
2. **查看峰值显存（`vram_gb`）**：推理时在另一窗口看 `amd-smi monitor`（或 Windows 任务管理器专用 GPU 内存）；
3. **测出 `TTFT (ms)` 与 `tok/s`**：用我们在 [第 1.3 节](../01-first-model/03-llama-server.md) 学过的 15 行流式测速脚本，在相同输入提示词下各跑 3 次取平均值；
4. **一键打印可直接粘贴进第 7 章简历作品集的对比表**：

```bash
# 先预览标准报告格式：
python src/main-assistant/quant_bench/quark_before_after_bench.py --demo-report
```

输出效果如下（请替换为你自己机器上的实测数字，这就是你的 **《作品集素材 #5：量化加速实测报告》**！）：

| 对比维度 | 量化前基线 | AMD Quark 量化后 | 优化提升幅度 |
|:---|:---:|:---:|:---:|
| **模型规格** | `Qwen3.5-9B (BF16)` | `Qwen3.5-9B (Quark INT4)` | 单模型前后同台对比 |
| **权重文件体积** | `18.10 GB` | `5.45 GB` | 体积缩减 **69.9%** |
| **峰值显存占用** | `20.80 GB` | `7.20 GB` | 显存节省 **65.4%** |
| **首字延迟 (TTFT)** | `620 ms` | `275 ms` | 等待缩短 **55.6%** |
| **生成吞吐速度** | `24.5 tok/s` | `58.2 tok/s` | 提速 **2.38x** |

*(顺手再跑一次第 5.4 节的 `eval_before_after_zh.py` 核验一眼准确率——你会惊喜地发现：体积砍掉近七成、速度翻了两倍多，而中文通知 JSON 提取准确率几乎波动不到 `1%`！这就是现代分组对称量化的工程魅力！)*

---

## 👉 下一节：除了压小模型体积，怎么不写一行 C++ 让算子本身再快一截？

把搬运的数据量砍掉 70% 之后，我们再来看看显卡计算核心内部：能不能不写一行晦涩的 C++ 代码，只靠三个现成的 Python / 环境变量开关，就把零碎的 GPU 算子自动焊接成一条高速公路？

➡️ **[6.3 零 C++ 代码算子加速：三招唤醒 `@torch.compile`、`hipBLASLt` 与 `AITER`](./03-zero-cpp-operators.md)**
