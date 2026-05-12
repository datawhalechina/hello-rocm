# 05 模型导出与推理验证：ONNX（CTR + Matching）

- **场景**：训练后的模型交付到推理/服务系统
- **模型**：DeepFM 排序模型 + DSSM 双塔召回模型
- **目标**：演示 Torch-RecHub 模型导出 ONNX、用 ONNXRuntime 做最小推理验证，并展示 INT8/FP16 量化入口

前面几篇关注“怎么训练模型”。这一篇关注“训练好的模型如何被服务系统使用”。中文 Serving 文档把生产链路概括为：

```text
训练模型 -> ONNX 导出 -> 量化 -> 向量索引 -> 在线服务
```

排序模型和召回模型的部署方式略有不同：

- 排序/精排：线上请求经过特征服务，拼出模型输入，调用 ONNXRuntime 得到点击率或转化率分数。
- 双塔召回：item tower 通常离线批量计算物品向量并写入 Annoy/FAISS/Milvus；user tower 在线计算用户向量，再查询向量索引。

## 依赖安装

只导出 ONNX：

```bash
pip install onnx
```

导出后想用 ONNXRuntime 做推理验证或 INT8 动态量化：

```bash
pip install onnxruntime
```

想做 FP16 转换：

```bash
pip install onnxconverter-common
```

如果你的环境支持 extra，也可以一次性安装：

```bash
pip install "torch-rechub[onnx]"
```

## 运行建议

- 推荐先用 CPU 导出 ONNX，兼容性最好；训练可以用 GPU。
- ONNX 输入名来自 Feature 名称，推理时传入的 numpy dict 必须和导出签名一致。
- 导出成功不等于线上可用，至少要做一次 PyTorch vs ONNXRuntime 的输出对齐。
- 量化后还要重新做精度和业务指标评估，因为速度变快不代表效果完全不变。

```python
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator
from torch_rechub.utils.onnx_export import ONNXExporter
from torch_rechub.utils.model_utils import generate_dummy_input_dict

SEED = 2022
DEVICE = "cuda:0"  # AMD ROCm 通常通过 cuda:0 设备名访问
EXPORT_DEVICE = "cpu"  # ONNX export defaults to CPU for better compatibility
torch.manual_seed(SEED)

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "examples").exists() else Path.cwd().parent
EXPORT_DIR = PROJECT_ROOT / "onnx_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
print("DEVICE:", DEVICE, "EXPORT_DEVICE:", EXPORT_DEVICE)
print("EXPORT_DIR:", EXPORT_DIR.resolve())
```

## 导出流程说明

Torch-RecHub 的排序/召回模型通常接收 `dict[str, Tensor]`。ONNX 不直接使用 Python dict 作为输入，因此 `ONNXExporter` 会自动做一层 wrapper：按 Feature 顺序把多个 positional tensor 还原成模型需要的输入字典。

你不需要手写 wrapper，但需要记住：

- 导出后的 ONNX 模型输入顺序和输入名由 Feature 列表决定。
- dense 输入通常应是 `float32`，sparse/sequence 输入通常应是整数类型。
- dynamic batch 会让导出的模型支持不同 batch size；如果要支持动态序列长度，还要额外确认模型算子和导出参数是否兼容。

这篇会先导出一个 DeepFM 排序模型，再导出 DSSM 的 user tower 和 item tower。前者对应在线排序打分，后者对应召回阶段的向量生成。

```python
# ---------- Part A: CTR（DeepFM）导出 + onnxruntime 推理验证 ----------

DATASET_PATH = PROJECT_ROOT / "examples/ranking/data/criteo/criteo_sample.csv"
EPOCH = 1
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-3


def convert_numeric_feature(val):
    v = int(val)
    if v > 2:
        return int(np.log(v) ** 2)
    else:
        return v - 2


def get_criteo_data_dict(data_path):
    data_path = Path(data_path)
    data = pd.read_csv(data_path, compression="gzip") if data_path.suffix == ".gz" else pd.read_csv(data_path)
    dense_features = [f for f in data.columns.tolist() if f.startswith("I")]
    sparse_features = [f for f in data.columns.tolist() if f.startswith("C")]

    data[sparse_features] = data[sparse_features].fillna("0")
    data[dense_features] = data[dense_features].fillna(0)

    for feat in tqdm(dense_features, desc="discretize dense"):
        sparse_features.append(feat + "_cat")
        data[feat + "_cat"] = data[feat].apply(lambda x: convert_numeric_feature(x))

    sca = MinMaxScaler()
    # MinMaxScaler 默认输出 float64，这会导致后续 dataloader/ONNX 输入变成 double。
    # 这里显式转成 float32，保证与导出的 ONNX（通常期望 float32）一致。
    data[dense_features] = sca.fit_transform(data[dense_features]).astype(np.float32)

    for feat in tqdm(sparse_features, desc="label encode sparse"):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(name) for name in dense_features]
    sparse_feas = [SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16) for name in sparse_features]

    y = data["label"]
    x = data.drop(columns=["label"])
    return dense_feas, sparse_feas, x, y


dense_feas, sparse_feas, x, y = get_criteo_data_dict(DATASET_PATH)
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=BATCH_SIZE)

ctr_model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [64, 32], "dropout": 0.1, "activation": "relu"},
)

ctr_trainer = CTRTrainer(
    ctr_model,
    optimizer_params={"lr": LR, "weight_decay": WEIGHT_DECAY},
    n_epoch=EPOCH,
    earlystop_patience=2,
    device=DEVICE,
    model_path="./",
)
ctr_trainer.fit(train_dl, val_dl)

ctr_onnx_path = str(EXPORT_DIR / "deepfm.onnx")
CTR_ONNX_EXPORTED = False
try:
    exporter = ONNXExporter(ctr_model, device=EXPORT_DEVICE)
    exporter.export(ctr_onnx_path, opset_version=14, dynamic_batch=True, verbose=False)
    CTR_ONNX_EXPORTED = True
    print("exported:", ctr_onnx_path)
except Exception as e:
    print("CTR ONNX export skipped/failed:", repr(e))
```


**输出：**

```text
discretize dense: 100%|██████████| 13/13 [00:00<00:00, 1990.00it/s]
label encode sparse: 100%|██████████| 39/39 [00:00<00:00, 5451.50it/s]
```


**输出：**

```text
the samples of train : val : test are  80 : 11 : 24
```


**输出：**

```text
epoch: 0
```


**输出：**

```text
train: 100%|██████████| 1/1 [00:00<00:00, 34.11it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 166.59it/s]
```


**输出：**

```text
epoch: 0 validation: auc: 0.2
```

```python
# 用 onnxruntime 做一次最小推理验证（允许浮点误差）
# 注意：如果你把 DEVICE 设为 cuda，需要把 batch 输入也搬到同一设备。

try:
    if not CTR_ONNX_EXPORTED:
        raise RuntimeError("CTR ONNX file was not exported; install onnx or check export logs first.")
    import onnxruntime as ort

    # 取一个 batch（dataloader 默认产出 CPU tensors）
    batch_x, _ = next(iter(test_dl))

    ctr_model.eval()
    model_device = next(ctr_model.parameters()).device

    # torch 推理（把输入搬到模型所在 device）；同时把 double → float32
    batch_x_torch = {
        k: (v.float() if v.dtype == torch.float64 else v).to(model_device)
        for k, v in batch_x.items()
    }
    with torch.no_grad():
        torch_out = ctr_model(batch_x_torch).detach().cpu().numpy()

    # ONNXRuntime 推理（输入需为 numpy，通常在 CPU 上即可）
    # 注意：ONNX 常见期望 float32；这里对所有 float64 显式转 float32，并按 onnx 输入签名补齐维度。
    ort_sess = ort.InferenceSession(ctr_onnx_path, providers=["CPUExecutionProvider"])

    # 根据 onnx 输入签名，修正 rank（常见：模型期望 (B,1)，而 dataloader 给的是 (B,)）
    ort_inputs = {}
    ort_input_info = {i.name: i for i in ort_sess.get_inputs()}
    for k, v in batch_x.items():
        if v.dtype == torch.float64:
            v = v.float()
        arr = v.detach().cpu().numpy()

        info = ort_input_info.get(k)
        if info is not None and hasattr(info, "shape"):
            expected_rank = len(info.shape)
            if expected_rank == 2 and arr.ndim == 1:
                arr = arr.reshape(-1, 1)

        ort_inputs[k] = arr

    ort_out = ort_sess.run(None, ort_inputs)[0]

    max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    print("torch_out shape:", torch_out.shape, "onnx_out shape:", ort_out.shape)
    print("max_abs_diff:", max_abs_diff)
except ImportError as e:
    print("onnxruntime not installed, skip inference check:", e)
except Exception as e:
    print("inference check failed:", repr(e))
```

## 推理验证说明

ONNXRuntime 需要 numpy 输入。这里根据 ONNX 模型签名对输入做两件事：

- 把 `float64` dense 输入转成 `float32`。
- 如果导出签名期望二维输入而 dataloader 给出一维数组，则 reshape 为 `[B, 1]`。

验证时 `max_abs_diff` 不需要严格为 0。只要差异很小，通常说明 PyTorch 和 ONNXRuntime 的结果一致。真实上线前还应补充更完整的样本覆盖，例如不同 batch size、缺失值、padding 序列、极端 id、空历史序列等。

## 可选：ONNX 模型量化（INT8 / FP16）

量化是部署优化，不是训练必需步骤。第一次跑教程时可以先看导出和推理验证，确认流程通了再打开量化。

- **INT8 动态量化**：更偏向 CPU 推理加速，常用于 MLP/Linear 较多的模型。
- **FP16 转换**：更偏向 GPU 推理，主要降低显存占用并提升吞吐。

量化后建议重新做一次推理对比和线上指标评估，因为模型体积和速度变好了，不代表效果一定完全不变。

对于双塔召回，量化只是部署的一部分。item tower 产出的物品向量还需要写入向量索引；本仓库 Serving 文档提供了统一的 `builder_factory`，可以在 Annoy、FAISS、Milvus 之间切换。小规模本地实验用 Annoy 更轻量，高性能单机场景常用 FAISS，需要服务化和持久化管理时可以考虑 Milvus。

```python
if not CTR_ONNX_EXPORTED:
    print("Skip quantization because CTR ONNX export did not complete.")
else:
    # 量化/转换导出的 ONNX（可选）

    from torch_rechub.utils.quantization import quantize_model

    ctr_onnx_int8_path = os.path.join(EXPORT_DIR, "deepfm.int8.onnx")
    ctr_onnx_fp16_path = os.path.join(EXPORT_DIR, "deepfm.fp16.onnx")

    # INT8 动态量化（需要 onnxruntime）
    try:
        quantize_model(ctr_onnx_path, ctr_onnx_int8_path, mode="int8")
        print("exported int8:", ctr_onnx_int8_path)
    except Exception as e:
        print("INT8 quantize skipped:", repr(e))

    # FP16 转换（需要 onnx + onnxconverter-common）
    try:
        quantize_model(ctr_onnx_path, ctr_onnx_fp16_path, mode="fp16", keep_io_types=True)
        print("exported fp16:", ctr_onnx_fp16_path)
    except Exception as e:
        print("FP16 convert skipped:", repr(e))
```


**输出：**

```text
Skip quantization because CTR ONNX export did not complete.
```

```python
# ---------- Part B: Matching（DSSM）双塔导出 + 最小推理验证 ----------

# 为了让本教程独立且快速，这里构造一个最小 DSSM 模型（不依赖完整训练），并导出 user/item tower。

# user tower features
user_features = [
    SparseFeature("user_id", vocab_size=1000, embed_dim=16),
    SparseFeature("gender", vocab_size=3, embed_dim=16),
    SparseFeature("age", vocab_size=10, embed_dim=16),
    SparseFeature("occupation", vocab_size=30, embed_dim=16),
    SparseFeature("zip", vocab_size=5000, embed_dim=16),
    SequenceFeature("hist_movie_id", vocab_size=5000, embed_dim=16, pooling="mean", shared_with="movie_id"),
]

# item tower features
item_features = [
    SparseFeature("movie_id", vocab_size=5000, embed_dim=16),
    SparseFeature("cate_id", vocab_size=50, embed_dim=16),
]

match_model = DSSM(
    user_features,
    item_features,
    temperature=0.02,
    user_params={"dims": [64], "activation": "prelu"},
    item_params={"dims": [64], "activation": "prelu"},
)

user_onnx_path = str(EXPORT_DIR / "user_tower.onnx")
item_onnx_path = str(EXPORT_DIR / "item_tower.onnx")
MATCH_ONNX_EXPORTED = False
try:
    match_exporter = ONNXExporter(match_model, device=EXPORT_DEVICE)
    match_exporter.export(user_onnx_path, mode="user", opset_version=14, dynamic_batch=True, verbose=False)
    match_exporter.export(item_onnx_path, mode="item", opset_version=14, dynamic_batch=True, verbose=False)
    MATCH_ONNX_EXPORTED = True
    print("exported:", user_onnx_path)
    print("exported:", item_onnx_path)
except Exception as e:
    print("Matching ONNX export skipped/failed:", repr(e))
```

```python
# 双塔最小推理验证：分别对 user/item tower 做一次 onnxruntime forward

try:
    if not MATCH_ONNX_EXPORTED:
        raise RuntimeError("Matching ONNX files were not exported; install onnx or check export logs first.")
    import onnxruntime as ort

    # 生成与 feature 定义一致的 dummy 输入
    dummy_user = generate_dummy_input_dict(user_features, batch_size=2, seq_length=10, device=EXPORT_DEVICE)
    dummy_item = generate_dummy_input_dict(item_features, batch_size=2, seq_length=10, device=EXPORT_DEVICE)

    match_model.eval()
    with torch.no_grad():
        # user tower
        match_model.mode = "user"
        torch_user_out = match_model(dummy_user).detach().cpu().numpy()
        # item tower
        match_model.mode = "item"
        torch_item_out = match_model(dummy_item).detach().cpu().numpy()

    # onnxruntime
    user_sess = ort.InferenceSession(user_onnx_path, providers=["CPUExecutionProvider"])
    item_sess = ort.InferenceSession(item_onnx_path, providers=["CPUExecutionProvider"])

    ort_user_in = {k: v.detach().cpu().numpy() for k, v in dummy_user.items()}
    ort_item_in = {k: v.detach().cpu().numpy() for k, v in dummy_item.items()}

    ort_user_out = user_sess.run(None, ort_user_in)[0]
    ort_item_out = item_sess.run(None, ort_item_in)[0]

    print("user torch/onnx shapes:", torch_user_out.shape, ort_user_out.shape)
    print("item torch/onnx shapes:", torch_item_out.shape, ort_item_out.shape)

    print("user max_abs_diff:", float(np.max(np.abs(torch_user_out - ort_user_out))))
    print("item max_abs_diff:", float(np.max(np.abs(torch_item_out - ort_item_out))))
except ImportError as e:
    print("onnxruntime not installed, skip inference check:", e)
except Exception as e:
    print("matching inference check skipped/failed:", repr(e))
```
