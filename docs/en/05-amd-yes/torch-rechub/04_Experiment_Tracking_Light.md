# 04 实验跟踪（轻量）：model_logger 接入演示

- **场景**：推荐模型实验管理
- **模型**：继续使用 DeepFM，避免引入新的模型概念
- **数据**：Criteo sample
- **目标**：演示如何把训练超参、训练/验证曲线、最终指标记录到 WandB / SwanLab / TensorBoardX

前面几个教程都能在本地 print 指标，但一旦开始调参，你会遇到几个实际问题：

- 这次实验用了哪些特征、学习率、batch size 和 epoch？
- 哪一次验证集 AUC 最好？
- 调参后是训练更稳了，还是只是随机波动？
- 多人协作时，怎么让别人复现你的结果？

Torch-RecHub 的训练器统一提供 `model_logger` 参数。中文“训练与评估”文档里也提到，`CTRTrainer`、`MatchTrainer`、`MTLTrainer` 都可以记录训练/验证指标和超参数；不需要记录时传 `None` 即可零开销。

## 可选依赖

按你实际使用的平台安装一个即可：

```bash
pip install wandb
pip install swanlab
pip install tensorboardX
```

第一次运行建议先保持所有开关为 `False`，确认训练能跑通；然后只打开一个 logger。多个 logger 同时打开适合正式实验，但新手排查登录、网络和目录权限问题会更困难。

```python
# import os
# os.environ['WANDB_API_KEY'] = "your API_KEY"
# os.environ['SWANLAB_API_KEY'] = "your API_KEY"
```

```python
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# 跟踪组件（按需导入，未安装时会 ImportError）
# from torch_rechub.basic.tracking import WandbLogger, SwanLabLogger, TensorBoardXLogger

SEED = 2022
DEVICE = "cuda:0"  # AMD ROCm 通常通过 cuda:0 设备名访问
PROJECT_ROOT = Path.cwd() if (Path.cwd() / "examples").exists() else Path.cwd().parent
DATASET_PATH = PROJECT_ROOT / "examples/ranking/data/criteo/criteo_sample.csv"

EPOCH = 1
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-3
EARLYSTOP_PATIENCE = 2

# 默认不启用任何 logger（保持轻量）
USE_WANDB = False
USE_SWANLAB = False
USE_TENSORBOARD = False
PROJECT_NAME = "tracking-demo"

torch.manual_seed(SEED)
print("DATASET_PATH:", DATASET_PATH.resolve())
```

## Logger 接入点

`CTRTrainer` 接收一个 logger 或 logger 列表。logger 只需要实现三个方法：`log_hyperparams`、`log_metrics`、`finish`，因此你可以同时传入 WandB、SwanLab 和 TensorBoardX，也可以接入自己的内部实验平台。

本篇的 `build_loggers()` 会根据开关创建 logger：

- `USE_WANDB=True`：把实验记录到 WandB，通常需要先登录或配置 API key。
- `USE_SWANLAB=True`：把实验记录到 SwanLab。
- `USE_TENSORBOARD=True`：把事件文件写到本地 `runs/`，可用 `tensorboard --logdir runs` 查看。

实际训练时，`CTRTrainer.fit()` 会在训练开始记录超参，在每个 epoch 记录 loss、learning rate 和验证指标，在训练结束或异常退出时调用 `finish()`。如果你只想本地跑通，把 `model_logger=None` 留着即可；如果你开始比较多组实验，建议至少打开 TensorBoardX 或其中一个在线平台。

```python
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
    data[dense_features] = sca.fit_transform(data[dense_features]).astype(np.float32)

    for feat in tqdm(sparse_features, desc="label encode sparse"):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(name) for name in dense_features]
    sparse_feas = [SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16) for name in sparse_features]

    y = data["label"]
    x = data.drop(columns=["label"])
    return dense_feas, sparse_feas, x, y


def build_loggers():
    loggers = []

    if USE_WANDB:
        try:
            from torch_rechub.basic.tracking import WandbLogger
            loggers.append(
                WandbLogger(
                    project=PROJECT_NAME,
                    name=f"deepfm-{SEED}",
                    config={"lr": LR, "batch_size": BATCH_SIZE, "seed": SEED},
                    tags=["criteo", "ctr", "deepfm"],
                )
            )
            print("WandbLogger initialized")
        except ImportError as e:
            print("Wandb not installed, skipped:", e)

    if USE_SWANLAB:
        try:
            from torch_rechub.basic.tracking import SwanLabLogger
            loggers.append(
                SwanLabLogger(
                    project=PROJECT_NAME,
                    experiment_name=f"deepfm-{SEED}",
                    config={"lr": LR, "batch_size": BATCH_SIZE, "seed": SEED},
                )
            )
            print("SwanLabLogger initialized")
        except ImportError as e:
            print("SwanLab not installed, skipped:", e)

    if USE_TENSORBOARD:
        try:
            from torch_rechub.basic.tracking import TensorBoardXLogger
            loggers.append(TensorBoardXLogger(log_dir=f"./runs/deepfm-{SEED}"))
            print("TensorBoardXLogger initialized: ./runs/deepfm-%s" % SEED)
        except ImportError as e:
            print("tensorboardX not installed, skipped:", e)

    return loggers if loggers else None
```

```python
dense_feas, sparse_feas, x, y = get_criteo_data_dict(DATASET_PATH)
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=BATCH_SIZE)

model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [64, 32], "dropout": 0.1, "activation": "relu"},
)

model_logger = build_loggers()  # None means tracking is disabled.

ctr_trainer = CTRTrainer(
    model,
    optimizer_params={"lr": LR, "weight_decay": WEIGHT_DECAY},
    n_epoch=EPOCH,
    earlystop_patience=EARLYSTOP_PATIENCE,
    device=DEVICE,
    model_path="./",
    model_logger=model_logger,
)

ctr_trainer.fit(train_dl, val_dl)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
print(f"test auc: {auc}")

# Trainer.fit calls finish() on active loggers. If you need to log test metrics,
# move evaluation into your own training loop before finalizing the logger.
```


**输出：**

```text
discretize dense: 100%|██████████| 13/13 [00:00<00:00, 2022.93it/s]
label encode sparse: 100%|██████████| 39/39 [00:00<00:00, 3900.75it/s]
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
train: 100%|██████████| 1/1 [00:00<00:00, 31.05it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 166.67it/s]
```


**输出：**

```text
epoch: 0 validation: auc: 0.2
```


**输出：**

```text
validation: 100%|██████████| 1/1 [00:00<00:00, 88.81it/s]
```


**输出：**

```text
test auc: 0.2875
```

## 运行后看什么

如果三个开关都保持 `False`，这篇和普通 DeepFM 训练一样，只在 notebook 输出里打印训练日志和测试 AUC。

如果你打开了某一个 logger，可以重点看三类信息：

- **超参记录**：学习率、batch size、epoch、模型结构参数和项目名，方便复现实验。
- **训练曲线**：`train/loss` 是否稳定下降，`val/auc` 是否提升后趋于平稳。
- **最终指标**：测试集 AUC，以及 best epoch 对应的验证指标。

实际调参时建议每次只改变一个主要变量，例如只改 `EMBED_DIM` 或只改 MLP 结构。这样实验追踪平台里的曲线差异才容易解释；如果一次同时改特征、学习率、batch size 和 epoch，即使指标变好，也很难判断是哪一个改动生效。
