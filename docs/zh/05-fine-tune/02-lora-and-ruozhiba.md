# 5.2 直觉算清 LoRA 显存账 + 5 分钟「弱智吧（`COIG-CQIA`）」中文破冰实验

> **本节目标**：用“原版新华字典 + 透明便利贴”的直觉彻底弄懂 **LoRA（Low-Rank Adaptation）** 为什么能让家用 AMD 显卡训动几十亿参数的大模型，并在 `LLaMA-Factory WebUI` + `SwanLab` 上用经典的中文「弱智吧（`COIG-CQIA`）」数据集跑通人生第一次 5 分钟微调破冰！

---

## 一、为什么“推理只要 6GB，全量微调却要 100GB+”？LoRA 是怎么救场的？

在第 1.1 节我们算过，跑一颗 9B 模型的推理只要几 GB 显存；但如果你要把这 **90 亿个旋钮全部解冻重新训练（Full Fine-Tuning，全参数微调）**，显卡里除了放权重本身，还得同时存下：
- 每个旋钮的**梯度（Gradients，往哪边拧）**：~18 GB
- AdamW 优化器的**一阶动量与二阶方差状态（Optimizer States，高精度账本）**：~72 GB
- 前向传播的中间**激活值（Activations）**：十几 GB  
加起来轻松突破 **110 GB 显存**！普通个人电脑连门都进不去。

这时候，天才的 **LoRA（低秩自适应）** 站了出来。它的核心思想简单得令人拍案叫绝：

> **“原版那本厚达 90 亿字的《新华字典》（原始权重 $W_0$）我一个字都不准改、全部冰冻锁死！我只在字典旁边贴两张极薄的‘透明修正便利贴’（低秩小矩阵 $A$ 和 $B$），专门记录你的新规矩！”**

$$\text{微调后的输出} = \underbrace{W_0 x}_{\text{冰冻原版大脑 (99.5% 参数不耗优化器显存!)}} + \underbrace{\frac{\alpha}{r} B \cdot A x}_{\text{极薄便利贴适配器 (仅占总参数 0.1%~0.8%!) }}$$

看看这一招带来了什么奇迹：
1. **训练显存暴降 80%+**：因为 99.5% 的原版参数被冻结了，根本不需要为它们保存庞大的优化器账本！一张 16G～24G 的 AMD 甜点卡就能稳稳训动 `Qwen3.5-9B`，8G～12G 入门卡就能训动 `Gemma 4 E4B`！
2. **适配器轻如鸿毛、随时插拔**：训完保存下来的 LoRA 便利贴文件只有 **几十 MB**！你可以周一贴上《办公通知抽取便利贴》，周五换贴《小红书文案便利贴》，也可以一键无缝合并（Merge）回原模型导出 GGUF！

---

## 二、5 个核心超参数白话字典（调参不再盲人摸象）

不管你用网页界面还是命令行，微调永远绕不开这 5 个核心旋钮，1 分钟全部记牢：

| 超参数名称 | 推荐新手甜点值 | 白话物理含义与调节口诀 |
|:---|:---:|:---|
| **`lora_rank` (`r`)** | **`8` 或 `16`** | **便利贴的“厚度”**。学 JSON 格式或语气风格，`r=8` 或 `16` 完全够用；设太大（如 128）既费显存又容易把小数据集死记硬背（过拟合）！ |
| **`lora_alpha`** | **`16` 或 `32`** | **便利贴发言权的“放大倍数”**。工程经验口诀：永远设为 `lora_rank` 的 **2 倍**（如 `r=16, alpha=32`）。 |
| **`learning_rate`（学习率）** | **`1.0e-4`** (`0.0001`) | **每次拧旋钮的步子大小**。LoRA 微调经典黄金值是 `1e-4`（比全量微调大一个数量级）；步子太大会震荡乱码，太小半天学不会。 |
| **`per_device_train_batch_size` + `gradient_accumulation_steps`** | **`1` + `8`（或 `2` + `4`）** | **省显存绝招“分期付款”**！显卡一次只塞 1 条样本算梯度（极省显存），攒够 8 条再统一更新一次参数，等效于 `全局 Batch = 8` 的稳定效果！ |
| **`gradient_checkpointing`** | **`True`（必开！）** | **用大约 15% 的重算时间，换取立省 40%+ 的激活值显存**——消费级显卡单卡微调必开保命符！ |

> [!TIP]
> **面试加分细节：为什么训练时 `instruction` 提问段的 `label` 要设为 `-100`？**  
> 因为我们希望模型只学习**“怎么写出标准答案（Output）”**，而不希望它去死记硬背**“用户是怎么提问的（Instruction）”**！在 PyTorch 交叉熵损失函数里，标记为 `-100` 的位置会自动跳过不计入 Loss，`LLaMA-Factory` 在底层已经自动帮你处理好了这一步。

---

## 三、🧪 5 分钟中文破冰实验：用「弱智吧（`COIG-CQIA`）」开光你的第一炉丹！

为什么中外大模型圈子里有一句玩笑话叫 **“弱智吧出题，大模型试金石”**？  
因为中文「弱智吧」段子里充满了精妙的**谐音双关、字面陷阱与反常识逻辑**——未经高质量中文逻辑微调的小模型，一遇到下面这种问题就会一本正经地掉进坑里：
- 🙋 **提问**：*“蓝牙耳机坏了，去医院应该挂牙科还是耳科？”*
- 🤖 **微调前迷糊小模型**：*“建议您挂耳鼻喉科进行检查，因为耳机主要佩戴在耳道部位……”*（全场爆笑！）
- 🧠 **微调后清醒模型**：*“去医院挂哪科都没用！‘蓝牙’是一种短距离无线通信技术，不是真的长了蓝色的牙齿，耳机坏了应该找数码维修店或售后！”*

中科院与魔搭开源的高质量中文指令集 **`m-a-p/COIG-CQIA`** 专门收录了经过人工精校的 `ruozhiba`（弱智吧逻辑解析）子集。现在我们只抽其中 **240 条**，花 5 分钟亲眼看着你的 AMD 显卡把它学会！

### Step 1：在 ROCm 10 虚拟环境中安装 `LLaMA-Factory` 与 `SwanLab`

确保你已激活 [第 1.2 节](../01-first-model/02-rocm-env-setup.md) 装好 ROCm PyTorch 的 `.venv` 环境：

```bash
source ~/ai_workspace/.venv/bin/activate  # Windows 为 D:\ai_workspace\.venv\Scripts\Activate.ps1

# 使用清华镜像安装 LLaMA-Factory 与 SwanLab 可视化套件（注意加 --no-deps 或锁定已有 ROCm torch 不被覆盖！）
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "llamafactory[metrics]" swanlab datasets peft accelerate
# 再次一行确认你的 ROCm PyTorch 依然稳稳在线：
python -c "import torch; print('ROCm GPU Ready:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 2：从魔搭拉取 240 条「弱智吧」精选中文样本到本地

```bash
mkdir -p ~/ai_workspace/fine_tune_lab/data && cd ~/ai_workspace/fine_tune_lab

python -c "
import json
from modelscope.msdatasets import MsDataset

ds = MsDataset.load('m-a-p/COIG-CQIA', subset_name='ruozhiba', split='train')
samples = []
for i, row in enumerate(ds):
    if i >= 240:
        break
    samples.append({
        'instruction': row.get('instruction', ''),
        'input': row.get('input', ''),
        'output': row.get('output', ''),
    })
with open('data/ruozhiba_240.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
with open('data/dataset_info.json', 'w', encoding='utf-8') as f:
    json.dump({'ruozhiba_240': {'file_name': 'ruozhiba_240.json'}}, f, ensure_ascii=False, indent=2)
print(f'✅ 已准备就绪 {len(samples)} 条弱智吧中文破冰数据！')
"
```

### Step 3：零代码启动 `LLaMA-Factory WebUI` 点火开训！

```bash
export USE_MODELSCOPE_HUB=1
llamafactory-cli webui
```

在浏览器打开 `http://127.0.0.1:7860`（首次可点击左上角语言切为 **简体中文 `zh`**）：
1. **模型名称与路径**：8G～12G 卡填入 `google/gemma-4-E4B-it`（16G～24G 卡可填 `Qwen/Qwen3.5-9B`）；
2. **微调方法**：`lora` ｜ **计算类型**：`bf16`；
3. **数据路径**：`./data` ｜ **数据集**：勾选 `ruozhiba_240`（可点「预览数据集」看一眼搞笑真题！）；
4. **参数设置**：学习率 `1e-4`，训练轮数 `3.0`，截断长度 `1024`，设备批大小 `1`，梯度累积 `8`；
5. **SwanLab 可视化**：展开底部「SwanLab 配置」勾选启用（可选离线模式 `local` 免注册直接看本地看板！）；
6. 点击最下方橙色大按钮 **「开始（Start）」**！

---

## 四、怎么看懂人生第一条 `Loss` 曲线？（健康 vs 翻车诊断）

盯着右侧实时刷新的训练折线图（同时另开一个终端运行 `amd-smi monitor` 欣赏你显卡算力拉满的英姿！）：
- **✅ 健康的曲线**：起点大约在 `2.2 ~ 2.8` 左右，随着步数推进**平滑、稳步地一路滑落到 `0.7 ~ 1.1` 附近并趋于平缓**——恭喜你，模型正在稳稳吸收中文双关拆解逻辑！
- **⚠️ 翻车的两种信号**：
  - 如果 Loss 突然变成 `0.0000` 或 `NaN`：说明学习率太大或精度溢出（确认选了 `bf16` 而不是易溢出的老 `fp16`！）；
  - 如果训了 10 轮 Loss 压到了 `0.05` 以下：小心**过拟合**（模型开始逐字背诵这 240 道原题，换道新题反而变呆，保持 **`2 ~ 3` 个 Epoch** 最健康！）。

训练结束后切到 WebUI 顶部的 **「Chat（对话测试）」** 标签页：先不加载适配器问一遍“蓝牙耳机挂哪科”或“生鱼片是死鱼片吗”，再点击 **「加载适配器」** 问同一题——亲眼见证 5 分钟带来的通透蜕变！

---

## 👉 下一节：把这股神奇的微调力量用回我们的核心办公主线！

破冰热身大获成功！接下来我们动真格的——针对第 2、4 章日常办公中最高频也最考验稳定性的 **「口语化中文微信群/邮件通知 → 100% 严守契约的 JSON 结构化提取」** 任务，带你走完从**隐私脱敏洗数据 → YAML 可复现训练 → 模型合并导出**的工业级标准全流程：

➡️ **[5.3 主线实战：中文办公群通知 JSON 精准提取微调（含隐私脱敏铁律）](./03-office-json-sft.md)**
