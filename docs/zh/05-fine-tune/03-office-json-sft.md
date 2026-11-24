# 5.3 主线实战：中文办公群通知 JSON 精准提取微调（含隐私脱敏铁律）

> **本节目标**：围绕全书「个人学习与工作助手」主线，亲手完成一次可直接写进简历的工业级垂直任务微调 —— **把充满缩写、倒装与口语省略的中文群通知，100% 稳定提取为严格遵循 `["事项", "截止时间", "负责人"]` 契约的合法 JSON**，并掌握 **训练数据隐私脱敏铁律** 与 **LoRA 权重合并导出**！

---

## 一、为什么要专门为「中文群通知抽取」训一个 LoRA？

在真实校园和职场群里，大家发通知从来不会按教科书语法好好说话：
- *“@所有人 差点忘了说，下周二下班前把三季度耗材盘点表甩给小陈啊，过时不候！”*
- *“紧急！！明早十点前各组长把答辩PPT终版汇总到学委那儿，还没定题目的今天先报个暂定名”*

未微调的通用小模型在处理这类通知时，常有两大痛点：
1. 你每次必须在提示词里塞满几百字的格式警告和 Few-Shot 样例（白白消耗显存与首字等待时间）；
2. 遇到多重时间干扰或口语省略负责人时，偶尔还是会多吐寒暄废话、写漏引号，或者把“未提及负责人”瞎填成“所有人”。

通过 **200～300 条高质量中文抽取样本的 LoRA 微调**，我们能让模型形成 **“零冗余提示词、张口就是 100% 标准合法 JSON、缺项精准填 `待确认`”** 的肌肉记忆！

---

## 二、Step 1：数据准备头号红线 —— 先过「隐私正则脱敏清洗机」！

> [!CAUTION]
> **数据安全工程铁律**：  
> 无论你是整理班级/公司真实历史群通知，还是在 `5.5` 节导出个人微信记录训练 `WeClone` 分身，**绝对禁止把包含真实手机号、身份证号、银行卡号、邮箱密码或家庭住址的原始文本直接喂进模型训练！** 因为大模型在微调时可能会记住特定字符串！

在 `~/ai_workspace/fine_tune_lab/` 下保存并运行下面这段 **25 行中文敏感信息自动脱敏与训练/测试集严格切分脚本** `prepare_office_sft.py`（严格切出 **训练集 `train`** 与 **绝不参与训练的独立测试集 `test.jsonl`**！）：

```python
"""prepare_office_sft.py — 中文敏感信息正则脱敏 + 办公通知抽取训练集/测试集生成"""
import json
import re
from pathlib import Path

def desensitize_cn_text(text: str) -> str:
    """清洗手机号、18位身份证号与长串银行卡号"""
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "<手机号已脱敏>", text)
    text = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "<证件号已脱敏>", text)
    text = re.sub(r"(?<!\d)\d{16,19}(?!\d)", "<卡号已脱敏>", text)
    return text

SYSTEM_INST = "请从以下群通知提取 JSON（严格仅输出合法 JSON，包含字段：事项、截止时间、负责人；若原文未提及某字段则填“待确认”）："

# 示例种子样本（实际实操可结合魔搭 MSAgent-Bench / InstructIE 中文抽取子集扩充至 200～500 条！）
RAW_PAIRS = [
    ("@全体成员 麻烦大家本周四下午17:00前把开源之夏结项报告PDF发给助教张伟（电话13812345678）汇总！",
     {"事项": "提交开源之夏结项报告PDF", "截止时间": "本周四下午17:00", "负责人": "张伟"}),
    ("各位老师，实验室安全自查表请务必在周五下班前交到B302办公室，别忘了签字！",
     {"事项": "提交签字版实验室安全自查表", "截止时间": "周五下班前", "负责人": "待确认"}),
    ("通知：关于征集年度优秀AI案例的事宜，有意向的同学请找团支书林晓报名，截止时间暂未最终敲定，等后续通知。",
     {"事项": "报名年度优秀AI案例征集", "截止时间": "待确认", "负责人": "林晓"}),
    ("【紧急】明早9:30前各小组把数据库课设源码压缩包传给课代表周杰，逾期的平时分扣5分！",
     {"事项": "提交数据库课设源码压缩包", "截止时间": "明早9:30", "负责人": "周杰"}),
]

out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

# 构造 LLaMA-Factory 标准 Alpaca 训练集（演示将种子样本扩增供快速跑通管线；真实训练建议 >=200 条不重复样本）
train_data = [
    {
        "instruction": SYSTEM_INST,
        "input": desensitize_cn_text(notice),
        "output": json.dumps(ans, ensure_ascii=False),
    }
    for notice, ans in RAW_PAIRS[:3] * 40
]
(out_dir / "office_json_train.json").write_text(json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8")

# 严格留出模型从未见过的 Hold-out 测试集供 5.4 节客观打分！
holdout_samples = [
    {"prompt": f"{SYSTEM_INST}\n{desensitize_cn_text(RAW_PAIRS[3][0])}", "gold": RAW_PAIRS[3][1]},
    {
        "prompt": f"{SYSTEM_INST}\n大家注意啦，11月团建意向投票下周一中午12点截止，还没填问卷的赶紧私聊行政陈静！",
        "gold": {"事项": "填写11月团建意向投票问卷", "截止时间": "下周一中午12点", "负责人": "陈静"},
    },
    {
        "prompt": f"{SYSTEM_INST}\n下学期选修课退补选系统今晚22:00关闭，请有需要的同学自行登录教务网操作。",
        "gold": {"事项": "登录教务网办理选修课退补选", "截止时间": "今晚22:00", "负责人": "待确认"},
    },
]
with (out_dir / "office_json_test.jsonl").open("w", encoding="utf-8") as f:
    for item in holdout_samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 注册到 dataset_info.json
info_path = out_dir / "dataset_info.json"
info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
info["office_json_train"] = {"file_name": "office_json_train.json"}
info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print("✅ 隐私脱敏与训练/测试集切分完成！（测试集已严格物理隔离）")
```

---

## 三、Step 2：用一份标准 YAML 配方一键可复现训练！

在工程项目里，相比每次去网页手点十几个框，把训练配置保存成一份干净的 **YAML 配方文件** `office_lora_sft.yaml`，别人拿到你的仓库一行命令就能 100% 复现你的训练结果（16G+ 卡推荐 `Qwen/Qwen3.5-9B`，8G～12G 卡将第一行改为 `google/gemma-4-E4B-it` 并把 `template` 设为 `gemma`）：

```yaml
### model
model_name_or_path: Qwen/Qwen3.5-9B

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

### dataset
dataset_dir: ./data
dataset: office_json_train
template: qwen
cutoff_len: 1024
overwrite_cache: true

### output
output_dir: ./saves/qwen35-9b-office-lora
logging_steps: 5
save_steps: 50
plot_loss: true
overwrite_output_dir: true

### train (ROCm 10 单卡稳健黄金参数)
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
report_to: swanlab
```

在终端一行命令点火训练：

```bash
USE_MODELSCOPE_HUB=1 llamafactory-cli train office_lora_sft.yaml
```

短短几分钟后，`./saves/qwen35-9b-office-lora` 目录下就会生成只占几十 MB 的专属中文抽取 LoRA 权重与漂亮的 `training_loss.png` 曲线图！

---

## 四、Step 3：合并 LoRA 权重并无缝接回三大标准推理引擎！

训出来的 LoRA 便利贴怎么交给我们第 1、2 章的 `vLLM` 或 `llama-server` 让 WorkBuddy 真正用上呢？有两种标准姿势：

### 姿势 A（最通用推荐）：一键合并（Merge）导出完整独立模型

新建 8 行导出配置 `office_lora_merge.yaml`：

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
adapter_name_or_path: ./saves/qwen35-9b-office-lora
template: qwen
finetuning_type: lora
export_dir: ./merged_models/Qwen3.5-9B-Office-Merged
export_size: 4
export_device: cpu
export_legacy_format: false
```

执行合并命令：
```bash
USE_MODELSCOPE_HUB=1 llamafactory-cli export office_lora_merge.yaml
```
合并完成后：
- **如果你用 `vLLM`（`:8000`）**：直接 `vllm serve ./merged_models/Qwen3.5-9B-Office-Merged --served-model-name qwen3.5-office` 即可满速上线！（还能直接交给第 6.2 节的 **AMD Quark** 做 4-bit 压缩瘦身！）
- **如果你用 `llama-server`（`:8080`）或 `lemonade-server`（`:13305`）**：使用 `llama.cpp` 自带的 `python convert_hf_to_gguf.py ./merged_models/Qwen3.5-9B-Office-Merged --outtype q8_0` 即可转成单文件 `.gguf`！

### 姿势 B（`vLLM` 多适配器热插拔）：不合并直接动态挂载

在 `vLLM` 启动时只需加上 `--enable-lora --lora-modules office-json=./saves/qwen35-9b-office-lora`，就能在一份底座显存上同时挂载十几个不同功能的 LoRA 便利贴！

---

## 👉 下一节：用真实客观数据说话！你的微调到底涨了多少分？

模型训完合并好了，千万别只靠拍脑袋说“我觉得变准了”！下一节我们立刻运行本仓库自带的自动化评测脚本，在刚才严格物理隔离的测试集上跑出硬核的 **微调前 vs 微调后对比成绩单**：

➡️ **[5.4 拒绝凭感觉吹牛！自建中文测试集微调前后客观评测实战](./04-eval-before-after.md)**
