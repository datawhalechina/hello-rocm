# 5.4 拒绝凭感觉吹牛！自建中文测试集微调前后客观评测实战

> **本节目标**：掌握大模型工程师与业余玩家的核心分水岭 —— **用严格隔离的独立测试集（Hold-out Test Set）做自动化前后盲测对比**！亲手运行本仓库内置的 **[`src/main-assistant/eval/eval_before_after_zh.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/eval/eval_before_after_zh.py)**，量化计算出 **JSON 格式合法率**、**关键字段精确匹配率** 与 **AI 客服腔破功词率**，拿到简历第 4 件硬核证据！

---

## 一、为什么面试官一听“我试了三条觉得效果很好”就皱眉头？

在秋招/春招面试中，80% 简历写过“LoRA 微调大模型”的同学都会栽在同一个追问上：
> 🎤 **面试官**：“你怎么证明你的微调真的有效、没有过拟合？微调前和微调后的具体评测指标提升了多少个百分点？”

如果你只能回答*“我自己在网页里问了几个问题，感觉回答得更顺溜了”*，这道题基本就只剩及格分了；  
但如果你能像本节这样淡定地答出下面这三句话，面试官眼里立刻会放光：
1. **数据物理隔离**：*“我在清洗阶段就严格切分并物理隔离了 30～50 条模型训练期间绝对没见过的口语化中文群通知作为 Hold-out 测试集；”*
2. **双维度程序硬指标**：*“编写自动化评测脚本同时考核 **JSON 语法合法解析率（Valid JSON Rate）** 与 **字段级精确匹配准确率（Exact Field Accuracy，含缺失项 `待确认` 召回）**；”*
3. **前后同台控制变量对比**：*“在完全相同的 `temperature=0.1` 与相同提示词下，基座模型字段准确率为 `74.4%`，经 3 Epoch LoRA 微调后提升至 `96.7%`（同时提示词长度缩减 65%）！”*

---

## 二、逐块读懂本仓库评测神器：`eval_before_after_zh.py`

打开本仓库配套源码 **[`src/main-assistant/eval/eval_before_after_zh.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/eval/eval_before_after_zh.py)**，它只用 Python 标准库就实现了稳健的三大核心打分器：

### 1. 宽容提取 + 严格验法的 JSON 解析器（`extract_json_object`）
有些未微调的模型喜欢把 JSON 包在 ```` ```json ... ``` ```` 代码块里，甚至前面带一句废话；而如果大括号写漏了则会触发 `json.JSONDecodeError`。脚本先做代码块剥离，再验证能否合法反序列化为 Python `dict`：

```python
def evaluate_json_sample(pred_text: str, gold: dict[str, Any]) -> dict[str, Any]:
    """评估单条中文通知抽取样本的 JSON 合法性与字段准确率"""
    parsed = extract_json_object(pred_text)
    if parsed is None:
        return {"json_valid": False, "field_accuracy": 0.0}
    if not gold:
        return {"json_valid": True, "field_accuracy": 1.0}
    matched = sum(1 for k, v in gold.items() if str(parsed.get(k, "")).strip() == str(v).strip())
    return {
        "json_valid": True,
        "field_accuracy": matched / len(gold),
    }
```

### 2. 专为下一节「网感分身」准备的“AI 客服腔破功词雷达”（`has_ai_cliche`）
当你在 `5.5` 节训练 `WeClone` 微信分身或小红书种草文案时，最扫兴的就是模型突然蹦出一句 *“作为一个人工智能语言模型，我没有个人情感”* 或 *“希望以上建议对您有所帮助”*。脚本内置了高频机械客服词库一键扫雷！

---

## 三、三步跑出你的《微调前 vs 微调后客观评测对比表》

### Step 1：先跑通评测器自身的单元测试（确保尺子本身是准的！）

```bash
cd ~/ai_workspace/hello-rocm
PYTHONPATH=. python tests/test_course_v2.py TestChineseFineTuneEval -v
```

看到 `test_json_extraction_and_persona_metrics ... ok`，说明正常代码块解析、残损括号降级为 0 分以及客服腔雷达全部校准无误！

### Step 2：测出「微调前（原版基座模型）」的基准分（Baseline）

确保你在第 1 章启动的原版基座模型服务（例如 `:8080` 上的 `qwen3.5-9b` 或 `gemma-4-e4b`）在线，传入我们在 `5.3` 节留出的独立测试集 `office_json_test.jsonl`（若暂未生成文件，不加 `--testset` 参数也会自动运行内置演示测试题！）：

```bash
python src/main-assistant/eval/eval_before_after_zh.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen3.5-9b \
  --testset ~/ai_workspace/fine_tune_lab/data/office_json_test.jsonl
```

记下终端打印的两行基准成绩（例如 `JSON 格式合法率: 86.7%`，`关键字段准确率: 73.3%`）。

### Step 3：换上「微调后合并模型」再跑同一套考卷，见证涨分时刻！

将服务端切换为你刚刚在 `5.3` 节合并导出的微调模型（如 `qwen3.5-office`），**一行字不改**再次运行同一条评测命令：

```bash
python src/main-assistant/eval/eval_before_after_zh.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.5-office \
  --testset ~/ai_workspace/fine_tune_lab/data/office_json_test.jsonl
```

---

## 🏆 填写你的《作品集素材 #4：中文微调前后客观评测报告卡》

将你亲手测出的两组数字填入下面这张对比表（连同 `5.2 / 5.3` 节的 `SwanLab` Loss 曲线截图一起保存，第 7 章直接嵌进你的个人作品集主页！）：

| 评测维度（独立测试集 $N = 30$ 条） | 微调前基座（Base） | LoRA 微调后（SFT） | 核心收益变化 |
|:---|:---:|:---:|:---|
| **底座模型与训练耗时** | `Qwen3.5-9B` / `Gemma 4 E4B` | `+ 中文办公抽取 LoRA (r=16)` | 单卡训练仅耗时 **___ 分钟** |
| **JSON 语法一次合法率** | `_____%`（如 `86.7%`） | **`_____%`（目标 `100.0%`）** | 彻底消灭多余寒暄与漏括号报错 |
| **三要素字段精确匹配率** | `_____%`（如 `73.3%`） | **`_____%`（如 `96.7%`）** | 口语省略与 `待确认` 兜底显著跃升 |
| **单次调用所需提示词长度** | ~320 Tokens（需带长示例） | **~65 Tokens（零示例直出）** | 每次调用节省 **~80%** 输入开销 |

---

## 👉 下一节：最受全网欢迎的趣味微调 —— 打造你的微信数字分身与小红书种草机！

严肃的办公 JSON 抽取让你拿到了硬核的工程涨分指标；而在本章压轴的 **5.5 节**，我们将把微调玩出最具社交分享欲的花样：**用你自己的聊天习惯训一个说话一模一样的 `WeClone` 数字分身，或者训一台彻底告别机械客服腔的小红书爆款种草机**！

➡️ **[5.5 趣味进阶专场（网感与分身）：`WeClone` 微信聊天数字分身 / 小红书爆款种草体（导流 `happy-llm`）](./05-weclone-and-xhs.md)**
