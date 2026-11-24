# 5.4～5.6 中文通知抽取主线微调、自建测试集评估与 `WeClone`/小红书网感分身

---

## 5.4 主线实战：基于 `MSAgent-Bench` 与 `InstructIE` 微调中文通知抽取与工具调用

针对第 4 章实战中发现的 **“口语化中文办公/课程群通知抽 JSON 偶发漏字段、工具调用格式不稳”** 痛点，我们可以直接组合魔搭社区两大原生中文数据集：
- **浙大 `ZJUNLP/InstructIE`**：高质量中文非结构化文本信息抽取数据集；
- **魔搭官方 `damo/MSAgent-Bench`**：原生中文多工具调用（Function Calling）标准训练集。

### 60 行命令行一键复现配置（`LLaMA-Factory` CLI 通道）

除了点鼠标的 WebUI，工程上更推荐保存一份可复现的 YAML 配置（可直接写进你的 GitHub 作品包）：

```bash
export USE_MODELSCOPE_HUB=1

llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path Qwen/Qwen3.5-9B \
  --dataset alpaca_zh_demo \
  --template qwen3 \
  --finetuning_type lora \
  --lora_target all \
  --lora_rank 16 \
  --output_dir ./saves/qwen3.5-9b-office-ie-lora \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1.0e-4 \
  --num_train_epochs 3.0 \
  --bf16 true \
  --report_to swanlab
```

---

## 5.5 【简历加分核心】自建中文测试集评估微调前后效果 & 导出重接服务

面试官看微调项目时，最看重的从来不是“我跑通了训练脚本”，而是 **“你在独立测试集上把任务准确率从多少提升到了多少”**！

### Step 1：合并导出微调权重并拉起服务

在 `LLaMA-Factory` 中执行 `llamafactory-cli export` 合并 LoRA 权重（或使用 `llama.cpp` 的 `convert_hf_to_gguf.py` 转为 GGUF），重新用第 1 章的 **`llama-server` / `vLLM` / `lemonade-server`** 启动服务。

### Step 2：一键运行本仓库内置的中文前后对比评测脚本

本仓库已为你备好零依赖客观评测脚本 **[`src/main-assistant/eval/eval_before_after_zh.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/eval/eval_before_after_zh.py)**，分别在加载微调权重前后各跑一次：

```bash
python3 src/main-assistant/eval/eval_before_after_zh.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3.5-9B
```

将两次运行输出的 **`JSON 格式合法率`** 与 **`关键字段准确率`**（例如：从微调前的 `76.0%` 跃升至微调后的 `98.0%`）填入你的《微调前后效果评估报告》，这就是最有说服力的工程证据！

---

## 5.6 趣味进阶实战（网感与分身）：`WeClone` 微信聊天记录数字分身 / 小红书爆款种草体

除了严肃的办公抽取，最受同学欢迎、最适合晒朋友圈的个性化中文微调方向就是 **「网感与数字分身」**！本节聚焦这一高人气赛道提供两种任选玩法：

1. **玩法一：个人微信聊天记录数字分身（参考开源明星项目 [`WeClone`](https://github.com/xming521/WeClone)）**
   - 导出自己与好友的日常聊天记录 → 执行严格隐私清洗（剔除手机号、密码、转账敏感词）→ 将连续发出的多条短消息拼接为自然轮次进行 LoRA 微调，打造口头禅和你一模一样的本地赛博分身！
   - *零门槛平替体验*：如果不想导出个人微信，可直接使用本仓库内置的经典角色台词集 **[`src/fine-tune/datasets/huanhuan-100.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/fine-tune/datasets/huanhuan-100.json)**（参考 [Chat-甄嬛微调实录](../05-amd-yes/huanhuan-chat.md)）快速体验语气迁移。
2. **玩法二：小红书爆款种草文案机**
   - 从魔搭社区拉取高赞中文种草笔记语料微调，彻底消灭大模型张嘴就是“总而言之、综上所述、作为一个人工智能”的机械客服腔！
   - **科学评测指标**：调用 `eval_before_after_zh.py` 中的 `has_ai_cliche()` 函数统计 **「AI 客服破功词出现率」**（如从 `42%` 降至 `0%`），并找 3 位朋友做 10 题双盲测试猜哪条是真人写的！

---

## 🔗 本章课程引用与进阶导流：直通 `happy-llm`

在本章你已经亲手跑通了人生第一次中文 LoRA 微调与科学评测！如果你心中升起了更深的好奇：
- **“大模型内部的 Attention 注意力层和 MLP 到底是怎么一行行代码搭出来的？”**
- **“从零开始做预训练（Pre-training）、全参数 SFT 指令微调以及 DPO / GRPO 强化学习偏好对齐，底层数学与工程是怎么运转的？”**

强烈推荐在完成本章后，无缝衔接进入 Datawhale 现象级王牌教程 —— **[`happy-llm`（从零开始的大语言模型原理与实践教程）](https://github.com/datawhalechina/happy-llm)**（本仓库第 9 章亦同步收录了在 AMD 多卡环境复现 [Happy-LLM 训练全流程](../05-amd-yes/happy-llm/index.md) 的配套指南；若想查阅更多单模型微调案例也可结合 [`self-llm`](https://github.com/datawhalechina/self-llm)）！

👉 搞定了“更聪明”，下一步我们零 C++ 代码解决“不够快”：**[第 6 章｜AMD 量化对比、零 C++ 算子加速与 `Hyperloom` 自动优化](../06-optimize-speed/index.md)**！
