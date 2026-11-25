# Case 08：复习讲义秒变双人对话播客 & `WeClone` 分身联动 —— 全隐私本地版 NotebookLM

> 🏷️ **场景分类**：个人知识库 / 语音学习  
> ⏱️ **预计耗时**：35 分钟 ｜ 🎮 **最低硬件档位**：🟢 入门档 8G / 🔵 甜点档 16G

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：一份枯燥催眠的 30 页专业课复习讲义 PDF、长篇论文或你自己的读书笔记；
  - **AI 自动执行**：作为海外爆款 NotebookLM 的 **100% 本地隐私开源平替**，使用 **[`lfnovo/open-notebook`](https://github.com/lfnovo/open-notebook)**（或下面的 40 行极简双人播客生成脚本 + `edge-tts` / 本地 TTS）连接你的本机 AMD 模型，把干巴巴的讲义改写成两位主持人一捧一逗、充满追问与生活举例的自然对谈台词，并合成为双声道对话 `.mp3` 音频！更酷的是，你还能换上 [第 5.5 节](../05-fine-tune/05-weclone-and-xhs.md) 微调的 **`WeClone` / 网感风格模型**，让播客彻底告别播音腔！
  - **最终交付物**：通勤路上戴着耳机就能把考点听进脑子里的 **《双人聊天式知识点播客.mp3》** 与配套文字逐字稿！

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Windows 11 / Ubuntu 24.04（🟢 8G 独显核显 / 🔵 16G 独显） |
| **底层推理服务引擎** | `llama-server`（`:8080`）/ `lemonade-server`（`:13305`）/ `vLLM`（`:8000`） |
| **推荐主体模型** | 🔵 **`Qwen3.5-9B`**（中文口语对谈节奏感极自然！）或第 5.5 节微调模型；🟢 **`Gemma 4 E4B`** |
| **语音合成（TTS）选择** | 轻量零显存体验：`edge-tts`（双角色男女声切换）；100% 纯断网本地 TTS：`Kokoro-82M` / `GPT-SoVITS` |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

下面提供两种体验路线：**路线 A（45 行 Python 极速出双人播客 MP3，3 分钟听到成品！）** 与 **路线 B（完整部署开源 `Open Notebook` 工作台）**。

### 路线 A（极速推荐）：45 行代码把任意讲义变成男女对谈播客 `.mp3`！

```bash
mkdir -p ~/ai_workspace/sandbox/case08_podcast && cd ~/ai_workspace/sandbox/case08_podcast
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openai edge-tts
```

新建 `notes_to_podcast.py`（直接连接你本机的 `:8080` / `:13305` / `:8000` 服务）：

```python
"""notes_to_podcast.py — 调用本机 AMD 大模型将枯燥讲义改写并合成为双人对话播客 MP3"""
import asyncio, json, os
import edge_tts
from openai import OpenAI

LECTURE_NOTE = """
LoRA（低秩自适应）微调核心原理：冻结预训练大模型 99% 以上的原始权重矩阵 W0 不更新，
在旁路引入两个低秩矩阵 A 和 B（秩 r 远小于原维度 d），仅训练 A 和 B，从而将优化器状态显存开销降低 80% 以上。
推理时可将 B*A 直接加回 W0，实现零额外推理延迟！
"""

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"), api_key="local-rocm")
resp = client.chat.completions.create(
    model=os.getenv("ROCM_MODEL", "qwen3.5-9b"),
    temperature=0.6,
    messages=[{
        "role": "user",
        "content": (
            "请将以下讲义改写为 6 轮自然生动、带幽默比喻的双人中文播客对话（主持人阿杰负责好奇追问，学霸小琳负责用大白话拆解）。\n"
            "严格仅输出 JSON 数组，每项格式为 {\"speaker\": \"阿杰\" 或 \"小琳\", \"text\": \"台词\"}，禁止加括号动作描写：\n"
            f"<context>{LECTURE_NOTE}</context>"
        ),
    }],
)

raw = resp.choices[0].message.content.strip().strip("`").removeprefix("json").strip()
dialogue = json.loads(raw)

async def synth() -> None:
    voices = {"阿杰": "zh-CN-YunxiNeural", "小琳": "zh-CN-XiaoxiaoNeural"}
    with open("podcast_episode.mp3", "wb") as out_f:
        for i, turn in enumerate(dialogue, 1):
            print(f"🎙️ [{turn['speaker']}]: {turn['text']}")
            comm = edge_tts.Communicate(turn["text"], voices.get(turn["speaker"], "zh-CN-XiaoxiaoNeural"))
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    out_f.write(chunk["data"])
    print("\n✅ 双人对话播客已生成：podcast_episode.mp3 ！快戴上耳机听听看！")

asyncio.run(synth())
```

运行 `python notes_to_podcast.py`，不到 20 秒即可双击播放 `podcast_episode.mp3`！

### 路线 B（完整图形工作台）：Docker 启动开源 [`Open Notebook`](https://github.com/lfnovo/open-notebook)

如果你希望拥有完整的图形化多笔记本管理界面，参照 `lfnovo/open-notebook` 官方 `docker-compose.yml` 启动容器，并在其模型设置中将 OpenAI Compatible Endpoint 指向宿主机 `http://host.docker.internal:8080/v1` 即可！

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **音频自然度与双声线验收**：播放 `podcast_episode.mp3`，确认男声“阿杰”与女声“小琳”轮流接话流畅，无生硬朗读 Markdown 符号（如 `**` 或 `#`）。
- [x] **核心考点保真度验收**：对照原讲义，确认“冻结原权重、省优化器显存、合并零额外推理延迟”三个核心得分点全部在对话中讲清且无事实扭曲。

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（15 分钟）**：把你下周要背的马原/毛概/计算机网络复习提纲塞进 `LECTURE_NOTE`，生成 5 分钟睡前磨耳朵音频发到手机上听。
- **⭐⭐ 白银改造（40 分钟）**：联动 [第 5.5 节](../05-fine-tune/05-weclone-and-xhs.md)，加载你微调后的 **网感/个人分身 LoRA 模型** 生成播客台词，调用 `has_ai_cliche()` 对比微调前后“播音客服腔”的消除效果！
- **⭐⭐⭐ 黄金改造（进阶深造）**：想深入探索大模型底层词表分词、生成采样策略（Temperature / Top-p / Repetition Penalty）与对齐训练原理？
  - 🔗 **进阶学习入口**：**[Datawhale `happy-llm`（从零开始的大语言模型原理与实践教程）](https://github.com/datawhalechina/happy-llm)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：`json.loads(raw)` 报错解析失败？**
  - **原因**：个别小模型在 JSON 数组前后加了一句说明文字。
  - **一行解决**：复用本仓库 `src/main-assistant/eval/eval_before_after_zh.py` 中的正则提取技巧截取 `[...]` 段即可稳健解析。

---

👉 进入下辑精彩语音、编程与游戏案例：**[Case 09｜全离线保密会议与网课实时纪要神器（`Meetily` + `Faster-Whisper`）](./case-09-offline-meeting-minutes.md)**
