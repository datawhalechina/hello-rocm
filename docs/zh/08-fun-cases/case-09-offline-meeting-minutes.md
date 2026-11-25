# Case 09：全离线保密会议与网课实时纪要神器 —— `Meetily` + 本地语音转写闭环

> 🏷️ **场景分类**：语音交互  
> ⏱️ **预计耗时**：30 分钟 ｜ 🎮 **最低硬件档位**：🟢 入门档 8G 独显 / 锐龙 AI 笔记本（拔掉网线也能全程转写总结！）

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：一段内部项目闭门研讨会录音、导师组会语音指导，或一节 45 分钟没有字幕的专业网课音频（`.mp3` / `.wav` 或麦克风实时收音）；
  - **AI 自动执行**：语音流**一秒不传任何云端服务器**！先由本地 `Faster-Whisper`（或支持原生音频输入的 **`Gemma 4 E4B`**）完成高精度语音转文字，再交由本机 `/v1` 大模型服务提炼核心决议与责任人待办表；
  - **最终交付物**：开源桌面会议助手 **[`Zackriya-Solutions/meetily`](https://github.com/Zackriya-Solutions/meetily)** 可视化纪要库，或一键导出的《保密会议核心结论与 Action Items 待办清单.md》！

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Windows 11 / Ubuntu 24.04（🟢 8G 即可同时跑 Whisper Small + 4B/9B 大模型！） |
| **底层推理服务引擎** | `llama-server`（`:8080`）/ `lemonade-server`（`:13305`）/ `vLLM`（`:8000`） |
| **推荐主体模型** | 🔵 **`Qwen3.5-9B`**（中文口语口水话清洗与待办抽取极准！）或 🟢 **`Gemma 4 E4B`** |
| **本地语音转写前端** | 图形桌面首选开源 [`Meetily`](https://github.com/Zackriya-Solutions/meetily)；脚本批量转写首选 `faster-whisper` |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

### 路线 A：桌面可视化开箱即用 —— `Meetily` 直连本机 `:8080/v1`

1. 前往 [`Zackriya-Solutions/meetily` Releases](https://github.com/Zackriya-Solutions/meetily/releases) 下载对应系统的桌面安装包并启动；
2. 打开 `Settings -> Model Settings`：
   - **Transcription（本地转写模型）**：选择内置本地 `Whisper (base / small)`；
   - **Summary LLM（纪要总结模型）**：选择 **`Custom OpenAI Compatible Endpoint`**，填入 `http://127.0.0.1:8080/v1`，API Key 填 `local-rocm`，Model 填 `qwen3.5-9b`（或 `gemma-4-e4b`）；
3. 点击红色录音键对着麦克风试讲 1 分钟本周计划，点击 `Stop & Generate Summary` 即可秒出结构化纪要！

### 路线 B：35 行批量录音/网课文件一键转《结构化纪要.md》脚本

```bash
mkdir -p ~/ai_workspace/sandbox/case09_audio && cd ~/ai_workspace/sandbox/case09_audio
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple faster-whisper openai
export HF_ENDPOINT="https://hf-mirror.com"
```

新建 `offline_minutes.py`：

```python
"""offline_minutes.py — 本地 Faster-Whisper 转写 + 本机 AMD 大模型结构化纪要提炼"""
import os, sys
from faster_whisper import WhisperModel
from openai import OpenAI

audio_path = sys.argv[1] if len(sys.argv) > 1 else "sample_meeting.mp3"
print(f"🎙️ 1. 正在本地离线转写音频: {audio_path} ...")
asr = WhisperModel("small", device="cpu", compute_type="int8")  # 不抢大模型显存，CPU int8 极快！
segments, _ = asr.transcribe(audio_path, language="zh", vad_filter=True)
transcript = "\n".join(f"[{s.start:.1f}s] {s.text}" for s in segments)

print("🧠 2. 正在调用本机 AMD 大模型提炼结构化会议纪要...")
client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"), api_key="local-rocm")
resp = client.chat.completions.create(
    model=os.getenv("ROCM_MODEL", "qwen3.5-9b"),
    temperature=0.1,
    messages=[{
        "role": "user",
        "content": (
            "请严格依据以下录音转写稿输出 Markdown 会议纪要，包含：\n"
            "1. 【三句话核心决议】；2. 【待办事项表（事项 | 负责人 | 截止节点，未提及填待确认）】；\n"
            f"<transcript>\n{transcript}\n</transcript>"
        ),
    }],
)
out_md = f"{audio_path}_纪要.md"
open(out_md, "w", encoding="utf-8").write(resp.choices[0].message.content)
print(f"✅ 保密纪要已生成：{out_md}")
```

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **拔网线全离线验证**：关闭电脑 Wi-Fi 后对一段含口头禅的 2 分钟测试录音执行转写与总结，确认正常产出干净无废话的纪要。
- [x] **待办字段核验**：确认录音中提到的具体人名与时间节点准确落入【待办事项表】，未提及截止时间的条目准确标为 `待确认`。

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（15 分钟）**：在提示词中加入你们专业的英文缩写词表（如 `ROCm, KV Cache, LoRA`），让模型自动纠正语音同音字（如把“罗拉微调”自动修正为“LoRA 微调”！）。
- **⭐⭐ 白银改造（30 分钟）**：联动 [Case 07（知识管理 MCP）](./case-07-obsidian-knowledge-mcp.md)，每次开完组会自动将纪要卡片写入 Obsidian `00-收件箱` 并打上 `#组会纪要` 标签。
- **⭐⭐⭐ 黄金改造（进阶深造）**：想深入学习 Whisper 语音家族、FunASR 中文语音识别与音频大模型部署？
  - 🔗 **进阶学习入口**：**[Datawhale `self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：为什么脚本里 `WhisperModel` 设了 `device="cpu", compute_type="int8"`？**
  - **原因**：在 8G 显存机器上，大模型占去约 5GB 显存后，把轻巧的 `Whisper-small` 放在多核 CPU `int8` 运行不仅达 5～8 倍实时倍速，还能把 GPU 显存 100% 留给大模型处理万字长转写稿不 OOM！

---

👉 下一篇案例：**[Case 10｜桌面 Live2D 实时语音伴侣与英语口语陪练（`Open-LLM-VTuber`）](./case-10-live2d-voice-tutor.md)**
