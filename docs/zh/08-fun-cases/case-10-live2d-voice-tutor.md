# Case 10：桌面 Live2D 实时语音伴侣与英语口语陪练 —— 随时插话打断的零尴尬口语私教

> 🏷️ **场景分类**：语音交互  
> ⏱️ **预计耗时**：35 分钟 ｜ 🎮 **最低硬件档位**：🟢 入门档 8G（`Open-LLM-VTuber` 轻量组合）/ 🔵 甜点档 16G（可选 `MiniCPM-o` 原生全双工）

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：扔掉键盘！直接对着麦克风开口说英语或日常聊天（甚至可在 AI 说话中途像真人对话一样随时开口插话打断！）；
  - **AI 自动执行**：开源高人气虚拟桌面伴侣 **[`Open-LLM-VTuber`](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)** 通过本地 VAD + ASR 听懂你的话 → 交由本机高速运行的 **`Gemma 4 E4B` / `Qwen3.5-9B`** 生成简短口语回复与表情标签 → 驱动桌面 Live2D 角色同步口型与神态回应；
  - **最终交付物**：一位驻扎在你电脑桌面上、**24 小时免费、绝不笑话你语法错误的全天候雅思口语考官 / 英文面试陪练**！

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Windows 11 / Ubuntu 24.04（务必戴普通耳机防外放回声自激！） |
| **底层推理服务引擎** | **`llama-server`（`:8080`）** 或 **`lemonade-server`（`:13305`）**（低首字延迟体验极丝滑） |
| **推荐主体模型** | 🟢 **`Gemma 4 E4B`**（英文口语极地道，吐字 >60 `tok/s` 零冷场！）或 🔵 **`Qwen3.5-9B`** |
| **原生全双工进阶备选** | 🔵 16G～24G 独显玩家亦可体验端到端原生多模态 [`OpenBMB/MiniCPM-o`](https://github.com/OpenBMB/MiniCPM-o) |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

### Step 1：确认本机低延迟模型服务在线

```bash
curl -s http://127.0.0.1:8080/v1/models
```

### Step 2：克隆并初始化 `Open-LLM-VTuber`

```bash
cd ~/ai_workspace
git clone https://github.com/Open-LLM-VTuber/Open-LLM-VTuber.git
cd Open-LLM-VTuber
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv sync
```

### Step 3：修改 `conf.yaml` 对接本机 `:8080/v1` 并注入“英语口语私教”人设

首次执行 `uv run run_server.py` 生成默认 `conf.yaml` 后，将其中 LLM 配置段指向本机服务：

```yaml
character_config:
  conf_name: "amd_english_coach"
  persona_prompt: |
    You are a warm, encouraging English speaking & tech interview coach running locally on the user's AMD PC.
    1. Keep every spoken reply concise (2-3 natural sentences max) so conversation flows fast!
    2. Gently point out at most ONE phrasing improvement, then ask a natural follow-up question.
  agent_config:
    conversation_agent_choice: "basic_memory_agent"
    agent_settings:
      basic_memory_agent:
        llm_provider: "openai_compatible_llm"
    llm_configs:
      openai_compatible_llm:
        base_url: "http://127.0.0.1:8080/v1"
        llm_api_key: "local-rocm"
        model: "gemma-4-e4b"
        temperature: 0.7
```

重新启动 `uv run run_server.py`，浏览器打开 `http://127.0.0.1:12393`（或桌面透明悬浮窗客户端），戴上耳机开口练起来吧！

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **语音响应延迟测试**：说完一句话停顿后，确认在 **2 秒左右** 听到角色自然开口且 Live2D 口型同步。
- [x] **插话打断（Barge-in）测试**：在 AI 朗读中途直接开口说 *"Hold on, let me try again"*，确认 AI 立即停止播报并转入倾听状态。

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（15 分钟）**：把 `persona_prompt` 换成“秋招技术面试官”，把 [第 7.3 节](../07-portfolio-resume/03-star-resume-interview.md) 的 6 道高频追问题放进去做全真语音模拟面试！
- **⭐⭐ 白银改造（40 分钟）**：换上你在 [第 5.5 节](../05-fine-tune/05-weclone-and-xhs.md) 微调的中文角色 LoRA 模型，打造语气鲜明的桌面伴学搭子。
- **⭐⭐⭐ 黄金改造（进阶深造）**：深入探索端到端实时视觉+语音全双工开源模型原理与部署：
  - 🔗 **进阶学习入口**：**[Datawhale `self-llm`（多模态与语音大模型部署专题）](https://github.com/datawhalechina/self-llm)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：AI 刚开口说两个词就自己打断自己停下了？**
  - **原因**：使用外放喇叭导致 AI 自己的声音又钻进了麦克风，误触发了语音活动检测（VAD）。
  - **一行解决**：**戴上耳机**即可 100% 解决！

---

👉 下一篇案例：**[Case 11｜`Qoder` 全离线 AI 编程搭档：内网代码重构与单测生成](./case-11-qoder-offline-ide.md)**
