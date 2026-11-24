# 2.2.1 接入 WorkBuddy 桌面办公智能体：三大引擎配置与五大避坑指南

> **本节目标**：安装桌面全能办公智能体 **WorkBuddy**，将你在第 1 章启动的本地 AMD 大模型（`llama-server` / `lemonade-server` / `vLLM`）完美接入其 **Craft 智能体工作台**，并一次性掌握 **5 个让新手 100% 零报错的黄金避坑口诀**！  
> *(本节工作流深度参考并对齐 [Datawhale WorkBuddy 官方实战教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)。)*

---

## 一、认识 WorkBuddy：为什么它是零代码同学的最佳拍档？

传统的聊天软件只能“你问一句、它答一段文字”，最后还是得你自己苦哈哈地打开 PowerPoint 一个个画文本框、打开 Excel 一格格复制粘贴。

而 **WorkBuddy** 是一款专为日常办公与知识管理打造的**桌面级 Agent 工作台**，它内置了两大核心模式：
1. **💬 Chat（对话问答模式）**：适合快速翻译、润色段落、头脑风暴；
2. **🛠️ Craft（智能体行动模式 —— 本章主力舞台！）**：能够安全挂载你指定的本地沙箱文件夹（`sandbox`），自主加载 **技能包（Skills，例如大师 PPT `dashi-ppt-skill`）**、连接 **MCP 工具（第 4 章你的个人知识库）**，并支持配置 **定时任务（Scheduled Tasks）** 替你读写本地文档与表格！

前往 WorkBuddy 官网下载对应 Windows 或 Linux/macOS 桌面客户端安装并打开后，先在左侧将工作目录绑定到我们在 [2.1 节](./index.md) 创建好的安全沙箱文件夹：**`D:\ai_workspace\sandbox`**（或 **`~/ai_workspace/sandbox`**）。

---

## 二、把本地 AMD 大模型接进 WorkBuddy（图形界面法 & 专业 JSON 一键法）

请先确认你在第 1 章启动的本地引擎正在后台运行（比如 `llama-server` 监听 `8080`、`lemonade-server` 监听 `13305` 或 `vLLM` 监听 `8000`）。你可以任选下面两种方式完成连接：

### 方式 A：图形界面点选配置（适合只配 1 个常用模型）

1. 点击 WorkBuddy 左下角 ⚙️ **「设置（Settings）」 → 「模型设置（Models）」 → 「添加自定义模型（Add Custom Model）」**；
2. 对照下表逐项填写（以 `llama-server` 启动的 `gemma-4-e4b` 或 `qwen3.5-9b` 为例）：

| 界面配置项 | 填写内容（直接照抄！） | 关键避坑说明 |
|:---|:---|:---|
| **协议类型（Provider）** | `OpenAI Compatible`（OpenAI 兼容） | 三大引擎统一选此标准协议 |
| **接口地址（Base URL）** | • `llama-server`：`http://127.0.0.1:8080/v1`<br/>• `lemonade-server`：`http://127.0.0.1:13305/v1`<br/>• `vLLM`：`http://127.0.0.1:8000/v1` | ⚠️ **图形界面里结尾只写到 `/v1`**（软件会自动在后面拼接 `/chat/completions`）！ |
| **API Key（密钥）** | `local-rocm` | ⚠️ **千万不能留空！** 虽然本地服务不扣钱，但留空会被客户端前端校验直接拦截报错，随便填个非空英文串如 `local-rocm` 即可！ |
| **模型 ID（Model Name）** | `gemma-4-e4b` 或 `qwen3.5-9b`（或 `Gemma-4-E4B-it-GGUF`） | 必须与第 1 章启动命令里的 `--alias` / `--served-model-name`（或 `/v1/models` 返回的 `id`）一模一样 |
| **上下文窗口（Max Context）** | **`32768`** | ⚠️ **必须改成 `32768`！** 默认 `4096` 会把 Craft 工具说明书直接截断！ |
| **工具调用（Tool Call / Function Calling）** | **✅ 开启（勾选）** | 不勾选的话，AI 在 Craft 模式里只会空谈步骤、没法真正动手生成文件！ |

---

### 方式 B（高手偷懒推荐！）：直接套用本仓库已验证的 `models.json` 全家桶配置

如果你想像切换电视频道一样，在 WorkBuddy 里随时一键切换 **三大引擎 × 四大主体模型（`Gemma 4 E4B` / `Qwen3.5-9B` / `Gemma 4 12B` / `Qwen3.6-35B-A3B`）**，可以直接使用本仓库经过自动化单测校验的标准模板 **[`src/configs/workbuddy-models.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/workbuddy-models.json)**！

> [!CAUTION]
> **全书第一号避坑重点（必看！）**：  
> 当你直接编辑底层的 **`~/.workbuddy/models.json`**（Windows 位于 `%USERPROFILE%\.workbuddy\models.json`）配置文件时，里面的 **`url` 字段代表完整请求端点，必须写全 `/v1/chat/completions` 后缀**（而在上面的图形界面 Base URL 输入框里只写到 `/v1`）！写错后缀是新手遇到 `404 Not Found` 的头号元凶！

查看并复制以下完整配置写入 `~/.workbuddy/models.json`：

```json
{
  "models": [
    {
      "id": "rocm-llama-gemma4-e4b",
      "name": "Local ROCm - Gemma 4 E4B (llama-server :8080 入门/轻薄本首选)",
      "provider": "openai-compatible",
      "model": "gemma-4-e4b",
      "url": "http://127.0.0.1:8080/v1/chat/completions",
      "apiKey": "local-rocm",
      "maxContextTokens": 32768,
      "maxOutputTokens": 8192,
      "supportsToolCall": true,
      "supportsVision": true
    },
    {
      "id": "rocm-llama-qwen35-9b",
      "name": "Local ROCm - Qwen3.5-9B (llama-server :8080 16G甜点中文办公主力)",
      "provider": "openai-compatible",
      "model": "qwen3.5-9b",
      "url": "http://127.0.0.1:8080/v1/chat/completions",
      "apiKey": "local-rocm",
      "maxContextTokens": 32768,
      "maxOutputTokens": 8192,
      "supportsToolCall": true,
      "supportsVision": false
    },
    {
      "id": "rocm-lemonade-gemma4-e4b",
      "name": "Local Ryzen AI - Gemma 4 E4B (lemonade-server :13305)",
      "provider": "openai-compatible",
      "model": "Gemma-4-E4B-it-GGUF",
      "url": "http://127.0.0.1:13305/v1/chat/completions",
      "apiKey": "local-rocm",
      "maxContextTokens": 32768,
      "maxOutputTokens": 8192,
      "supportsToolCall": true,
      "supportsVision": true
    },
    {
      "id": "rocm-lemonade-qwen36-35b",
      "name": "Local Strix Halo 旗舰 - Qwen3.6-35B-A3B (lemonade-server :13305)",
      "provider": "openai-compatible",
      "model": "Qwen3.6-35B-A3B-GGUF",
      "url": "http://127.0.0.1:13305/v1/chat/completions",
      "apiKey": "local-rocm",
      "maxContextTokens": 32768,
      "maxOutputTokens": 8192,
      "supportsToolCall": true,
      "supportsVision": false
    },
    {
      "id": "rocm-vllm-qwen35-9b",
      "name": "Local ROCm - Qwen3.5-9B (vLLM :8000 高并发主力)",
      "provider": "openai-compatible",
      "model": "qwen3.5-9b",
      "url": "http://127.0.0.1:8000/v1/chat/completions",
      "apiKey": "local-rocm",
      "maxContextTokens": 32768,
      "maxOutputTokens": 8192,
      "supportsToolCall": true,
      "supportsVision": false
    }
  ]
}
```

---

## 三、1 分钟牵手验收：让本地模型在沙箱里写下第一张问候卡！

切换到 WorkBuddy 的 **Craft 模式**，在顶部模型下拉框选中你刚刚配好的本地模型（如 `Local ROCm - Gemma 4 E4B` 或 `Qwen3.5-9B`），输入下面这句测试指令：

```text
请检查当前沙箱目录，新建一个 hello_amd.md 文件，在里面写下今天的日期和一句“我的本地 AMD AI 办公助手已成功接通 WorkBuddy！”，写完后读取确认告诉我。
```

观察右侧工作区：当你看到模型自动弹出 **`write_file` 工具调用卡片**，并在你的 `sandbox` 文件夹里真实生成了 `hello_amd.md` 文件时——**大功告成！你的本地显卡已经正式长出了操作办公软件的“双手”！**

---

## 🚑 WorkBuddy 接入本地模型「五大高频深坑」速查表

如果刚才那一步报错或没反应，100% 逃不出下面这 5 个原因，按表 10 秒钟即可药到病除：

| 序号 | 症状现象 | 病根定位 | 一秒修复方案 |
|:---:|:---|:---|:---|
| **坑 1** | 报错 `404 Not Found` | `url` 后缀写重或写漏 | 记住口诀：**UI 界面填 `/v1`，`models.json` 文件填 `/v1/chat/completions`**！ |
| **坑 2** | 点击发送立刻弹窗报错 `API Key is required / Unauthorized` | 把 API Key 留空了 | 随便填入非空英文如 `local-rocm` 保存即可。 |
| **坑 3** | Craft 模式发消息报错 `context length exceeded` 或答非所问 | 客户端或服务端有一边忘了开 `32K` 上下文 | 检查两端：① 第 1 章引擎启动时有 `--ctx-size 32768`（或 `--max-model-len 32768`）；② WorkBuddy 里 `maxContextTokens` 设为 `32768`。 |
| **坑 4** | AI 只在聊天框里贴代码/文字，不自动创建文件也不调工具 | ① 没切到 **Craft 模式**；② 引擎漏加 `--jinja`；③ `supportsToolCall` 没开 | 切换到 Craft 模式，确认 `llama-server` 带了 `--jinja` 参数（`vLLM` 带了 `--enable-auto-tool-choice`）。 |
| **坑 5** | 提示 `Connection refused` 连不上 | 第 1 章的终端窗口被随手关掉了，或者端口号填错 | 看一眼后台终端是否还在监听对应端口（`8080` / `13305` / `8000`）。 |

---

## 👉 马上进入高光时刻！

连接一通百通！接下来，让我们用全书唯一钦定的 PPT 生成神器 **`dashi-ppt-skill`（大师 PPT）**，3 分钟把一份普通笔记变成惊艳全场的高颜值原生可编辑 PPT：

➡️ **[2.2.2 办公王牌实战一：用 `dashi-ppt-skill`（大师 PPT）3 分钟生成原生可编辑 PPT](./03-workbuddy-dashi-ppt.md)**
