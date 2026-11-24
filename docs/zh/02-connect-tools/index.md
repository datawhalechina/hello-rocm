# 第 2 章｜即刻用起来：零代码办公先行 —— 接入 WorkBuddy、Qoder 与 Codex

> **本章目标（首发核心章节）**：告别“只能在终端敲命令聊天”的玩具感！使用本仓库真机验证通过的配置文件，10 分钟内将第 1 章拉起的本地服务接入 **WorkBuddy、Qoder 与 Codex**，零 API 费用完成 **AI PPT 生成、AI 自动填表、定时处理日常办公与全离线辅助编程**！

---

## 2.1 读懂“万能插座”（OpenAI 兼容接口）与本地 Agent 三大安全铁律

### 一、为什么三大引擎能无缝接入上百款 AI 软件？

第 1 章我们用 `llama-server`、`lemonade-server` 或 `vLLM` 启动的服务，本质上都在你的电脑本机（`127.0.0.1`）装上了一个标准的 **“OpenAI 兼容三孔插座”**。任何支持自定义模型的桌面软件，只要填对以下 **4 个参数** 就能直接通电干活：

| 参数名称 | 白话含义 | 对应我们三大引擎的填法 |
|:---|:---|:---|
| **`Base URL`（接口地址）** | 插座在哪个房间号？ | `http://127.0.0.1:8080/v1`（`llama-server`）<br/>`http://127.0.0.1:13305/v1`（`lemonade-server`）<br/>`http://127.0.0.1:8000/v1`（`vLLM`） |
| **`API Key`（密钥）** | 门禁口令是什么？ | 本地不扣费，但**客户端校验要求非空字符串**，统一填 `rocm-local`（Lemonade 填 `lemonade`） |
| **`Model ID`（模型名）** | 点名让哪位助手干活？ | 与服务端启动时的 `--alias` 或 `--served-model-name` 完全一致 |
| **`Tool Call`（工具调用）** | 允许模型动手读写文件吗？ | 必须开启（服务端已加 `--jinja` 或 `--enable-auto-tool-choice`） |

> 📦 **抄作业福利**：本仓库已在 [`src/configs/`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/README.md) 目录为你备好了经过校验的现成配置文件！

---

### 二、⚠️ 动手前必看：本地桌面 Agent 三大安全铁律

WorkBuddy、`Qoder` 与 Codex 具备直接读写本地文件夹、修改表格和执行脚本的真实“手脚”。在让本地小模型操作你的电脑前，请务必遵守以下 **三大安全铁律**：

1. **铁律一：建立独立英文沙盒工作区（严禁直接授权桌面或全盘）**
   - 专门新建一个纯英文路径文件夹（例如 `D:\AI_Workspace\office_demo` 或 `~/ai_workspace/office_demo`），把需要处理的文档、报销单或代码复制进去再拖给 AI。**绝不要把整个 `C:\`、`Desktop` 或唯一原件目录直接交给智能体批量删除/改名！**
2. **铁律二：缓存与大文件远离系统 C 盘**
   - PPT 导出素材、模型权重与临时生成物统一放在独立工作目录，防止系统盘空间告警。
3. **铁律三：批量改名与填表先走 `dry-run`（先预览、后执行）**
   - 凡是涉及批量重命名文件、覆盖已有 Excel 报表的指令，在提示词末尾加上一句：**“请先列出拟修改的预览清单，待我回复确认后再正式写入文件。”**

---

## 本章实战导航

1. 🌟 **[2.2【第一步重点推出】接入 WorkBuddy：AI PPT 生成（`dashi-ppt-skill`）、AI 自动填表与定时办公自动化](./workbuddy-office-suite.md)**
2. 💻 **[2.3 & 2.4 接入 `Qoder` 全离线编程 IDE 与 `Codex CLI` 终端智能体](./qoder-and-codex.md)**

> 🔗 **本章课程引用与进阶导流**：本章 WorkBuddy 办公自动化部分直接联动参考 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**；跑通本章后欢迎前往该专精教程解锁更多高阶桌面自动化工作流！
