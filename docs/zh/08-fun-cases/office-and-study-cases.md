# 第 8 章案例上辑（Case 01 ～ Case 08）：日常办公、论文阅读与个人知识库实战

> 本辑所有案例底层统一使用 [第 1 章](../01-first-model/index.md) 启动的 **`llama-server`（`8080`）**、**`lemonade-server`（`13305`）** 或 **`vLLM`（`8000`）** 服务，客户端配置见 [`src/configs/`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/README.md)。

---

## Case 01：🌟【首发王牌】一句话文档变可编辑汇报 PPT（WorkBuddy + `dashi-ppt-skill`） {#case-01}

- **🎬 效果演示**：拖入一份 Markdown 总结或实验记录，AI 自动规划页面逻辑、匹配商业/学术版式（SWOT、漏斗图、数据卡片），打开浏览器微调台并一键导出 **WPS / PowerPoint 原生可二次编辑的 `.pptx` 文件**。
- **✅ 支持配置**：🟢 8G / 🔵 16G；`llama-server` / `vLLM` / `lemonade-server`（必开 `32768` 上下文 + Tool Call）；WorkBuddy + [`chuspeeism/dashi-ppt-skill`](https://github.com/chuspeeism/dashi-ppt-skill)。
- **💻 代码与步骤**：
  1. 在工作区终端安装技能：`npx dashi-ppt-skill@latest`；
  2. 将配置 [`src/configs/workbuddy-models.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/workbuddy-models.json) 放入 `~/.workbuddy-ai/models.json`；
  3. 在 WorkBuddy Craft 模式发送：*“调用 dashi-ppt-skill 将 `周总结.md` 制作为 8 页汇报 PPT，开启本地预览台并导出可编辑 `output/汇报.pptx`。”*
- **🔍 验证方法**：用 PowerPoint 或 WPS 打开导出的 `.pptx`，双击修改任意标题文字与配色框，确认均为原生矢量可编辑对象而非死图。
- **🚀 扩展任务与导流**：尝试切换学术答辩主题制作组会 PPT；🔗 进阶学习 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**。
- **❓ 试学 FAQ**：*若回复空白？* → 检查服务端是否漏加 `--ctx-size 32768`。

---

## Case 02：【办公必装】AI 自动填表与杂乱文件夹智能归档 {#case-02}

- **🎬 效果演示**：扫描本地沙盒目录中多份杂乱单据、简历或下载文件，自动按类别归档重命名，并将非结构化单据信息严格提取填入汇总 Excel 报表。
- **✅ 支持配置**：🟢 8G 入门档即可；WorkBuddy + 三大引擎任一（推荐开启 `Gemma 4 E4B` 或 16G 卡开启 `Qwen3.5-9B`）。
- **💻 代码与步骤**：
  1. 新建沙盒目录 `office_sandbox/inbox/` 并放入测试报销单与杂乱文档副本；
  2. 第一步归档指令：*“请先以 dry-run 模式列出拟将 `inbox/` 文件按【单据/课件/代码】分类移动的预览表，待我确认后执行并生成《目录盘点清单.md》。”*
  3. 第二步自动填表指令：按 [2.2.3 节自动填表实战](../02-connect-tools/04-workbuddy-excel-and-cron.md) 字段契约提取单据写入 `汇总表.xlsx`（缺失项强制填 `待确认`）。
- **🔍 验证方法**：核对 Excel 列名与源文件金额 100% 一致，故意放入一份无金额单据确认填为 `待确认` 无编造。
- **🚀 扩展任务与导流**：结合第 5 章中文抽取微调模型对比复杂口语报销说明的提取准确率；🔗 进阶学习 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**。

---

## Case 03：【到点自动干活】用 WorkBuddy AI 定时任务自动处理日常办公事项 {#case-03}

- **🎬 效果演示**：无需人工值守，每天早上 9:00 或每周五 17:00，你的 AMD 电脑自动扫描工作目录新增记录、更新统计表格并生成日报/周报。
- **✅ 支持配置**：🟢 8G（锐龙笔记本推荐用 `lemonade-server` 低功耗常驻后台）；WorkBuddy 定时任务模块。
- **💻 代码与步骤**：
  1. 先在普通会话跑通汇总 Prompt 并挂载第 4 章的 [`weekly-office-check` 技能](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/skills/weekly-office-check/SKILL.md)；
  2. 打开 WorkBuddy 定时任务面板，绑定独立沙盒工作目录，填入规则 `FREQ=WEEKLY;BYDAY=FR;BYHOUR=17;BYMINUTE=0`；
  3. 将任务状态由 `PAUSED` 切换为 `ACTIVE` 并点击「立即试运行」。
- **🔍 验证方法**：检查指定输出目录按时生成带时间戳的《周报汇总.md》与更新后的统计表。
- **🚀 扩展任务与导流**：增加每日待办提醒卡片自动写入 Obsidian 收件箱；🔗 进阶学习 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**。

---

## Case 04：全本地万字深度调研报告专家（`Local Deep Research`） {#case-04}

- **🎬 效果演示**：输入一个调研课题（如“2026 端侧小模型量化技术对比”），AI 自动拆解子问题、多轮检索网页与本地文献，撰写带完整引用角标的万字深度研报。
- **✅ 支持配置**：🔵 16G 甜点档及以上；推荐 `vLLM` 或 `llama-server`（`Qwen3.5-9B` 或 `Gemma 4 12B`，开启 Prefix Caching）。
- **💻 代码与步骤**：
  ```bash
  uv pip install local-deep-research
  export LDR_LLM__PROVIDER=openai_endpoint
  export LDR_LLM__OPENAI_ENDPOINT_URL=http://127.0.0.1:8000/v1
  export LDR_LLM__MODEL=Qwen3.5-9B
  ldr-web
  ```
- **🔍 验证方法**：浏览器打开本地 Web 界面输入课题，确认完成 ≥2 轮子问题检索且文末参考文献链接真实可点。
- **🚀 扩展任务与导流**：将你的本地论文文件夹接入检索源；🔗 进阶学习 **[`all-in-rag`](https://github.com/datawhalechina/all-in-rag)** 与 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)**。

---

## Case 05：多篇英文文献横评对比表 & 组会汇报 PPT 一键生成 {#case-05}

- **🎬 效果演示**：在 WorkBuddy 中同时 `@` 3～5 篇同领域英文 arXiv 论文 PDF，先自动输出《核心创新点/数据集/局限性横评 Markdown 表》，再无缝调用 `dashi-ppt-skill` 直出双语组会汇报 `.pptx`。
- **✅ 支持配置**：🔵 16G 甜点档（需开启 `32768` 上下文容纳多篇论文摘要与实验表）；WorkBuddy + `dashi-ppt-skill`。
- **💻 代码与步骤**：
  1. 将 3 篇同主题论文 PDF 放入 `papers_review/`；
  2. 在 WorkBuddy **Plan 模式** 下发送：*“读取 `papers_review/` 下全部 PDF，先输出包含【论文标题、核心方法、基准指标、局限性】的对比表存入 `横评.md`，再调用 dashi-ppt-skill 生成 10 页组会汇报 `组会分享.pptx`。”*
- **🔍 验证方法**：抽查横评表中的实验数字与原论文 Table 一致，且 `.pptx` 结构完整。
- **🚀 扩展任务与导流**：将生成的横评表自动归档进你的知识管理 MCP；🔗 进阶学习 **[WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)** 与 **[`all-in-rag`](https://github.com/datawhalechina/all-in-rag)**。

---

## Case 06：公式图表纹丝不乱 —— 双语学术论文精翻神器（`PDFMathTranslate` + `Zotero`） {#case-06}

- **🎬 效果演示**：调用本地免费算力将整本英文双栏学术 PDF 翻译成「左英右中对照版」与「纯中文版」，复杂的 LaTeX 数学公式、算法伪代码框与矢量图表位置 **100% 零损保留**！
- **✅ 支持配置**：🟢 8G（`llama-server`）/ 🔵 16G（强烈推荐用 `vLLM` 开启 `-t 8` 多线程并发批处理，整篇 15 页论文 1 分钟翻完！）。
- **💻 代码与步骤**：
  ```bash
  uv pip install pdf2zh
  export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
  export OPENAI_API_KEY=rocm-local
  pdf2zh paper.pdf -s openai:Qwen3.5-9B -t 4
  ```
- **🔍 验证方法**：打开生成的 `paper-dual.pdf` 与 `paper-mono.pdf`，检查行内/行间公式未乱码、双栏图片未错位。
- **🚀 扩展任务与导流**：配合 Zotero `PDF2zh` 插件实现文献库右键一键本地双语精翻；🔗 进阶学习 **[`self-llm`](https://github.com/datawhalechina/self-llm)**。

---

## Case 07：`WorkBuddy + Obsidian` 双链第二大脑 & 可共享「知识管理 MCP」 {#case-07}

- **🎬 效果演示**：不仅能对本地 Obsidian 双链笔记库进行带精确出处的开卷问答，还能通过 `FastMCP` 将知识库打包成标准 MCP 服务，在 `Qoder`、Codex 及队友电脑上跨端调用！
- **✅ 支持配置**：🟢 8G；三大引擎任一 + 仓库 [`src/main-assistant/knowledge_mcp/server.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/knowledge_mcp/server.py)。
- **💻 代码与步骤**：完整分步代码与挂载 JSON 见 **[第 3.3 节](../03-prompt-and-rag/workbuddy-obsidian-rag.md)** 与 **[第 4.2 节](../04-mcp-and-skills/index.md)**。
- **🔍 验证方法**：在 `Qoder` 中调用 `search_notes` 查到笔记，再调用 `create_note_card` 新增卡片并在 Obsidian 图谱中看到双链生效。
- **🚀 扩展任务与导流**：为 `server.py` 增加一个 `summarize_weekly_notes` 工具统计近 7 天新增卡片；🔗 进阶学习 **[`all-in-rag`](https://github.com/datawhalechina/all-in-rag)** 与 **[`hello-agents`](https://github.com/datawhalechina/hello-agents)**。

---

## Case 08：复习讲义秒变双人播客 & 微信/小红书网感分身（`Open Notebook` + `WeClone`） {#case-08}

- **🎬 效果演示**：作为 Google NotebookLM 的全隐私本地平替，把枯燥的期末复习讲义一键转为自然生动的一问一答双人对话播客 MP3 路上听；并用第 5 章方法微调出你专属语气的微信/小红书分身！
- **✅ 支持配置**：🟢 8G / 🔵 16G；本地 `/v1` 服务 + 开源 [`lfnovo/open-notebook`](https://github.com/lfnovo/open-notebook) + 第 5.6 节 `WeClone` 风格 LoRA。
- **💻 代码与步骤**：
  1. 使用 Docker 启动 `open-notebook` 并将 OpenAI Base URL 指向宿主机本地引擎端口；
  2. 上传课程 PDF 讲义，点击生成 Podcast 音频脚本并合成双人对话 MP3；
  3. 联动 [第 5.6 节](../05-fine-tune/chinese-lora-and-persona.md) 换上你微调后的个性化模型生成更具网感的播客台词！
- **🔍 验证方法**：播放导出的 MP3 音频，确认双角色轮流对谈自然且核心知识点准确无遗漏。
- **🚀 扩展任务与导流**：将你微调前后的台词做一次客服腔破功词对比测试；🔗 进阶学习 **[`happy-llm`](https://github.com/datawhalechina/happy-llm)**。
