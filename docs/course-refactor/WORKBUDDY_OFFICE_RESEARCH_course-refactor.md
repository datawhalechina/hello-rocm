# 全网「WorkBuddy 接入本地 AI 与爆款办公实战案例」调研与前置设计报告

> **关联任务**：`course-refactor`（hello-rocm 2.0 第 2 章「工具接入」与第 7 章「办公案例最前置」专项设计）  
> **核心策略调整**：将 **零代码、人人每天刚需的 WorkBuddy AI 办公实战（理文件、洗 Excel、写周报、读论文出 PPT）放在最前面**（既是第 2 章第一个上手工具，也排在第 7 章 15 个趣味案例的 **Case 01–04 头牌位置**），彻底消除非编程背景小白的距离感！  
> **调研来源**：WorkBuddy 官方文档、腾讯云开发者社区、知乎/掘金/CSDN/SegmentFault/少数派高赞实战帖与真实避坑总结。

---

## 一、为什么要把「WorkBuddy 办公案例」放到最前面？

1. **零代码门槛，受众面扩大 10 倍**：无论理工科还是文商科学生、职场新人，不一定每天写 Python 代码，但 **100% 每天都要面对杂乱文件夹、Excel 表格、周报汇报和 PDF 文献**。
2. **从“只会陪聊”一步跨入“直接交付文件”**：传统网页套壳 UI 只能吐出一段文字让你自己复制粘贴排版；而 WorkBuddy 接上本地 AMD 模型后，能直接在本地文件夹里**替你建好分类目录、洗好 `.xlsx` 报表、生成 `.docx` 分析报告和 20 页 `.pptx` 演示文稿**——获得感瞬间拉满！
3. **自然暴露本地小模型真实边界，完美引出后文**：在实操 WorkBuddy 的 `Ask / Plan / Craft` 三模式时，学习者会亲眼看到上下文窗口与工具调用对本地模型的考验，顺理成章引出第 3 章（Skills/MCP）、第 4 章（中文工具调用微调）与第 5 章（Prefix Caching 首字加速）。

---

## 二、全网实测汇总：WorkBuddy 接入本地 AMD 模型保姆级规范与 5 大隐形坑

### 2.1 两种接入方式与官方验证 `models.json` 标准模板

WorkBuddy 支持通过 **OpenAI 兼容接口** 连接本机 `llama.cpp (llama-server)`、`vLLM`、`Ollama` 或 `LM Studio`。除了界面填写外，课程将直接提供防丢失的用户级配置文件模板：

- **配置文件路径**：
  - Windows：`%USERPROFILE%\.workbuddy-ai\models.json`（或同源 `%USERPROFILE%\.codebuddy\models.json`）
  - Linux / macOS：`~/.workbuddy-ai/models.json`（或 `~/.codebuddy/models.json`）

```json
{
  "models": [
    {
      "id": "gemma-4-e4b-it",
      "name": "AMD 本地轻量助手 (llama.cpp 8G入门档)",
      "vendor": "Custom",
      "apiKey": "rocm-local",
      "url": "http://127.0.0.1:8080/v1/chat/completions",
      "maxInputTokens": 32768,
      "maxOutputTokens": 4096,
      "temperature": 0.4,
      "supportsToolCall": true,
      "supportsImages": true
    },
    {
      "id": "Qwen/Qwen3-8B",
      "name": "AMD 本地办公主力 (vLLM 16G甜点档)",
      "vendor": "Custom",
      "apiKey": "rocm-vllm",
      "url": "http://127.0.0.1:8000/v1/chat/completions",
      "maxInputTokens": 32768,
      "maxOutputTokens": 8192,
      "temperature": 0.3,
      "supportsToolCall": true,
      "supportsImages": false
    }
  ]
}
```

### 2.2 全网踩出来的「5 大必避隐形巨坑」（写进第 2.2 节与 FAQ，帮小白省下 3 小时排错！）

| 坑点编号 | 真实翻车现象 | 根本原因剖析 | 课程给出的标准解法 |
|:---:|:---|:---|:---|
| **坑 1** | 界面测试连得上，手改 `models.json` 却报 `404`（反之亦然） | **UI 界面与 JSON 文件的 URL 规则不同**！UI 里填 Base URL 只需写到 `/v1`；而 `models.json` 里的 `"url"` 必须写全 `/v1/chat/completions` | 文档并列配图高亮对比两种写法，并提供一键生成配置脚本 |
| **坑 2** | 本地明明免密，点击保存却一直报错不让存 | WorkBuddy 前端校验 `API Key` 为必填非空字段 | 随便填入 `"rocm-local"` 等非空字符即可通过 |
| **坑 3** | 显示“任务已完成”，但对话框**回复一片空白**或死循环乱调工具 | WorkBuddy 后台自动注入**数万 Token 的内置工具与技能说明书**！若本地引擎默认上下文仅 `4096`，提示词被腰斩截断 | 启动本地引擎时强制加 `--ctx-size 32768`（`llama-server`）或 `--max-model-len 32768`（`vLLM`），并加 `--jinja` / `--enable-auto-tool-choice` |
| **坑 4** | 8G 小显存跑极小模型（如 1.5B）在 `Craft` 模式下反复工具报错 | 极小参数未对齐模型难以驾驭复杂多步文件修改 Schema | 8G 小模型初期在 `Ask` 模式用，或将 `"supportsToolCall": false` 作为纯阅读润色；想让小模型稳调工具 → 去第 4 章做微调！ |
| **坑 5** | 软件装在 C 盘撑爆空间，或授权整个 D 盘导致重要原文件被改乱 | Agent 具有真实读写磁盘权限与缓存索引开销 | **三大安全铁律**：① 软件与缓存装数据盘；② **专案专用独立英文/下划线沙盒目录**（如 `D:\WB_Sandbox\demo01`）；③ 批量操作提示词必带 **“不覆盖原文件，先输出 dry-run 预览清单”** |

---

## 三、放在最前面的「四大爆款 AI 办公实战场景」（第 2 章首秀 + 第 7 章 Case 01–04）

结合知乎、掘金、少数派与官方最受欢迎的工作流，提炼出 **Ask（只读盘点）→ Plan（大纲确认）→ Craft（自动交付文件）** 黄金三步法，打造排在全课最前面的四大王牌办公案例：

### 办公头牌 1（Case 01）：🗂️ 拯救杂乱下载夹 —— 智能分类归档与《文件清单盘点》生成
- **玩法**：把塞满 PDF 发票、课件讲义、会议截图与合同的测试沙盒文件夹拖进 WorkBuddy，用 `Plan → Craft` 模式自动创建「财务票据 / 课程资料 / 会议记录 / 项目合同」子目录归类重命名（先 `dry-run` 预览确认），并在根目录自动生成一份总结每个文件核心内容的 `《文件清单盘点.md》`！（进阶可联动 `LlamaFS` 开启后台实时守护）。

### 办公头牌 2（Case 02）：📊 办公三件套全自动流水线 —— 脏乱 Excel 清洗 ➡️ Word 报告 ➡️ 20 页汇报 PPTX
- **玩法（全网最震撼办公连招！）**：
  1. **洗 Excel**：扔进含重复行、空值、日期金额格式混乱的《销售/实验明细表.xlsx》，一句话自动去重、按维度汇总并在原表旁生成美化好的《汇总看板.xlsx》；
  2. **出 Word**：基于刚算好的汇总表，无缝追问生成结构严谨的《阶段分析报告.docx》；
  3. **做 PPT**：“去 AI 味”三步法（`Ask` 盘点亮点 → `Plan` 确认 15–20 页大纲 → `Craft` 交付克制专业的结构化卡片 `.pptx` 演示文稿）！

### 办公头牌 3（Case 03）：📈 告别周报焦虑与资料大海捞针 —— 一键高级感周报 + 全本地深度调研报告
- **玩法**：将零散聊天记录、工作流水账（或 `git log`）拖入工作区，套用社区高赞「核心成果 + 量化指标 + 下周计划 + 风险协同」模板 30 秒生成周报（并顺手封装成 `/weekly-report` 专属 Skill！）；复杂课题则联动 `Local Deep Research` 自动多轮检索生成万字带引用调研报告。

### 办公头牌 4（Case 04）：📄 学术与研报党福音 —— 多篇 PDF 文献横评对比表 & 组会双语汇报 PPT
- **玩法**：授权 3–5 篇同方向 PDF 论文或行业研报，先用 `Ask` 模式自动拉出「研究痛点 / 核心方法 / 实验数据集 / 局限性」横向对比 Markdown 表，再用 `Craft` 模式一键生成可直接上台讲的组会汇报 PPT（配合 Case 05 `PDFMathTranslate` 实现公式零损全文精翻）！

---

## 四、章节与案例顺序重排对照

1. **第 2 章内部顺序优化（先办公、后编程，人人零门槛起步）**：
   - `2.1` 读懂万能插座（OpenAI 兼容接口）与本地 Agent 沙盒安全铁律
   - `2.2` **【第一顺位 · 零代码办公首秀】接入 WorkBuddy：配置 `models.json`、避开 5 大隐形坑，10 分钟跑通本地文件归档与报表生成**
   - `2.3` 接入 VS Code 编程助手（`Continue` + `Cline` 已验证配置）
   - `2.4` 接入 Codex CLI 与终端智能体（含 `toy-cli`）
2. **第 7 章「黄金 15 案例」重排（Case 01–08 全部为日常办公、论文、知识库与会议纪要，编程与游戏置于 Case 09–15）**：让任何专业背景的小白翻开案例集第一页就能立刻用起来！


---

## 五、专项优选：用 WorkBuddy 生成 PPT 最好的开源项目（第一步重点推出方案）

针对「第一步重点推出的破圈首发内容」，横向对比全网 GitHub 上的 WorkBuddy / CodeBuddy PPT 技能与引擎后，确定以下 **「双冠军首发组合 + 两大备选」**：

1. **🥇 首发主推 No.1：`dashi-ppt-skill`（大师 PPT Skill，`github.com/chuspeeism/dashi-ppt-skill`）**
   - **入选理由**：彻底解决普通 AI 生成 PPT“只能看不能在 WPS/PowerPoint 里改字”的痛点！内置 **12 套视觉主题 + 1,020 个专业商业/学术版式**（SWOT、漏斗图、雷达图、画布等），自带**浏览器内可视化局部编辑面板**，并支持一键导出 **真实可二次编辑的 `.pptx` 文件**、离线 HTML 与 PDF。
   - **安装命令**：`npx dashi-ppt-skill@latest`（或挂载至 `~/.codebuddy/skills/` / `~/.workbuddy-ai/` 技能目录）。
2. **🥈 首发搭档 No.2（8G 入门卡零依赖保底王）：`frontend-presentation-slides`（`github.com/chouraycn/frontend-presentation-slides`）**
   - **入选理由**：专为 **WorkBuddy / CodeBuddy** 原生打造！`git clone https://github.com/chouraycn/frontend-presentation-slides ~/.codebuddy/skills/frontend-presentation-slides` 10 秒装好，**零外部 npm/Python 依赖**（纯内联单文件 HTML 动效幻灯片），内置 8 套高级手工设计模板（支持与 `.pptx` 双向互转），在 8G 显存本地小模型上 **100% 零环境报错出大片**！（同类高颜值杂志风可搭配 `op7418/guizang-ppt-skill` 归藏 PPT Skill）。
3. **🥉 独立软件与评测拓展备选**：
   - **`presenton/presenton`**：Gamma 开源本地桌面/Web 平替，直接连本地 OpenAI API 生成 `.pptx`；
   - **`icip-cas/PPTAgent`**（中科院开源）：支持参考模板版式学习与 `PPTEval` 三维自动打分评测；
   - **`jinwyp/open-ppt-skill`** & **`GongRzhe/Office-PowerPoint-MCP-Server`**：PPTD 中间层与原生 PowerPoint MCP 操控方案。
