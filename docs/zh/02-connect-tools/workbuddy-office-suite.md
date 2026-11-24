# 2.2 🌟【第一步重点推出】接入 WorkBuddy：AI PPT 生成、AI 自动填表与定时办公自动化

作为全书**第一步重点推出的王牌实战**，本节参考并联动 **[Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**，带你把第 1 章跑起来的 AMD 本地模型接入桌面办公智能体 **WorkBuddy**，一口气打通打工人和学生党每天最刚需的 **办公自动化三件套**！

---

## 一、保姆级接入配置与 5 大隐形避坑清单

WorkBuddy 支持通过界面添加或直接放入 `models.json` 配置文件来接入本地模型。最稳妥的“零出错抄作业法”是直接使用本仓库已验证通过的配置文件 **[`src/configs/workbuddy-models.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/workbuddy-models.json)**：

### Step 1：一键放入已验证 `models.json`

将本仓库 `src/configs/workbuddy-models.json` 复制到你的用户配置目录（`~/.workbuddy-ai/models.json`，部分版本共用 `~/.codebuddy/models.json`）：

```json
{
  "models": [
    {
      "id": "rocm-llama-qwen3.5-4b",
      "name": "AMD 本地引擎 1：llama-server (Qwen3.5-4B GGUF)",
      "vendor": "custom",
      "apiKey": "rocm-local",
      "url": "http://127.0.0.1:8080/v1/chat/completions",
      "model": "qwen3.5-4b-q4_k_m",
      "maxInputTokens": 28672,
      "maxOutputTokens": 4096,
      "maxContextTokens": 32768,
      "temperature": 0.3,
      "supportsToolCall": true,
      "supportsImages": false
    }
  ]
}
```

### Step 2：必读！新手 90% 卡壳的「5 大隐形避坑表」

| 序号 | 常见翻车现象 | 真实根因 | 一招解决 |
|:---:|:---|:---|:---|
| **坑 1** | 报 `404 Not Found` | UI 界面填法与 `models.json` 填法混用 | **UI 界面**填到 `/v1` 即可；而 **`models.json` 里的 `"url"` 必须写全 `/v1/chat/completions`**！ |
| **坑 2** | 模型显示“思考完”但**回复一片空白** | WorkBuddy 内置注入了数万字工具与技能说明，撑爆了引擎默认的 4096 上下文 | 本地启动引擎时**必须显式指定 `32768` 上下文**（`--ctx-size 32768` 或 `--max-model-len 32768`）！ |
| **坑 3** | 保存配置报错或拒绝发请求 | 本地服务虽不鉴权，但客户端强制校验非空密钥 | `"apiKey"` 随便填非空字符（如 `"rocm-local"` 或 `"lemonade"`），**绝不能留空 `""`**。 |
| **坑 4** | AI 只会在对话框嘴上说说，不会真建文件 | 未开启工具调用支持 | `models.json` 中确保 `"supportsToolCall": true`，且服务端带 `--jinja`（`llama-server`）或 `--enable-auto-tool-choice`（`vLLM`）。 |
| **坑 5** | Windows 路径报错找不到文件 | 工作区含特殊符号或过深中文嵌套 | 使用独立沙盒英文路径，如 `D:\AI_Workspace\office_demo`。 |

---

## 二、实战一：AI PPT 一键生成（唯一锁定 `dashi-ppt-skill` 大师 PPT Skill）

市面上很多网页生成 PPT 工具只能导出死板的图片，没法在 PowerPoint 或 WPS 里改字调排版。本课程经过全网横向实测，**唯一锁定开源王牌项目 [`dashi-ppt-skill`（大师 PPT Skill）](https://github.com/chuspeeism/dashi-ppt-skill)**！

### 为什么选它？
- **12 套专业视觉主题 + 1,020 个商业与学术版式**（原生涵盖 SWOT 矩阵、漏斗图、时间轴、对比卡片等）；
- **自带浏览器交互微调工作台**：生成后可直接在网页里点选任意文本框局部微调；
- **真正可二次编辑的 `.pptx` 导出**：一键导出原生 `.pptx` 文件（同时支持 PDF 与离线单文件 HTML），在 WPS / Microsoft PowerPoint 中每个文字框与图形均可自由编辑！

### 三步实操出片

1. **一行命令安装 `dashi-ppt-skill`**：
   在你的沙盒工作目录终端运行安装命令，选择安装到当前工作区或全局技能目录（WorkBuddy 支持自动识别 `.codebuddy/skills/` 或 `.claude/skills/`）：
   ```bash
   npx dashi-ppt-skill@latest
   ```
2. **准备一份原始材料**：在沙盒目录 `office_demo/` 下放一份你的《季度项目复盘.md》或《组会论文笔记.md》。
3. **在 WorkBuddy（Craft 模式）输入一键生成指令**：
   ```text
   请调用 dashi-ppt-skill 技能，读取当前工作区的《季度项目复盘.md》，为我生成一份 10 页的技术汇报 PPT：
   1. 选用科技商务深色/浅色高对比主题，包含封面、目录、核心指标卡片页、架构流程图页、问题复盘对比表与下阶段计划时间轴；
   2. 每页提炼不超过 4 个精炼要点，拒绝大段文字堆砌；
   3. 打开本地预览工作台，并同步导出可编辑的 .pptx 文件到 output/季度汇报.pptx。
   ```
4. **验收效果**：用 PowerPoint 或 WPS 打开 `output/季度汇报.pptx`，双击任意标题与图表文字，确认可自由二次编辑！

---

## 三、实战二：AI 自动填表（非结构化单据/简历 ➡️ 标准 Excel 报表）

参考 [Datawhale WorkBuddy 官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) 的经典办公场景：当你手头收到十几份格式五花八门的报销明细、活动报名表或候选人简历时，再也不用人工逐个复制粘贴！

### 分步实操

1. **构造测试沙盒文件**：在 `office_demo/invoices/` 下新建 3 个模拟报销文本或单据说明（例如：`01_差旅单.txt`、`02_设备采购.txt`、`03_缺发票号单据.txt`）。
2. **发送「强字段契约 + 缺失防幻觉」填表指令**：
   ```text
   请扫描当前工作区 invoices/ 文件夹下的所有单据文件，提取关键信息并汇总写入 output/报销汇总表.xlsx：
   1. 表格必须严格包含以下 6 列：【序号、文件名、报销人、费用类别、金额(元)、发票号码、状态备注】；
   2. 若原文件中某项信息缺失（如未写发票号码），对应单元格必须填入“待确认”，严禁编造数字；
   3. 先在对话框列出提取结果预览表（dry-run），确认无误后生成 Excel 文件并在最后一行计算已确认金额总计。
   ```
3. **验收效果**：打开生成的 `output/报销汇总表.xlsx`，核对正常单据金额无误，且缺失发票号的那一单准确标记为 `待确认`！

---

## 四、实战三：用 AI 定时任务自动处理日常办公事项（到点自动干活）

除了手动发指令，WorkBuddy 还支持**定时任务（Scheduled Tasks）** —— 只要你的 AMD 电脑和本地模型服务开着，它就能像全天候秘书一样每天早上自动统计昨日数据、每周五下午自动汇总周报！

### 遵循「三步黄金避坑流程」配置定时任务

> 💡 最佳实践源自 [Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)：**先在普通对话跑通 Prompt ➡️ 绑定独立沙盒工作目录 ➡️ 配置定时周期（RRULE）并激活**。

1. **第 1 步：先在对话框手动跑通一次**：把你要自动执行的提示词（例如：“读取 `daily_logs/` 下今日新增记录，追加填入 `汇总表.xlsx` 并生成 `今日简报.md`”）在普通会话跑通，确保本地模型能一次成功。
2. **第 2 步：新建定时任务并绑定沙盒目录**：
   - 进入 WorkBuddy 左侧 **「定时任务 / Automation」** 面板 → 点击新建任务；
   - **工作目录（Workspace）**：显式指定为你的沙盒目录 `office_demo/`（切勿留空）。
3. **第 3 步：设置触发时间规则（支持界面选择或标准 `RRULE` 语法）**：
   - **每日早 9:00 自动汇总昨日数据**：`FREQ=DAILY;BYHOUR=9;BYMINUTE=0`
   - **每周五下午 17:00 自动生成本周周报与下周待办**：`FREQ=WEEKLY;BYDAY=FR;BYHOUR=17;BYMINUTE=0`
   - 保存后将任务状态从 `PAUSED` 切换为 **`ACTIVE`**，点击一次「立即试运行（Run Now）」验证产出文件正常落盘！

---

## 🔗 本节进阶导流

恭喜你用自己的 AMD 电脑零成本跑通了「AI PPT + AI 自动填表 + 定时办公自动化」三件套！想进一步解锁 WorkBuddy 的多智能体协作、飞书/企微企信联动与更多高频办公模板，推荐前往 **[Datawhale WorkBuddy 官方完整教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)** 继续深造！

👉 接下来，我们顺手把本地模型也接进你的编程环境：**[2.3 & 2.4 接入 `Qoder` 与 `Codex CLI`](./qoder-and-codex.md)**。
