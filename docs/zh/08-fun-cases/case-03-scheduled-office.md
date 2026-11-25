# Case 03：用 WorkBuddy AI 定时任务自动处理日常办公事项 —— 每天早 9 点的零成本数字秘书

> 🏷️ **场景分类**：任务规划（首发置顶办公王牌 #3）  
> ⏱️ **预计耗时**：20 分钟 ｜ 🎮 **最低硬件档位**：🟢 入门档 8G（锐龙 AI 笔记本强烈推荐开启 `lemonade-server` NPU 超低功耗后台常驻！）

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：无需每天手动打开对话框敲字！绑定你的日常收件箱沙盒目录并设定 `RRULE` 日历定时触发规则（如每个工作日 `09:00` 或每周五 `17:00`）；
  - **AI 自动执行**：到点自动唤醒本地 AMD 模型与 [第 4.3 节 `weekly-office-check` 技能](../04-mcp-and-skills/03-custom-office-skill.md)，自动比对已处理台账、增量提取新增文件更新 Excel，并生成今日重点跟进晨报或周五三段式周报；
  - **最终交付物**：每天推开办公室门/打开电脑就已自动躺好的《今日待办晨报_YYYY-MM-DD.md》与增量更新报表，**365 天 0 元 API 费**！

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Windows 11 / Ubuntu 24.04（推荐不休眠办公台式机，或搭载 Ryzen AI XDNA 2 NPU 的轻薄本） |
| **底层推理服务引擎** | 锐龙本首推 **`lemonade-server`（`:13305`）** 低功耗常驻；桌面独显首推 **`llama-server`（`:8080`）** |
| **推荐主体模型** | 🟢 `Gemma 4 E4B`（响应快、功耗极低）或 🔵 `Qwen3.5-9B` |
| **核心挂载组件** | WorkBuddy「定时任务（Scheduled Tasks）」模块 + 仓库 [`weekly-office-check/SKILL.md`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/skills/weekly-office-check/SKILL.md) |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

### Step 1：挂载办公核对技能包并初始化幂等日志目录

```bash
CRON_BOX=~/ai_workspace/sandbox/case03_cron
mkdir -p "$CRON_BOX/daily_inbox" "$CRON_BOX/reports" "$CRON_BOX/.codebuddy/skills"

# 装入本仓库标准周报与表格核对技能
cp -r ~/ai_workspace/hello-rocm/src/main-assistant/skills/weekly-office-check "$CRON_BOX/.codebuddy/skills/"

# 放入一条模拟昨晚收到的待办备忘
echo "【项目进度】已完成知识管理 MCP 单元测试编写；【卡点】待确认周五演示教室投影仪转接头型号。" > "$CRON_BOX/daily_inbox/周四晚工作记录.md"
```

### Step 2：在 WorkBuddy 新建 `RRULE` 定时任务并写入幂等指令

1. 打开 WorkBuddy 左侧 ⏰ **「定时任务（Scheduled Tasks）」**，点击 **「新建任务」**；
2. **绑定工作区**：`~/ai_workspace/sandbox/case03_cron`；
3. **设置触发周期（任选其一）**：
   - 每个工作日早 9 点晨报：`FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0`
   - 每周五下午 5 点周报归档：`FREQ=WEEKLY;BYDAY=FR;BYHOUR=17;BYMINUTE=0`
4. **填入带“增量去重锁”的执行提示词**：

```text
请自动激活 weekly-office-check 技能执行定时办公巡检：
1. 读取 reports/processed_history.txt（若不存在则创建）；
2. 扫描 daily_inbox/ 中尚未登记在 processed_history.txt 的新增记录文件；
3. 若存在新增文件：按技能要求的三段式结构（✅ 本周核心交付 / ⚠️ 风险与卡点 / 📅 下周计划）汇总写入 `reports/自动周报汇总.md`，并将已处理文件名追加进 `processed_history.txt`；若无新增文件，仅追加一行巡检正常时间戳日志。
4. 严禁删除 daily_inbox/ 原始文件。
```

### Step 3：点击「立即试运行」两次验证防重复锁，切换为 `ACTIVE`！

先连续点击两次 **「立即运行（Run Now）」** 验收幂等性，确认无误后将任务开关由 `PAUSED` 拨至 **`ACTIVE`**！

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **首次运行测试**：`reports/自动周报汇总.md` 成功生成，且严格包含 `✅ 本周核心交付`、`⚠️ 风险与卡点`（准确抓出投影仪转接头卡点）与 `📅 下周计划` 三段结构。
- [x] **二次运行防重测试**：在不放新文件的情况下再点一次试运行，确认 `processed_history.txt` 生效，未重复堆叠相同内容。
- [x] **定时自动触发测试**：可将触发时间临时设为 **3 分钟后**，锁屏喝口水回来检查文件修改时间戳是否准点刷新！

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（15 分钟）**：联动你在第 4 章挂载的 `my-knowledge-vault` MCP，让每周五定时生成的周报自动调用 `create_note_card` 同步沉淀进你的 Obsidian 仓库。
- **⭐⭐ 白银改造（30 分钟）**：如果你使用的是 Ryzen AI 300 / Max+ 395 笔记本，按 [第 6.4 节](../06-optimize-speed/04-agent-prefix-caching.md) 将该定时任务绑定至 `lemonade-server` 的 NPU 模型，对比 GPU 与 NPU 后台挂机时的风扇转速与电量消耗。
- **⭐⭐⭐ 黄金改造（进阶深造）**：设计一套“每天早 8:30 自动读取本地 RSS 订阅摘要生成晨读卡片”的晨间学习流。
  - 🔗 **进阶学习入口**：**[Datawhale WorkBuddy 从入门到实战官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：到了设定时间为什么定时任务没有跑？**
  - **原因**：① 任务创建后仍停留在默认的 `PAUSED` 草稿状态未切为 `ACTIVE`；② 电脑进入了系统深度睡眠（Sleep/Hibernate）断开了进程调度。
  - **一行解决**：确认任务卡片绿灯为 `ACTIVE`，并在系统电源设置中将接通电源时的自动睡眠设为“从不（仅关闭显示器）”。

---

👉 下一篇案例：**[Case 04｜全本地万字深度调研报告专家（`Local Deep Research`）](./case-04-local-deep-research.md)**
