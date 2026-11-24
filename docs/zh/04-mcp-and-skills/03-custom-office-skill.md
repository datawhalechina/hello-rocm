# 4.3 模块化技能包 Agent Skills（`SKILL.md`）：省显存原理与手写办公技能实战

> **本节目标**：彻底搞懂 **MCP 与 `SKILL.md` 的黄金分工**，从第 1 章“显存草稿本（KV Cache）”视角算清为什么 `SKILL.md` 的 **渐进式披露（Progressive Disclosure）** 是本地 8G/16G 显卡的救星，并亲手编写加载本仓库的 **`weekly-office-check` 办公技能包**，与你的「知识管理 MCP」双剑合璧！

---

## 一、灵魂一问：既然有了 MCP 工具，为什么还要发明 `SKILL.md`？

很多刚接触 Agent 的同学容易把 **MCP** 和 **Agent Skills（`SKILL.md`）** 搞混。其实用一个最形象的职场比喻，你一秒钟就能分得清清楚楚：

- **🔌 MCP（Model Context Protocol）= 给员工配的「硬件工具箱与系统权限」**  
  比如给它配一把螺丝刀（读写文件）、一台内部数据库查询终端（`search_notes`）、一个地图导航仪（高德 MCP）。**MCP 决定了 AI “能不能碰到外部世界的数据”。**
- **📘 Agent Skills（`SKILL.md`）= 发给员工的「岗位标准作业指导书（SOP 手册）」**  
  光有工具箱还不够——新员工不知道你们团队的周报必须按“交付/风险/下周计划”三段式写，不知道做 PPT 要调哪个版式，也不知道遇到缺失发票号要填 `待确认`。**`SKILL.md` 决定了 AI “按什么专业流程和规矩把活干漂亮”。**

---

## 二、算一笔显存账：为什么不把所有规矩直接写进 System Prompt？

你可能会问：*“那我把写周报的规矩、审代码的规矩、排错的规矩、写小红书文案的规矩全都粘贴进客户端的全局系统提示词（System Prompt）里不就行了吗？”*

在云端你或许只是多花几毛钱 Token 费；但在本地消费级显卡上，这会直接带来灾难——让我们算一笔实打实的显存与速度账：

```mermaid
flowchart TB
    subgraph BAD["❌ 传统做法：10 套规矩硬塞进全局 System Prompt"]
        B1["你只问一句：今天星期几？"] --> B2["客户端被迫把 15,000 字无关规矩<br/>全部塞进当前上下文发给显卡！"]
        B2 --> B3["💥 后果：白占 ~1.2GB KV Cache 显存<br/>首字等待暴增 4 秒，还容易把小模型看串台！"]
    end

    subgraph GOOD["✅ 现代 SKILL.md：三层渐进式按需披露 (Progressive Disclosure)"]
        G1["平时待机：仅加载每个 SKILL.md 顶部的<br/>name + description 两行目录 (~40 Tokens/个！)"] --> G2["当你提到『帮我做本周周报核对』时<br/>智能体按目录精准翻开 weekly-office-check/SKILL.md"]
        G2 --> G3["🚀 效果：平时零显存负担，用时百发百中<br/>还可按需继续读取 references/ 子手册！"]
    end
```

看懂这个精妙设计了吗？
1. **第 1 层（目录索引层，常驻仅几十 Token）**：启动对话时，智能体只看每个技能文件夹里 `SKILL.md` 最顶部的 YAML 头（`name` 和 `description`）；
2. **第 2 层（主体规程层，触发时才加载）**：只有当你的意图匹配 `description` 时，客户端才把那几百字的 `SKILL.md` 正文读进上下文；
3. **第 3 层（深度附件层，按需下钻）**：如果技能还附带了长篇参考表（就像我们在第 0.4 节装的 [`src/hello-rocm-skill/references/`](https://github.com/datawhalechina/hello-rocm/tree/dev_1.0/src/hello-rocm-skill/references) 那样），只有遇到特定细分问题时才会进一步打开子文件！

这就是为什么在本地 `Gemma 4 E4B` 或 `Qwen3.5-9B` 上，你可以放心安装 **20 个专业 `SKILL.md`** 而丝毫不拖慢日常聊天速度！

---

## 三、动手解剖并加载你的第一个办公技能：`weekly-office-check`

打开本仓库为你准备的极简标准范例：**[`src/main-assistant/skills/weekly-office-check/SKILL.md`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/skills/weekly-office-check/SKILL.md)**。整个文件只有精炼的 16 行，却五脏俱全：

```markdown
---
name: weekly-office-check
description: 日常办公表格核对与周报排版自检技能。当用户要求汇总本周工作、检查报销/统计表格缺失项或生成结构化周报时自动激活。
---

# 日常办公核对与周报生成技能规范（省显存精简版）

请严格按以下四步执行，禁止自由编造未出现在本地材料中的数据：

1. **安全预检（Dry-Run 优先）**：先列出拟读取的沙盒目录文件清单，若涉及覆盖修改原表，先输出预览摘要请用户确认。
2. **缺失项显式标注**：提取报销单、简历或原始记录时，遇到原文缺失的字段一律填入 `待确认`，绝不猜测补全。
3. **三段式周报结构**：
   - `✅ 本周核心交付`（每条必须包含具体量化结果或产出文件名）
   - `⚠️ 风险与卡点`（标明原因与需要的支持）
   - `📅 下周计划`（不超过 3 项可验收动作）
4. **双链归档兼容**：若输出至 Obsidian 笔记库，自动补充 YAML `tags: [周报, 办公归档]` 并保留关联项目 `[[双向链接]]`。
```

### ✍️ 编写高触发率 `SKILL.md` 的两个黄金心法
1. **`description` 一定要写清“什么时机触发（When to use）”**：大模型决定翻不翻开这本手册，**100% 只看 `description` 这一句话**！一定要包含用户常说的触发关键词（如*“当用户要求汇总本周工作、检查表格缺失项或生成周报时自动激活”*）；
2. **正文用带编号的强祈使句**：不写长篇背景故事，直接用 `1. 2. 3.` 给出可检查的硬标准，小模型执行起来稳如钟表！

---

## 四、高潮联调：让「知识管理 MCP + 自定义办公 Skill + 大师 PPT」三剑合璧！

先把 `weekly-office-check` 技能一键装入你的工作区与全局技能目录：

```bash
mkdir -p ~/ai_workspace/sandbox/.codebuddy/skills ~/.agents/skills
cp -r ~/ai_workspace/hello-rocm/src/main-assistant/skills/weekly-office-check ~/ai_workspace/sandbox/.codebuddy/skills/
cp -r ~/ai_workspace/hello-rocm/src/main-assistant/skills/weekly-office-check ~/.agents/skills/
```

现在打开已挂载好 `my-knowledge-vault` MCP 的 **WorkBuddy（Craft 模式）** 或 **`Qoder`**，发出一句只用日常说话就能触发全套自动化链路的指令：

```text
请帮我汇总本周的学习与办公进展并生成结构化周报：
1. 调用我的知识管理 MCP，检索我本周在 Obsidian 里记录的所有与 `rocm`、`workbuddy`、`复盘` 相关的笔记卡片，同时检查当前沙箱 output/ 下的报销汇总表是否有“待确认”跟进项；
2. 严格按照周报核对技能规范生成本周周报，并调用 create_note_card 归档回我的 Obsidian `00-收件箱/本周AI工作周报`！
```

*(🚀 进阶连招：生成完周报卡片后，你甚至可以顺手接上一句第 2.2.2 节学的口令：“再调用 `dashi-ppt-skill` 把这篇周报转成 5 页周五组会汇报 PPT！” —— **查私人笔记 MCP → 守自定义规矩写周报 Skill → 出原生可编辑 PPT Skill** 一气呵成！)*

---

## 👉 下一节：系好智能体的“安全带”，完成 `个人学习助手 v2.0` 正式验收！

当你的本地 AI 助手既能查笔记库、又能写文件、还能跑定时任务时，如何从工程上彻底杜绝它“死循环卡死”或“越权乱改文件”？同时，在日常重度使用中，本地小模型暴露出的哪两个短板值得我们在第 5、6 章继续升级？

马上进入本章收官节：**[4.4 本地 Agent 安全护栏、`个人学习助手 v2.0` 验收与进阶导流（`hello-agents`）](./04-safety-guardrails.md)**！
