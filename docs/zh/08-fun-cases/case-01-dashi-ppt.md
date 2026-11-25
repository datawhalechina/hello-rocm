# Case 01：一句话文档变可编辑汇报 PPT —— WorkBuddy + `dashi-ppt-skill` 大师级制稿流

> 🏷️ **场景分类**：报告生成（首发置顶办公王牌 #1）  
> ⏱️ **预计耗时**：20 分钟 ｜ 🎮 **最低硬件档位**：🟢 入门档 8G 独显 / 锐龙 AI 核显本（甜点档 16G 更快更稳）

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：一份本地杂乱的周报总结、课程读书报告或实验记录 Markdown 文档（数据 100% 不出本机）；
  - **AI 自动执行**：WorkBuddy 驱动本地 AMD 模型调用全书唯一指定的演示文稿技能 **[`chuspeeism/dashi-ppt-skill`（大师 PPT Skill）](https://github.com/chuspeeism/dashi-ppt-skill)**，从 **15 套主题 × 68 种语义组件（共 1,020 种专业版式）** 中自动编排封面、痛点三栏卡、KPI 大数字页与路线图；
  - **最终交付物**：浏览器本地可视化微调预览台 + 一键导出的 **WPS / PowerPoint 100% 原生矢量可编辑 `.pptx` 文件**（绝非死图片！）。

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Windows 11 / Ubuntu 24.04（🟢 8G 独显核显 / 🔵 16G 独显 / 🟣 Max+ 395 / ☁️ Radeon Cloud） |
| **底层推理服务引擎** | `llama-server`（`:8080`）/ `lemonade-server`（`:13305`）/ `vLLM`（`:8000`） |
| **推荐主体模型** | 🔵 `Qwen3.5-9B` 或 `Gemma 4 12B`（中文提炼与排版结构最稳）；🟢 `Gemma 4 E4B`；🟣 `Qwen3.6-35B-A3B` |
| **必带服务端启动参数** | `--ctx-size 32768 --jinja`（`llama-server`）或 `--max-model-len 32768 --enable-auto-tool-choice`（`vLLM`） |
| **客户端与配套配置** | WorkBuddy（Craft 模式）+ [`src/configs/workbuddy-models.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/workbuddy-models.json) + Node.js 18+ |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

### Step 1：确认本地 `/v1` 模型服务在线且开启 `32K` 上下文

```bash
# 以 llama-server :8080 为例验证服务在线（13305 或 8000 同理）
curl -s http://127.0.0.1:8080/v1/models
```

### Step 2：在沙箱环境一键安装 `dashi-ppt-skill` 并准备汇报素材

```bash
mkdir -p ~/ai_workspace/sandbox/case01_ppt/output && cd ~/ai_workspace/sandbox/case01_ppt

# 安装大师 PPT 技能包（按交互提示装入全局 ~/.agents/skills 或当前工作区）
npx dashi-ppt-skill@latest

# 生成一份标准测试提纲（亦可直接放入你自己的真实周报或答辩草稿！）
cat << 'EOF' > defense_notes.md
# 毕业设计中期答辩：基于 AMD 端侧算力的实验室安全巡检智能体
## 1. 课题背景与痛点
- 传统人工每日巡检 12 间化学实验室耗时 2 小时，纸质台账易漏填；
- 实验室监控截图与试剂台账属于内部数据，严禁调用外部公有云 API。
## 2. 已完成核心进展
- 部署本地 Gemma 4 E4B 多模态服务，显存占用仅 4.6 GB，响应速度达 68 tok/s；
- 完成 240 张危化品柜门未关/标签破损样本测试，告警准确率达 94.2%；
- 接入每日 18:00 自动汇总巡检日报 Excel 工作流。
## 3. 下一阶段计划
- 第 1～2 周：引入 300 条实验室规范问答构建双链 RAG 知识库；
- 第 3～4 周：采用 AMD Quark 量化压缩并完成毕业论文实验章节撰写。
EOF
```

### Step 3：在 WorkBuddy（Craft 模式）发送大师 PPT 制稿指令

将 WorkBuddy 工作区绑定至 `~/ai_workspace/sandbox/case01_ppt`，发送以下指令：

```text
请读取当前目录下的 defense_notes.md，调用 dashi-ppt-skill（大师 PPT Skill）制作一份 7 页学术答辩汇报 PPT：
1. 页面编排：包含答辩封面页、研究背景双栏痛点页、系统技术架构页、阶段实验成果 KPI 卡片页（突出 4.6GB 低显存、68 tok/s、94.2% 准确率）、下阶段时间轴路线图页与致谢页；
2. 禁止出现整页大段文字堆砌，每张卡片提炼精悍短句；
3. 启动本地交互预览控制台，并将原生可编辑文件保存至 output/中期答辩汇报.pptx。
```

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **冒烟与原生编辑测试**：用 WPS 或 PowerPoint 打开 `output/中期答辩汇报.pptx`，双击任意一页的标题和 `94.2%` 数字框直接打字修改、拖动色块，确认 **100% 为原生可编辑形状与文本框**。
- [x] **长输入边界测试**：将一份超过 4,000 字的课程综述 Markdown 丢入测试，确认 `32768` 上下文正常承载、无半途截断报错。
- [x] **离线隐私测试**：断开外网 Wi-Fi 后重新针对新提纲导出 `.pptx`，确认全程本地闭环可用。

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（15 分钟）**：把你下周组会或部门例会要讲的真实提纲拖进来，在大师 PPT 预览台对比切换 3 套不同配色主题，选出最适合你们团队审美的一套。
- **⭐⭐ 白银改造（30 分钟）**：串联 [Case 05](./case-05-paper-review-ppt.md)，先让 AI 读完 3 篇 PDF 论文写出横评表，再一气呵成自动生成组会汇报 PPT。
- **⭐⭐⭐ 黄金改造（进阶深造）**：定制属于你们学校/公司的专属汇报规范 `SKILL.md` 与大师 PPT 组合调用。
  - 🔗 **进阶学习入口**：**[Datawhale WorkBuddy 从入门到实战官方教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**（深入探索更多桌面智能体多技能办公编排！）

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：WorkBuddy 只在对话框里写出了一堆文字大纲，没有真正生成 `.pptx` 文件？**
  - **原因**：① 当前处于普通 `Chat` 模式而非 `Craft` 智能体模式；② 本地引擎启动时漏加了 `--jinja`（`llama-server`）或 `--enable-auto-tool-choice`（`vLLM`）。
  - **一行解决**：切换到 **Craft 模式**，并核对 [第 2.2.1 节五大避坑清单](../02-connect-tools/02-workbuddy-setup.md)。
- **Q2：运行 `npx dashi-ppt-skill@latest` 提示 `npx: command not found`？**
  - **原因**：本机尚未安装 Node.js 运行环境。
  - **一行解决**：前往 [nodejs.org](https://nodejs.org/) 安装 LTS 版 Node.js 后重开终端即可。

---

👉 下一篇案例：**[Case 02｜AI 零幻觉自动填 Excel 与杂乱文件夹智能归档](./case-02-auto-excel-archive.md)**
