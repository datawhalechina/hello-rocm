---
layout: home

hero:
  name: hello-rocm
  text: 每个 AI 小白，从零开始，动手实践的第一站。
  tagline: 基于 AMD 平台的 AI 与 Agentic AI 入门实践手册（全栈总览先导课）—— 把你的 AMD 电脑，变成学习和工作的 AI 助手。
  image:
    src: /images/logo.png
    alt: hello-rocm
  actions:
    - theme: brand
      text: 🚀 从第 0 章导读出发
      link: /zh/00-guide/
    - theme: alt
      text: 🌟 首发实战：WorkBuddy 办公三件套
      link: /zh/02-connect-tools/workbuddy-office-suite
    - theme: alt
      text: 🎮 15 个趣味实战案例库
      link: /zh/08-fun-cases/
    - theme: alt
      text: 🤝 组队学习
      link: /zh/learning/
    - theme: alt
      text: ☁️ AMD Radeon Cloud
      link: /zh/cloud/amd-radeon-cloud
    - theme: alt
      text: 🗺️ ROCm 图谱
      link: /atlas/rocm/index.html
      target: _blank
      rel: noreferrer
    - theme: alt
      text: English
      link: /

features:
  - title: 🚀 三大标准引擎极速部署
    details: 直观算清参数量、权重大小与运行显存“三本账”，收敛锁定 llama-server、lemonade-server 与 vLLM 最低成本跑起本地服务。
    link: /zh/01-first-model/
  - title: 💼 零代码办公与编程先行
    details: 第一步直连 WorkBuddy（dashi-ppt-skill 一键导出可编辑 .pptx、AI 自动填表、定时处理办公事项）以及 Qoder 与 Codex。
    link: /zh/02-connect-tools/
  - title: 🧠 个人知识库与可共享 MCP
    details: 闯关 6 节本地提示词课，用 WorkBuddy + Obsidian 搭建双链笔记库，并用 FastMCP 封装成可分享给别人调用的「知识管理 MCP」。
    link: /zh/04-mcp-and-skills/
  - title: 🔧 第一次中文模型微调与评测
    details: 基于魔搭纯中文生态，完成弱智吧 5 分钟破冰、中文群通知抽取与 WeClone/小红书网感分身微调，跑通自建测试集前后对比。
    link: /zh/05-fine-tune/
  - title: ⚡ AMD Quark 量化对比与自动加速
    details: 全程零 C++ 代码！直观实测同一模型量化前后体积与速度差异，唤醒 hipBLASLt / AITER 融合算子并体验 Hyperloom 自动寻优。
    link: /zh/06-optimize-speed/
  - title: 🏆 简历作品集与 AMD 生态远航
    details: 沉淀五件套简历成果与 15 个趣味案例，汇聚 AMD 官方开源宝藏项目，并精准导流至 Datawhale 各垂直专精课程继续深造。
    link: /zh/07-portfolio-resume/
---

## 🧭 6 站用户学习体验旅程（第 0 章 ～ 第 9 章）

| 站点与章节 | 你将亲手完成的核心实践 | 双向联动的 Datawhale 专精课程 |
|:---|:---|:---|
| **[第 0 章｜导读与硬件自检](/zh/00-guide/)** | 3 分钟核对「绿/蓝/紫/云」四档配置，看懂“把 AMD 电脑变成 AI 助手”全景地图 | 全栈总览入口 |
| **[第 1 章｜直觉认知与三大引擎部署](/zh/01-first-model/)** | 算清参数/显存/精度三本账，用 `llama-server` / `lemonade-server` / `vLLM` 拉起 `/v1` 服务 | 🔗 [`self-llm`](https://github.com/datawhalechina/self-llm) |
| **[第 2 章｜办公三件套与 IDE 接入](/zh/02-connect-tools/)** | **首发重点**：WorkBuddy（`dashi-ppt-skill` 出 PPT + 自动填表 + 定时任务）+ `Qoder` + `Codex` | 🔗 [Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage) |
| **[第 3 章｜提示词闯关与个人知识库](/zh/03-prompt-and-rag/)** | 6 关本地提示词训练 + 黑暗料理菜谱切分实验 + `WorkBuddy + Obsidian` 双链笔记库 | 🔗 [`all-in-rag`](https://github.com/datawhalechina/all-in-rag) |
| **[第 4 章｜知识管理 MCP 与 Skills](/zh/04-mcp-and-skills/)** | 把个人知识库制作成可共享给别人调用的 **知识管理 MCP** + 自定义办公 `SKILL.md` | 🔗 [`hello-agents`](https://github.com/datawhalechina/hello-agents) |
| **[第 5 章｜第一次中文模型微调](/zh/05-fine-tune/)** | 弱智吧破冰 + 中文通知抽取微调 + `WeClone` 微信分身/小红书种草体 + 自建测试集评测 | 🔗 [`happy-llm`](https://github.com/datawhalechina/happy-llm) |
| **[第 6 章｜AMD 量化与算子自动加速](/zh/06-optimize-speed/)** | `AMD Quark` 单模型量化前后速度/大小实测对比 + 零 C++ 唤醒融合算子 + `Hyperloom` 自动优化 | 🔗 [`hello-gpu`](https://github.com/datawhalechina/hello-gpu) |
| **[第 7 章｜作品集与简历成果锻造](/zh/07-portfolio-resume/)** | 打包个人 AI 作品集五件套、STAR 法则简历模板与高频面试追问指南 | 毕业深造驿站 |
| **[第 8 章｜15 个趣味实战案例库](/zh/08-fun-cases/)** | 办公三件套置顶，100% 覆盖 8 大日常场景，统一「五要素模板 + 新手试学 FAQ」 | 全场景实战 |
| **[第 9 章｜AMD 优秀项目与资源百宝箱](/zh/09-amd-ecosystem/)** | 汇聚 `amd/gaia`、`Hyperloom`、`Quark`、`AITER`、具身智能、推荐系统、HIP 算子与算力福利 | 生态进阶宝典 |
