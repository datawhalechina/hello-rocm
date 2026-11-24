# TODO：hello-rocm 2.0 后续待办与外部支持清单（精简版）

> 以下事项涉及真机界面录屏、社群试学招募或发版决策，整理在此方便项目负责人直接对接支持：

---

## 1. 🖼️ 真机运行截图与 30 秒动图演示补充（建议优先级：高）
- **缺少的素材**：
  - 第 2.2 节 / Case 01：WorkBuddy 调用 `dashi-ppt-skill` 浏览器微调台及导出 PowerPoint/WPS 二次编辑的 **30 秒 GIF 动图**（建议存放于 `docs/public/images/02-connect-tools/`）；
  - 第 2.3 节：`Qoder` 设置页填入本地 `/v1` 地址并通过「校验并添加模型」的界面截图；
  - 第 6.2 节：在一张主流 Radeon 独显（如 RX 7800 XT / 9070 XT）或 Ryzen AI Max+ 395 上跑出的真实 `AMD Quark` 量化前后 `amd-smi` 显存对比截图。

## 2. 🧪 首批「新手试学官」盲测招募（建议优先级：高）
- **待办事项**：从 Datawhale 社群招募 **2～4 名零基础同学**（建议覆盖 **1 台 Windows 11 游戏本/核显本** + **1 台 Ubuntu / AMD Radeon Cloud 云工作区**），仅凭 `dev_1.0` 分支的第 0～2 章文档从零跑通 WorkBuddy 办公三件套，并通过 `.github/ISSUE_TEMPLATE/novice-trial-feedback.yml` 收集真实版本号与首批 FAQ。

## 3. 🌐 本地 Node 构建复核与英文版（`docs/en/`）同步策略（建议优先级：中）
- **本地 VitePress 打包预览**：当前本地工作区未安装 `node_modules`（已通过 `tests/test_course_v2.py` 完成全量路由与越界死链静态校验）；合入主分支前可在联网终端执行一次 `npm install && npm run docs:build` 预览线上视觉效果。
- **多语言策略确认**：目前 2.0 新增的 `00-guide` ～ `09-amd-ecosystem` 已在中文站（`docs/zh/`）完整落地，英文站（`docs/en/`）保留原有结构零报错；需确认是否需要将 2.0 新主线同步翻译至 `docs/en/` 与 `README_en.md`。

## 4. 🔀 `dev_1.0` 合入 `master` 发布时机（建议优先级：按宣发节奏）
- 当前全部改动安全隔离在远程 `origin/dev_1.0` 分支，待完成上述截图点缀与首轮试学内测后，发起 PR 合并至 `master` 即可自动触发 GitHub Actions 更新线上站点。
