# TODO — `hello-rocm 2.0` 后续发布上线与社区运营待办清单

> 本清单精简列出当前代码与全套 57 篇教材交付完毕后，需要项目负责人在发布与社区运营阶段确认或执行的待办事项（附直接可用的操作命令）。

---

## 一、上线发布类待办（需负责人确认操作）

### 1. 本地预览渲染效果并验证 VitePress 静态构建（可选）
当前仓库工作环境未预装 `node_modules` 前端依赖（路由与相对链接已通过 `tests/test_course_v2.py` 100% 校验）。如需在合并上线前本地启动浏览器预览排版、数学公式与 Mermaid 图表效果，可在终端执行：

```bash
npm install
npm run docs:build
npm run docs:preview
```

### 2. 发起 Pull Request 将 `dev_1.0` 合入 `master` 主分支正式上线
目前全部 `2.0` 重构成果均安全隔离在 `dev_1.0` 分支（`master` 仍保持 `78a6f19` 未动）。待负责人最终审阅确认无误后，可通过 GitHub 页面创建 `dev_1.0 -> master` 的 PR，或执行以下命令触发 GitHub Pages 文档站自动部署：

```bash
git checkout master
git merge --no-ff dev_1.0 -m "release: launch hello-rocm 2.0 full beginner-first curriculum"
git push origin master
```

---

## 二、外部配置与社区共学运营类待办（需团队协同支持）

### 3. 配置 GitHub Issue Template（「🧪 新手试学反馈卡」）
第 8 章与 `README.md` 已引导学习者提交「新手试学反馈卡」。建议在 `.github/ISSUE_TEMPLATE/` 下新增对应的 YAML 表单模板（包含：操作系统、AMD 显卡型号、所在章节/Case 编号、卡顿步骤与报错日志），方便第一期共学营收集高频 FAQ。

### 4. 更新课程总大纲 PDF 归档（如对外宣发需要）
若后续公众号推文或组队学习招募海报需要对外发放最新章节页码对应的 PDF 手册，可基于最新 `README.md` 与 `00-guide/04-roadmap-and-skill.md` 重新导出一版宣发版 PDF。

### 5. 第一期「20 天组队学习营」算力券与助教排班对接
对照更新后的 `docs/zh/learning/20-day-learning-plan.md`，与 AMD 开发者社区及魔搭社区确认新一期组队营的 **AMD Radeon Cloud 100 小时算力兑换券批次**，并在飞书共学文档更新 `hello-rocm 2.0` 新版学习入口链接。
