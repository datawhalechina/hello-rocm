# hello-rocm Skill（2.0 伴学助教）

`hello-rocm Skill` 是本仓库内置的 AI 伴学导航技能，适配 **WorkBuddy**（`.codebuddy/skills/`）、**`Qoder`**（`.agents/skills/`）、**Codex** 与 **Claude Code**。

## 快速安装

```bash
mkdir -p .codebuddy/skills .agents/skills
cp -r src/hello-rocm-skill .codebuddy/skills/hello-rocm
cp -r src/hello-rocm-skill .agents/skills/hello-rocm
```

详细课程章节与全系 GPU `gfx` 字典见 `docs/zh/00-guide/` 与 `docs/zh/09-amd-ecosystem/`。
