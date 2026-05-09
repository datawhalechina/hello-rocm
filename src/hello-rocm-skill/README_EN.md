# hello-rocm-skill

A Claude Code skill that helps you navigate the [hello-rocm](https://github.com/datawhalechina/hello-rocm) tutorial project — deploy, fine-tune, and optimize LLMs on AMD GPUs with ROCm.

## Installation

Copy the `hello-rocm-skill/` directory to your IDE's skills folder:

| IDE | Destination |
|-----|-------------|
| **Claude Code** | `.claude/skills/hello-rocm-skill/` |
| **Cursor** | `.cursor/skills/hello-rocm-skill/` |
| **Windsurf** | `.windsurf/skills/hello-rocm-skill/` |
| **Trae** | `.trae/skills/hello-rocm-skill/` |
| **Codex / Qoder** | `.agents/skills/hello-rocm-skill/` |
| **Kiro** | `.kiro/skills/hello-rocm-skill/` |
| **OpenCode** | `.opencode/skills/hello-rocm-skill/` |
| **VS Code Copilot** | `.github/skills/hello-rocm-skill/` |

### One-command install (Claude Code)

```bash
git clone https://github.com/datawhalechina/hello-rocm.git
cp -r hello-rocm/hello-rocm-skill/ .claude/skills/hello-rocm-skill/
```

### Verify

Start a new conversation and say: **"How do I deploy a model on my AMD GPU?"**

## Quick Start by Persona

| I am... | Try saying... |
|---------|---------------|
| **New AMD device owner** | "Does my GPU support ROCm?" / "Help me install ROCm and run my first model" |
| **AI learner** | "I want to fine-tune a model with LoRA" / "Show me practice projects" |
| **Experienced developer** | "Help me write a HIP kernel" / "Deploy vLLM in production on ROCm" |

## What This Skill Does

- Detects your experience level from what you say (no questionnaire)
- Points you to the exact tutorial file you need in the hello-rocm project
- Summarizes relevant content and suggests next steps
- Falls back to AMD official docs for version-specific or edge-case questions

## What This Skill Does NOT Do

- Install software on your machine
- Diagnose hardware failures
- Replace official AMD support

## Files

| File | Purpose |
|------|---------|
| `SKILL.md` | Core agent instructions (consumed by LLM) |
| `skill.json` | Machine-readable skill manifest |
| `README.md` | Human-facing documentation (Chinese) |
| `README_EN.md` | Human-facing documentation (English) |
| `references/quick-deploy/SKILL.md` | 5-step lightning deploy checklist for new users |
| `references/troubleshooting/SKILL.md` | Common error patterns and fixes |

## Links

- [hello-rocm Project](https://github.com/datawhalechina/hello-rocm)
- [ROCm Official Docs](https://rocm.docs.amd.com/)

## Version

0.1.0 — Initial framework. Trigger tables will be expanded as the project grows.
