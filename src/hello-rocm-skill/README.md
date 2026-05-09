# hello-rocm Skill

面向 AI 助手的 **Cursor / Codex 类 Skill**：加载后可根据用户问题，准确说明 **hello-rocm** 项目定位、目录结构与 **AMD ROCm + 大模型推理/微调** 的推荐学习顺序。

## 本 Skill 能做什么

| 场景 | 说明 |
|:---|:---|
| 解释项目 | 说明 hello-rocm 是 ROCm 上的教程仓库、包含哪些章节（00–05） |
| 学习路径 | 环境 → 部署推理 → 微调 → Infra；推理入门可从 LM Studio / vLLM 开始 |
| 指路 | 指向仓库内具体 `README.md`、模型目录（如 `01-Deploy/models/Qwen3/`） |
| 贡献 | 提示先读 `规范指南.md` 与 `CONTRIBUTING.md` |

## 安装方式

将本目录放到 Agent 的 Skill 目录下（任选其一，目录名可保持 `hello-rocm-skill` 或改名为 `hello-rocm`）：

| 工具 | 路径 |
|:---|:---|
| Cursor | `.cursor/skills/hello-rocm-skill/` |
| 通用 | `.agents/skills/hello-rocm-skill/` |
| Claude Code | `.claude/skills/hello-rocm-skill/` |

示例（在克隆下来的 hello-rocm 仓库旁或本仓库内）：

```bash
# 若本仓库已在本地，仅复制 skill 子目录到 Cursor 项目下：
mkdir -p .cursor/skills
cp -r hello-rocm-skill .cursor/skills/
```

确保目录内存在 **`SKILL.md`**，Agent 即可在匹配描述时加载。

## 文件说明

- `SKILL.md` — 元数据与 Agent 指令（核心）
- `references/links-and-context.md` — 官方链接与补充上下文

## License

与 hello-rocm 主仓库保持一致时使用 MIT（见仓库根目录 `LICENSE`）。
