# 内容规范指南（hello-rocm 2.0 纯净版）

本文档约定 **hello-rocm 2.0** 中第 0～9 章教材文档、15 个趣味实战案例、已验证客户端配置文件与实战源码的目录、命名、排版与自动化自检规范。

对应贡献流程见 **[CONTRIBUTING.md](./CONTRIBUTING.md)**。

---

## 1. 目录与命名规范

| 内容类型 | 仓库路径约定 | 规范要求与示例 |
|:---|:---|:---|
| **主线章节文档（第 0～9 章）** | `docs/zh/00-guide/` ～ `docs/zh/09-amd-ecosystem/` | 每一核心小节独立成篇，采用两位数字前缀排序（如 `02-rocm-env-setup.md`、`03-llama-server.md`） |
| **趣味实战案例（第 8 章）** | `docs/zh/08-fun-cases/case-01-xxx.md` ～ `case-15-xxx.md` | 每个案例必须独立成篇，严格遵循 `docs/zh/08-fun-cases/_CASE_TEMPLATE.md` 五要素结构 + 试学 FAQ |
| **已验证客户端配置库** | `src/configs/` | 覆盖 `llama-server`、`lemonade-server`、`vLLM` 三大标准引擎与四大分档主体模型 |
| **主线助手与评测脚本** | `src/main-assistant/` | 包含 `knowledge_mcp/`、`skills/`、`eval/`、`quant_bench/`，均须配套单元测试 |
| **趣味案例配套源码** | `src/cases/` | 存放第 8 章各独立案例配套脚本与 Notebook |
| **配图资源目录** | `docs/public/images/` | 文内引用必须使用相对当前 Markdown 的路径（如 `../../public/images/...`），禁止写越出 `docs/` 的相对链接 |

---

## 2. 技术收敛红线（保持全书口径高度统一）

1. **四大分档主体模型**：
   - 🟢 **入门档（8G–12G）**：`Gemma 4 E4B`（`google/gemma-4-E4B-it`）
   - 🔵 **甜点档（16G–24G）**：`Qwen3.5-9B`（`Qwen/Qwen3.5-9B`）& `Gemma 4 12B`（`google/gemma-4-12B-it`）
   - 🟣 **旗舰档（64G–128G 统一内存 / 云）**：`Qwen3.6-35B-A3B`（`Qwen/Qwen3.6-35B-A3B`）
2. **三大标准推理服务引擎**：全书仅使用 **`llama-server`（`8080`）**、**`lemonade-server`（`13305`）**、**`vLLM`（`8000`）** 拉起 `/v1` 服务。
3. **办公与编程工具**：PPT 生成唯一使用 **`dashi-ppt-skill`**；桌面办公使用 **WorkBuddy**；编程 IDE 使用 **`Qoder`** 与 **`Codex CLI`**。

---

## 3. 提交前自动化自检清单

在提交 PR 前，请在仓库根目录执行自动化门控脚本并确保全部 `OK`：

```bash
PYTHONPATH=. python3 tests/test_course_v2.py -v
```

- [ ] 无任何越出 `docs/` 目录的 VitePress 非法相对链接，无任何残留旧路径引用；
- [ ] 所有图片路径与站内章节路由在磁盘上 100% 真实存在；
- [ ] 配置文件与源码无任何真实个人 API Key（统一使用 `.env.example` 占位）。
