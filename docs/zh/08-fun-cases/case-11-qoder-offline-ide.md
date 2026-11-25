# Case 11：`Qoder` 全离线 AI 编程搭档 —— 断网环境与内网代码重构单测生成实战

> 🏷️ **场景分类**：编程助手  
> ⏱️ **预计耗时**：25 分钟 ｜ 🎮 **最低硬件档位**：🟢 入门档 8G（`Gemma 4 E4B`）/ 🔵 甜点档 16G（强烈推荐 **`Gemma 4 12B`** 跨文件重构！）

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：一个处于飞行模式（完全断网）或严禁源码上传公网的本地项目仓库；
  - **AI 自动执行**：`Qoder` IDE 通过 `OpenAI Compatible` 接口直连本机 AMD 显卡服务（开启 Prefix Caching 后多轮追问秒回），自动阅读多个关联文件、按项目编码规范扩展新工具函数、自动编写 `unittest` 边界测试并在内置终端一键跑绿；
  - **最终交付物**：断网状态下为本仓库 `knowledge_mcp/server.py` 成功研发并跑通第 4 个原创统计工具函数（可直接收入你第 7 章简历作品集！）。

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Windows 11 / Ubuntu 24.04 / macOS |
| **底层推理服务引擎** | `llama-server`（`:8080`，加 `--cache-reuse 256`）或 `vLLM`（`:8000`，加 `--enable-prefix-caching`） |
| **推荐主体模型** | 🔵 **`Gemma 4 12B`** 或 **`Qwen3.5-9B`**；🟣 **`Qwen3.6-35B-A3B`**；🟢 **`Gemma 4 E4B`** |
| **标准配置模板** | 仓库 [`src/configs/qoder-openai-compatible.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/qoder-openai-compatible.json) |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

### Step 1：启动带前缀缓存的本地编程模型服务

```bash
llama-server \
  -m ~/ai_workspace/models/gemma-4-12b/gemma-4-12B-it-Q4_K_M.gguf \
  --alias gemma-4-12b \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -fa on --ctx-size 32768 --jinja --cache-reuse 256
```

### Step 2：在 `Qoder` 接入本地模型并添加项目编码规范 `SKILL.md`

1. 按 [第 2.3 节](../02-connect-tools/05-qoder-ide.md) 在 `Qoder` 模型设置中添加 `http://127.0.0.1:8080/v1`（API Key `local-rocm`，窗口 `32768`）并通过一键校验；
2. 创建项目级编码规范技能 `.agents/skills/clean-python/SKILL.md`：

```bash
mkdir -p ~/ai_workspace/sandbox/case11_qoder/.agents/skills/clean-python
cat << 'EOF' > ~/ai_workspace/sandbox/case11_qoder/.agents/skills/clean-python/SKILL.md
---
name: clean-python
description: Python 特性开发与重构规范。修改源码或生成单测时自动激活。
---
1. 必须包含完整 Python 3.12 类型注解与中文 Docstring；
2. 先写测试后写实现：同步补充覆盖【正常输入、空边界、非法参数异常】的 unittest 单测；
3. 严禁绕过沙盒路径安全校验。
EOF
```

### Step 3：断网实战 —— 让 `Qoder` 为你的知识库 MCP 新增统计工具！

在 `Qoder` 打开项目副本，发送以下开发任务：

```text
请阅读 @src/main-assistant/knowledge_mcp/server.py 与 @tests/test_course_v2.py：
为 KnowledgeVault 新增一个 `get_vault_stats()` 方法（返回字典：总笔记数 total_notes、去重标签总数 total_tags、全部标签有序列表 all_tags），同步在 build_mcp_server 中注册为 `@mcp.tool()`，补充对应单测并在终端运行确认 100% 通过！
```

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **单测全绿验收**：终端运行 `PYTHONPATH=. python tests/test_course_v2.py -v` 全部 `OK`。
- [x] **次轮秒回验收**：连续追问第 2 个问题时观察服务端命中 Prefix Cache，首字响应低于 0.5 秒。

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（15 分钟）**：把新增的 `get_vault_stats()` 收录进你的 [第 7.2 节五件套作品集](../07-portfolio-resume/02-portfolio-5-pieces.md)，作为你的个人独立扩展亮点！
- **⭐⭐ 白银改造（30 分钟）**：在 `Qoder` 同时挂载 `my-knowledge-vault` MCP，实现写代码遇到棘手报错自动查 Obsidian 历史错题本。
- **⭐⭐⭐ 黄金改造（进阶深造）**：深入学习 Coding Agent 的代码图谱检索与多智能体协同开发架构：
  - 🔗 **进阶学习入口**：**[Datawhale `hello-agents`（从零开始构建智能体）](https://github.com/datawhalechina/hello-agents)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：代码补丁偶尔缩进不齐？**
  - **原因**：采样温度偏高或使用了低于 4-bit 的极限有损量化。
  - **一行解决**：使用 `Q4_K_M / Q8_0` 权重并将采样温度控制在 `0.1 ~ 0.2`。

---

👉 下一篇案例：**[Case 12｜终端自主修 Bug 智能体与百行源码手搓（`Codex CLI` & `toy-cli`）](./case-12-codex-and-toy-cli.md)**
