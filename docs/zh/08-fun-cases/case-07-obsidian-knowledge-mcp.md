# Case 07：`WorkBuddy + Obsidian` 双链第二大脑 & 可共享「知识管理 MCP」

> 🏷️ **场景分类**：个人知识库  
> ⏱️ **预计耗时**：30 分钟 ｜ 🎮 **最低硬件档位**：🟢 入门档 8G 独显 / 锐龙核显本（全档位零门槛通吃）

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：你自己的本地 Obsidian Markdown 笔记仓库（课程讲义、错题本、面试八股、项目踩坑日志）；
  - **AI 自动执行**：一方面在 WorkBuddy 中通过 `obsidian-skills` 实现句句带 `[[文件名#小节]]` 双链溯源的零幻觉开卷问答；另一方面通过本仓库自研的 `FastMCP` 服务端，将整座知识库变成标准 **「知识管理 MCP」**——在 `Qoder` 写代码时跨软件随查随写，更能通过局域网 `SSE` 或知识胶囊包**直接分享给室友、师弟师妹或同事的 AI 助手调用**！
  - **最终交付物**：生长出漂亮双链关系图谱的私人第二大脑 + 一套通过全部安全单测的可共享知识库 MCP 服务！

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Windows 11 / Ubuntu 24.04 / macOS（连接远程/本机 AMD 引擎） |
| **底层推理服务引擎** | `llama-server`（`:8080`）/ `lemonade-server`（`:13305`）/ `vLLM`（`:8000`） |
| **推荐主体模型** | 🟢 `Gemma 4 E4B` · 🔵 `Qwen3.5-9B` / `Gemma 4 12B` · 🟣 `Qwen3.6-35B-A3B` |
| **核心配套源码** | 仓库 [`src/main-assistant/knowledge_mcp/server.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/knowledge_mcp/server.py) + [Obsidian](https://obsidian.md/) + [`obsidian-skills`](https://github.com/obsidianmd/obsidian-skills) |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

> 💡 本案例融合了主线 [第 3.3 节](../03-prompt-and-rag/03-workbuddy-obsidian-rag.md) 与 [第 4.2 节](../04-mcp-and-skills/02-knowledge-mcp-server.md) 的完整精华，下面给你一套 5 分钟端到端速通指令：

### Step 1：安装依赖并跑通服务端防越权单元测试

```bash
cd ~/ai_workspace/hello-rocm
source ~/ai_workspace/.venv/bin/activate
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r src/main-assistant/knowledge_mcp/requirements.txt

# 验证检索、双链建卡与防 ../../ 目录穿越安全锁 100% 绿灯！
PYTHONPATH=. python tests/test_course_v2.py TestKnowledgeMCPServer -v
```

### Step 2：在 WorkBuddy 与 `Qoder` 中同时挂载你的知识库 MCP

将以下配置加入客户端的 `MCP Servers` 设置（替换为你真实的绝对路径）：

```json
{
  "mcpServers": {
    "my-knowledge-vault": {
      "command": "/你的绝对路径/ai_workspace/.venv/bin/python",
      "args": ["/你的绝对路径/ai_workspace/hello-rocm/src/main-assistant/knowledge_mcp/server.py"],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/你的绝对路径/ai_workspace/my-obsidian-vault"
      }
    }
  }
}
```

### Step 3：实战体验“跨软件查笔记 + 自动沉淀复习卡”闭环

打开 `Qoder` 或 WorkBuddy，发送以下组合指令：

```text
1. 请调用 my-knowledge-vault 的 search_notes 工具检索我关于“显存”和“models.json”的笔记；
2. 严格依据查到的笔记作答，每条结论末尾标注 [[笔记标题]]，若问到库中不存在的“服务器机柜密码”必须直接回答未提及；
3. 调用 create_note_card 在 `01-课程讲义` 下创建一张名为 `MCP跨端问答速查卡` 的新卡片，打上标签 `["mcp", "知识库实战"]` 并用 [[双向链接]] 关联原笔记！
```

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **跨软件读写测试**：在 `Qoder` 里下达 `create_note_card` 指令后，切回 Obsidian 客户端点击「关系图谱（Graph View）」，确认新节点与双向连线实时出现。
- [x] **超纲陷阱防幻觉测试**：上述指令中的“服务器机柜密码”问题，确认模型诚实回答未提及、绝不瞎猜。
- [x] **同伴共享测试**：按 [第 4.2 节第四部分](../04-mcp-and-skills/02-knowledge-mcp-server.md) 开启局域网 `SSE` 模式或打包发给同桌，验证对方的客户端能直接调用 `search_notes` 查到你的专题笔记！

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（15 分钟）**：为你这学期的期末专业课新建一个专属文件夹 `04-期末冲刺考点`，导入 10 篇课堂重点，让 AI 每天随机抽 3 个知识点考你（苏格拉底式自测官！）。
- **⭐⭐ 白银改造（45 分钟）**：在 `server.py` 里动手新增第 4 个 `@mcp.tool()` 函数 `find_orphan_notes()`，自动帮你找出 Obsidian 里那些没有任何 `[[双链]]` 指向的孤岛笔记并推荐关联关系！
- **⭐⭐⭐ 黄金改造（进阶深造）**：给 `search_notes` 引入本地轻量中文向量模型（如 `BAAI/bge-small-zh-v1.5`）实现“关键词 + 语义向量”混合检索（Hybrid Search）！
  - 🔗 **进阶学习入口**：**[Datawhale `all-in-rag`](https://github.com/datawhalechina/all-in-rag)** + **[Datawhale `hello-agents`](https://github.com/datawhalechina/hello-agents)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：客户端 MCP 状态灯显示红色 `Failed to start`？**
  - **原因**：`command` 填了系统默认 `python`（里面没装 `mcp` 包），或者用了带 `~` 的相对路径（许多桌面客户端启动子进程时不展开 `~`）。
  - **一行解决**：`command` 与 `args` 一律填写完整**绝对路径**（Linux 用 `which python` 查虚拟环境解释器全路径）。

---

👉 下一篇案例：**[Case 08｜复习讲义秒变双人对话播客 & `WeClone` 网感分身联动](./case-08-podcast-and-weclone.md)**
