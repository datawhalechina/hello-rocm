# 4.2 核心实战：把个人知识库制作成可共享的「知识管理 MCP Server」

> **本节目标**：基于官方轻量级框架 `FastMCP`，深入理解并运行本仓库内置的 **[`src/main-assistant/knowledge_mcp/server.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/knowledge_mcp/server.py)**，将第 3 章的 Obsidian 个人知识库封装成包含 **检索、制卡、按标签盘点** 三大标准接口的 MCP 服务——不仅实现你本机 **WorkBuddy + `Qoder` + `Codex` 三端互通**，更能直接**打包或开局域网端口分享给别人调用**！

---

## 一、先定接口契约：一个好用的「知识管理 MCP」该暴露哪 3 个工具？

按照工业级软件工程“先设计接口契约、再写实现”的好习惯，我们设计的「知识管理 MCP Server」既不能功能太碎（放二十个函数把本地小模型看晕），也不能缺少安全边界。精选出最实用的 **黄金三工具契约**：

| 工具函数名（Tool Name） | 输入参数 | 输出数据结构 | 解决什么真实场景？ |
|:---|:---|:---|:---|
| **`search_notes`** | `query: str`, `top_k: int = 5` | 匹配笔记列表（含 `title`、相对路径 `source`、`tags` 与命中上下文 `snippet`） | 在 `Qoder` 写代码或在 WorkBuddy 写报告时，随时检索自己记过的知识点并拿回精确出处路径！ |
| **`create_note_card`** | `title: str`, `content: str`, `tags: list[str]`, `folder: str = "00-收件箱"` | 保存成功后的知识库内相对路径确认消息 | 聊完一段精彩结论或排完一个 Bug，一句话让 AI 自动补齐 YAML 日期/标签头，写回 Obsidian 收件箱！ |
| **`list_notes_by_tag`** | `tag: str` | 包含该标签的所有笔记清单（同时识别 YAML `tags: [...]` 与正文 `#标签`） | 期末复习或写周报时，一句*“盘点所有带 `#必修考点` 或 `#周报` 标签的笔记”*瞬间拉出全目录！ |

---

## 二、Step 1：安装轻量依赖并逐块读懂核心安全源码

激活我们在第 1.2 节创建的虚拟环境 `.venv`，安装官方 `mcp` SDK（清华源 3 秒装完）：

```bash
cd ~/ai_workspace/hello-rocm
source ~/ai_workspace/.venv/bin/activate  # Windows 为 D:\ai_workspace\.venv\Scripts\Activate.ps1
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r src/main-assistant/knowledge_mcp/requirements.txt
```

现在打开本仓库配套源码 **[`src/main-assistant/knowledge_mcp/server.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/main-assistant/knowledge_mcp/server.py)**，重点看懂里面最值得写进你简历的 **两处核心工程设计**：

### 亮点 1：防目录穿越沙箱锁（`_safe_resolve`）
当你把 MCP 服务开放给大模型甚至局域网队友调用时，万一模型产生幻觉传入 `folder="../../etc"` 怎么办？源码第 28～32 行用标准 `Path.is_relative_to()` 筑起了铁壁防御：

```python
def _safe_resolve(self, relative_path: str | Path) -> Path:
    candidate = (self.vault_root / relative_path).resolve()
    if not candidate.is_relative_to(self.vault_root):
        raise ValueError(f"安全拦截：禁止访问知识库沙盒目录之外的路径 -> {relative_path}")
    return candidate
```

### 亮点 2：用 `@mcp.tool()` 装饰器 15 行零样板暴露标准协议
不需要手写任何繁琐的 JSON-RPC 通信报文！`FastMCP` 会**自动读取 Python 函数的类型注解（Type Hints）与中文注释（Docstring）**，自动生成大模型能看懂的工具说明书：

```python
def build_mcp_server(vault_dir: str | Path | None = None):
    from mcp.server.fastmcp import FastMCP

    root = vault_dir or os.environ.get("OBSIDIAN_VAULT_PATH", "./my-obsidian-vault")
    vault = KnowledgeVault(root)
    mcp = FastMCP("hello-rocm-knowledge-mcp")

    @mcp.tool()
    def search_notes(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索个人 Obsidian 知识库中的笔记卡片并返回出处路径与摘要"""
        return vault.search_notes(query=query, top_k=top_k)

    @mcp.tool()
    def create_note_card(
        title: str,
        content: str,
        tags: list[str] | None = None,
        folder: str = "00-收件箱",
    ) -> str:
        """在个人知识库指定分类文件夹下新建一张支持 [[双向链接]] 的 Markdown 笔记卡片"""
        return vault.create_note_card(title=title, content=content, tags=tags, folder=folder)

    @mcp.tool()
    def list_notes_by_tag(tag: str) -> list[dict[str, str]]:
        """按标签名称盘点知识库中所有关联的笔记清单"""
        return vault.list_notes_by_tag(tag=tag)

    return mcp
```

### 🔍 先跑一次单元测试验证你的 MCP 核心逻辑 100% 绿灯！

遵循先测后连的工程习惯，在终端直接运行本仓库配套的 `TestKnowledgeMCPServer` 单元测试（覆盖正常建卡检索、空输入边界与 `../../etc` 恶意穿越拦截）：

```bash
PYTHONPATH=. python tests/test_course_v2.py TestKnowledgeMCPServer -v
```

看到 `Ran 3 tests in 0.005s — OK`，说明你的服务端在逻辑与安全性上已经无懈可击！

---

## 三、Step 2：在 WorkBuddy、`Qoder` 与 `Codex` 中一键挂载互通

打开 **WorkBuddy** 或 **`Qoder`** 的 **`MCP Servers` 设置面板**，点击添加本地服务，粘贴以下标准 JSON（只需将两处路径替换为你自己电脑上的真实绝对路径！）：

### 🐧 Ubuntu / macOS 挂载配置示例

```json
{
  "mcpServers": {
    "my-knowledge-vault": {
      "command": "/home/你的用户名/ai_workspace/.venv/bin/python",
      "args": [
        "/home/你的用户名/ai_workspace/hello-rocm/src/main-assistant/knowledge_mcp/server.py"
      ],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/home/你的用户名/ai_workspace/my-obsidian-vault"
      }
    }
  }
}
```

### 🪟 Windows 11 挂载配置示例（注意 JSON 里反斜杠写成正斜杠 `/` 或双反斜杠 `\\` 最省心！）

```json
{
  "mcpServers": {
    "my-knowledge-vault": {
      "command": "D:/ai_workspace/.venv/Scripts/python.exe",
      "args": [
        "D:/ai_workspace/hello-rocm/src/main-assistant/knowledge_mcp/server.py"
      ],
      "env": {
        "OBSIDIAN_VAULT_PATH": "D:/ai_workspace/my-obsidian-vault"
      }
    }
  }
}
```

### 🎉 跨软件互通实测：在 `Qoder` 写代码时直接查你的 Obsidian 笔记！

现在打开 **`Qoder` IDE**（哪怕当前打开的是另一个毫不相干的代码仓库！），确认 `my-knowledge-vault` 旁边的绿灯亮起，直接对本地模型发话：

```text
请调用我的 my-knowledge-vault 知识库工具，先盘点一下带有 `rocm` 或 `排错` 标签的笔记有哪些，再搜索“models.json 后缀”，把查到的避坑要点总结给我，并把这次跨软件联调成功的记录调用 create_note_card 存回 `00-收件箱/Qoder联调成功纪要`！
```

看着 `Qoder` 依次自动调用 `list_notes_by_tag` → `search_notes` → `create_note_card`，再切回 Obsidian 一看，新卡片已经静静躺在收件箱里——**这种“全电脑所有软件共用一颗私人大脑”的丝滑体验，试过一次就再也回不去了！**

---

## 四、Step 3：🌟 怎么把你的个人知识库 MCP 给别人调用？（两种实战共享方案）

题目要求的核心高阶能力来了：**当你把某个领域的知识整理得炉火纯青，怎么把它变成别人也能调用的 MCP？** 根据你们是否在同一个局域网，有两种超好用的标准交付模式：

### 方案 A：「知识胶囊包」离线分发（适合发给异地同学、开源到 GitHub / 魔搭）

把以下 4 样东西放进一个干净的文件夹（例如 `lab-sop-mcp-pack/`）打成 `.zip` 或推上 Git 仓库发给同学：
1. `server.py` 与 `requirements.txt`（直接复用本仓库 `src/main-assistant/knowledge_mcp/`）；
2. `.env.example`（示范环境变量配置，绝不包含你个人的隐私路径）；
3. `shared-vault/`（脱敏筛选后专门用于分享的专题 Markdown 笔记库，比如《数模国赛常用算法模板库》或《课题组服务器新人上手手册》）；
4. 一份 10 行 `README.md`：告诉对方只需运行 `uv pip install -r requirements.txt`，然后把 `OBSIDIAN_VAULT_PATH` 指向解压后的 `shared-vault/` 即可在他的 WorkBuddy / `Qoder` 里零门槛调用！

### 方案 B：局域网 `SSE / Streamable HTTP` 实时共享服务（适合同宿舍、同实验室、同部门多人共用！）

如果你们 5 个人在同一个实验室 Wi-Fi 下做联合大作业，连文件都不用拷！你只需在自己开着 AMD 显卡的主机上，把启动传输方式从本机管道 `stdio` 切为网络 `sse` 模式（建议为共享专门指向一个只读或团队公共子目录 `team-shared-vault`，保护你自己的私人日记不被别人搜到！）：

```bash
# 在你的主机上用一行 Python 启动局域网共享知识库 MCP 端点（监听局域网 8765 端口）
OBSIDIAN_VAULT_PATH=~/ai_workspace/team-shared-vault python -c "
from src.main_assistant_pkg import load_knowledge_mcp
import os
from mcp.server.fastmcp import FastMCP

km_mod = __import__('importlib.util').util
# 直接调用 server.py 中的 build_mcp_server 并以 SSE 网络协议运行
import sys; sys.path.insert(0, 'src/main-assistant/knowledge_mcp')
import server
mcp = server.build_mcp_server(os.environ['OBSIDIAN_VAULT_PATH'])
mcp.settings.host = '0.0.0.0'
mcp.settings.port = 8765
mcp.run(transport='sse')
"
```

同局域网的室友或队友只需在自己 WorkBuddy / `Qoder` 的 MCP 配置里填一行 URL：
```json
{
  "mcpServers": {
    "team-shared-knowledge": {
      "url": "http://你的局域网IP:8765/sse"
    }
  }
}
```
无需拷一页文件，全队每个人的 AI 助手瞬间连上同一座实时同步的团队智库！

---

## 👉 下一节：给长出手脚的助手再配一本“超省显存的岗位 SOP 手册”！

现在你的助手已经有了随时查读写知识库的 **MCP 手脚**。但如果每次干活你都要手动敲两百字提示词教它“周报该分哪三段写、缺项该怎么标”，还是不够优雅；可要是把所有规矩全塞进默认 System Prompt，又会白白浪费好几千个显存 Token！

下一节，看 **`SKILL.md`（Agent Skills）** 如何用绝妙的「渐进式披露」四两拨千斤解决这个问题：

➡️ **[4.3 模块化技能包 Agent Skills（`SKILL.md`）：省显存原理与手写办公技能实战](./03-custom-office-skill.md)**
