# 2.3 接入 `Qoder` 全离线 AI 编程助手：代码解释、重构与单测生成实战

> **本节目标**：将你在第 1 章启动的本地 AMD 大模型（强烈推荐代码与推理表现极佳的 **`Gemma 4 12B`** / **`Qwen3.5-9B`**，8G 档可用灵巧的 **`Gemma 4 E4B`**，旗舰档直上 **`Qwen3.6-35B-A3B`**）接入新一代 AI 原生编程 IDE **`Qoder`**，零 API 账单、100% 代码不出本机完成 **项目解读、坏味道重构与单元测试自动生成** 三连闯关！

无论是计算机专业的同学做课程设计、科研党改实验脚本，还是在企业内网开发保密业务代码，传统的云端编程插件都有两个绕不开的顾虑：**一是内部核心源码不允许上传公网模型，二是高频多文件对话每月动辄几十美元订阅费**。

在本节中，我们直接把 **`Qoder`** 的自定义模型插座连到本机 AMD 显卡服务上，打造一个既懂整个项目上下文、又一分钱不花的专属全离线编程搭档！

---

## Step 1：打开项目并配置 `Qoder` 本地 `OpenAI Compatible` 模型

1. 前往 [`Qoder` 官网](https://qoder.com/) 下载安装对应操作系统的客户端；
2. 打开 `Qoder`，点击 `File -> Open Folder` 打开本课程纯英文仓库目录（如 `~/ai_workspace/hello-rocm` 或 `D:\ai_workspace\hello-rocm`）；
3. 进入右上角或左下角 ⚙️ **`Settings（设置） -> Models（模型管理） -> Add Model / OpenAI Compatible（添加 OpenAI 兼容模型）`**；
4. 对照本仓库已验证通过的标准配置模板 **[`src/configs/qoder-openai-compatible.json`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/src/configs/qoder-openai-compatible.json)** 填入以下信息：

| `Qoder` 配置项 | 推荐填写值 | 避坑提示 |
|:---|:---|:---|
| **Provider / 协议** | `OpenAI Compatible` | 标准 OpenAI `/v1` 协议 |
| **Display Name（显示名称）** | `Local AMD ROCm - Gemma 4 12B / Qwen3.5-9B` | 方便在侧边栏一眼认出是本地免费模型 |
| **Base URL（接口根地址）** | • `llama-server`：`http://127.0.0.1:8080/v1`<br/>• `lemonade-server`：`http://127.0.0.1:13305/v1`<br/>• `vLLM`：`http://127.0.0.1:8000/v1` | 结尾写到 `/v1` 即可，勿带末尾斜杠或重复路径 |
| **API Key** | `local-rocm` | 任意非空英文串即可通过前端非空校验 |
| **Model ID（模型标识）** | `gemma-4-12b` / `qwen3.5-9b` / `gemma-4-e4b` / `Qwen3.6-35B-A3B-GGUF` | 与你当前后台引擎加载的模型别名完全一致 |
| **Context Length（上下文窗口）** | **`32768`** | 跨文件读源码必须保证 `32K` 窗口充裕 |
| **Tool Calling / Agent 支持** | **✅ 开启** | 允许 `Qoder` 智能体自动检索代码库与运行终端单测 |

点击面板上的 **`Verify / Test Connection`（一键测试连接）**，看到绿色的连接成功提示后，即可将其设为侧边栏对话与智能体默认模型！

> [!TIP]
> **⚡ 让 `Qoder` 多轮改代码“秒回”的性能小贴士**：  
> 在 IDE 里连续追问改代码时，前面几千行的项目文件上下文其实是固定不变的。还记得我们在 `1.3` 节启动 `llama-server` 时加的 `--cache-reuse 256`（以及 `1.5` 节 `vLLM` 的 `--enable-prefix-caching`）吗？开启它之后，**从第 2 轮提问开始，本地显卡无需重算旧代码，首字响应时间（TTFT）直接从数秒骤降到几百毫秒！**

---

## Step 2：三大真实编程场景闯关实战（跟着做一遍！）

既然我们已经打开了 `hello-rocm` 仓库，现在直接在 `Qoder` 右侧 AI 面板中，拿本仓库真实的工业级 Python 源码做三次实战演练：

### 🎯 第一关：10 秒读懂陌生源码架构与安全设计

在 `Qoder` 对话框中输入 `@` 引用本仓库第 4 章的核心源码文件 **`@src/main-assistant/knowledge_mcp/server.py`**，发送指令：

```text
请阅读 @src/main-assistant/knowledge_mcp/server.py，用通俗清晰的表格帮我梳理：
1. 这个 MCP 服务一共向外暴露了哪几个工具函数？各自的参数和返回值是什么？
2. 源码里的 _resolve_safe_folder 函数是如何防止恶意路径穿越（比如 ../../etc）攻击的？
```

观察你的本地模型：不到 10 秒钟，它就能把整份源码的接口契约和安全防御逻辑拆解得明明白白——以后接手师兄师姐留下的几千行祖传代码再也不慌了！

---

### 🎯 第二关：选中“坏味道”草稿代码，一键重构并补齐类型注解与异常保护

在沙箱目录新建一个故意写得很粗糙的测速计算小脚本 `~/ai_workspace/sandbox/calc_speed.py`：

```python
def calc(tokens, sec, old_tps):
    tps = tokens / sec
    gain = (tps - old_tps) / old_tps * 100
    return {"tps": round(tps, 2), "gain_pct": round(gain, 1)}
```

在 `Qoder` 编辑器中用鼠标全选这 4 行代码，唤起内联编辑或发送到右侧面板下达重构要求：

```text
请帮我按工业级 Python 3.12 规范重构这段函数：
1. 补充完整的类型注解（Type Hints）与中文 Docstring 说明；
2. 增加完备的边界条件防御：当 sec <= 0、old_tps <= 0 或 tokens < 0 时抛出清晰的中文 ValueError，防止除零崩溃；
3. 保持返回字典结构不变，代码精简易读。
```

点击 `Accept / Apply` 应用修改，看着简陋的 4 行草稿瞬间蜕变成严谨规范的生产级函数！

---

### 🎯 第三关：自动编写边界单元测试并在内置终端一键跑绿！

继续在 `Qoder` 智能体模式下追问一句：

```text
请为刚才重构好的 calc_speed.py 编写一份基于标准库 unittest 的单元测试文件 test_calc_speed.py，要求同时覆盖：
1. 正常提速计算场景；
2. tokens 为 0 的边界场景；
3. sec <= 0 或 old_tps <= 0 触发 ValueError 的异常场景；
写完后在终端运行测试确认 100% 通过！
```

当 `Qoder` 自动生成测试文件并在下方终端跑出 **`Ran 3 tests in 0.001s — OK`** 时，你已经完整跑通了 **「先定边界测试 → 重构实现 → 自动化验证」** 的现代 AI 辅助软件工程闭环！

---

## Step 3：给 `Qoder` 装上本仓库专属 `hello-rocm-skill` 记忆

如果你在 [第 0.4 节](../00-guide/04-roadmap-and-skill.md) 还没有把 `hello-rocm-skill` 复制进项目或全局 `.agents/skills/` 目录，现在顺手在 `Qoder` 终端里执行一行：

```bash
mkdir -p ~/.agents/skills && cp -r ./src/hello-rocm-skill ~/.agents/skills/hello-rocm-skill
```

从此以后，你在 `Qoder` 里开发任何与 AMD 显卡、PyTorch ROCm 10 或本地模型服务相关的项目时，它都会自动参考技能包里的已验证规范，绝不会给你瞎写过时的旧版命令！

---

## 👉 本章最后一站：走进极客最爱的终端智能体与百行源码解剖！

用爽了图形化的 WorkBuddy 与 `Qoder` IDE 之后，你心里一定会好奇：
*“为什么大模型明明只会吐文字，却能自己修 Bug、自己读写文件、自己跑命令？它肚子里的 Agent 循环到底长什么样？”*

在下一节，我们先花 5 分钟把 **`Codex CLI`** 接上本地 AMD 服务体验纯命令行自主修 Bug，再花 15 分钟带你逐行拆解经典百行终端智能体 **`toy-cli`**，彻底捅破 Agent 开发的最后一层窗户纸！

➡️ **[2.4 接入 `Codex CLI` 终端智能体 & 百行源码解剖 `toy-cli`](./06-codex-and-toy-cli.md)**
