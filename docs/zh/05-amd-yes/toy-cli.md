# 🧸 toy-cli - LLM 轻量化终端助手 AMD ROCm 版本

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)


</div>

**toy-cli** 是一个极简的命令行工具，用于快速调用大语言模型 API，采用简化的 Claude Code 风格的代码 Agent 设计。它让你在 3 分钟内就能上手大模型调用，是学习 API 调用的最佳入门项目。

> toy-cli 项目地址：[*Link*](https://github.com/KMnO4-zx/toy-cli.git)

### toy-cli 本地 Agent 调用链路

<div align='center'>
    <img src="../../../public/images/05-amd-yes/toy-cli/toy_cli_rocm_agent_flow_zh.png" alt="图5.1.1 toy-cli 本地 LLM Agent 调用链路" width="95%">
</div>

***OK，那接下来我将会带领大家亲自动手，一步步实现 toy-cli 的安装和使用过程，让我们一起来体验一下吧~***

## Step 1: 环境准备

本文基础环境如下：

```
----------------
LM Studio 
python 3.12
----------------
```
> 本文默认学习者使用的是 AMD ROCm 支持的显卡 或 搭载 Ryzen AI 系列芯片 AI PC 设备，LM Studio 本地加载大模型请参考 [Getting Started with ROCm Deploy](/zh/01-deploy/)


> 📖 `uv` 基础环境安装请参考 [00-Environment](/zh/00-environment/)，完成后再继续以下步骤。

- 使用 `uv` 下载并安装依赖包

```shell
uv pip install requests python-dotenv chardet
```

## Step 2: 本地模型配置

### 2.1 LM-Studio 加载 AgentCPM-explore 模型

<div align='center'>
    <img src="../../../public/images/05-amd-yes/toy-cli/AgentCPM-Explore-GGUF.png" alt="alt text" width="90%">
    <p>AgentCPM-explore model</p>
</div>

AgentCPM-Explore 是一款专为“端侧设备”打造的高性能开源智能体基础模型。它仅凭 4B（40 亿）的小巧参数量，就在多项复杂任务榜单中达到了同类领先水平，不仅支持超过 100 轮的超长持续交互与深度信息搜索，还配套开源了从强化学习训练到工具沙箱管理的全栈基础设施，让开发者能低门槛地构建具备“深度思考”能力的移动端 AI 助手。

> AgentCPM-Explore 项目地址：[*Link*](https://github.com/OpenBMB/AgentCPM)

### 2.2 配置端点地址和模型名称

在 LM-Studio 中查看端点地址（默认为： `http://127.0.0.1:1234` ）：

<div align='center'>
    <img src="../../../public/images/05-amd-yes/toy-cli/config_model.png" alt="alt text" width="90%">
    <p>AgentCPM-explore model</p>
</div>

使用 AMD ROCm 作为推理引擎：

<div align='center'>
    <img src="../../../public/images/05-amd-yes/toy-cli/ROCm_config.png" alt="alt text" width="90%">
    <p>AgentCPM-explore model</p>
</div>

在项目代码中配置模型名称 `agent.py` 第 365 行到第 372 行修改 `llm = LocalLLM(model="agentcpm-explore@q4_k_m")`  ：

```python
if __name__ == "__main__":
    # DeepSeek reasoner：
    # llm = DeepSeekLLM(model="deepseek-reasoner")
    # llm = DeepSeekLLM(model="deepseek-chat")
    # 其他：
    # llm = SiliconflowLLM(model="deepseek-ai/DeepSeek-V3.2")
    # llm = LocalLLM(model="openai/gpt-oss-20b")
    llm = LocalLLM(model="agentcpm-explore@q4_k_m")

    agent = Agent(llm=llm, use_todo=True)

    agent.loop()
```

在项目代码中配置模型端点地址 `llm.py` 第 126 行到第 127 行修改 `self.base_url`：

```python
class LocalLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "agentcpm-explore@f16"):
        super().__init__(api_key, model)
        self.api_key = api_key if api_key else "xxxxxxx"
        # self.base_url = "http://192.168.1.5:1234/v1"
        self.base_url = "http://127.0.0.1:1234/v1"
        self.model = model
        self.platform = "LMStudio"
```



### 2.3 不同量级本地模型的参数推荐建议

> 如果在使用过程中发现模型出现循环调用，重复思考的行为出现，可以参考以下指南修改部分参数

| 参数维度 | 4B / 8B (小模型) | 20B / 30B (中模型) | 120B-moe (大模型) |
| :--- | :--- | :--- | :--- |
| **核心目标** | **严谨性、防逻辑崩溃** | **平衡性、工具调用** | **深度推理、长上下文** |
| **Temperature** | `0.1` (或 0，追求极度稳定) | `0.6` (兼顾逻辑与灵活性) | `1.0` (释放推理潜力) |
| **Top_p** | `0.7` (强制过滤低概率噪声) | `0.85` (标准采样范围) | `1.0` (全开放，信任模型概率) |
| **上下文上限** | `8k - 16k` (防止注意力涣散) | `32k` (适合中型项目分析) | `128k+` (全库代码检索) |

#### 📖 参数补充说明

- 关于 Temperature:

    - 小模型容易在 1.0 温度下出现“Token 乱码”或逻辑死循环，因此建议“降温”。

    - 大模型（特别是 DeepSeek R1/o1 类推理模型）需要高温度来探索不同的推理路径，如果温度太低，反而会限制其思维深度。

- 关于 Top_p:

    - 对于小模型，将 Top_p 压低至 0.7 是防止其胡言乱语最有效的手段之一。

- 关于上下文:

    - 4B/8B 模型的有效注意力（Recall）通常在 16k 以后急剧下降。

    - 120B 规模的模型由于其参数冗余度高，即使在长文本末尾也能保持极高的逻辑准确度。

## Step 3: 使用示例

### 3.1 基础对话

```shell
    (toy-cli) PS C:\Users\aup\Desktop\ROCm\toy-cli> python .\agent.py
    Info: Using model: agentcpm-explore@q4_k_m || Platform: LMStudio 
    User: 你好，请介绍你自己写入 info.txt 文件中

    # Agent 自动规划任务 → 创建待办 → 写入文件 → 标记完成

    Tool: Calling tool: run_write
    Tool: Tool result: Wrote 214 bytes to info.txt (encoding: utf-8)
    
    你好！我已成功将自我介绍内容写入 `info.txt` 文件。你可以通过以下方式查看：

    ```
    cat info.txt  # Linux/Mac
    type info.txt  # Windows
    ```

    文件内容为：

    你好，我是ROCm的玩具CLI助手。我可以在Windows和Linux环境下执行命令、管理文件、运行脚本等操作。你可以通过简单的中文指令与我交互，我会尽力完成你的任务。
```
### 3.2 代码生成

```shell
    (toy-cli) PS C:\Users\aup\Desktop\ROCm\toy-cli> python .\agent.py
    Info: Using model: agentcpm-explore@q4_k_m || Platform: LMStudio 
    User: 生成一份可以删除文件的py代码，写入delete.py中

    # Agent 自动生成代码 → 写入文件 → 报告完成

    Tool: Calling tool: run_write
    Tool: Tool result: Wrote 580 bytes to C:/Users/aup/Desktop/ROCm/toy-cli/delete.py (encoding: utf-8)

    ✅ 已完成任务：生成可执行文件删除功能的 Python 脚本并写入 delete.py 文件中。

    **脚本内容说明**
    - 功能：接收文件路径参数，校验存在后执行删除，含错误处理
    - 执行方式：`python delete.py “C:\path\to\file.txt”`
```

### 3.3 文件处理

```shell
    (toy-cli) PS C:\Users\aup\Desktop\ROCm\toy-cli> python .\agent.py
    Info: Using model: agentcpm-explore@q4_k_m || Platform: LMStudio 
    User: 使用脚本delete.py删除info.txt文件            

    # Agent 自动规划：检查文件存在 → 执行删除 → 验证结果

    Tool: Calling tool: run_powershell
    Tool: Tool result: 确认 info.txt 和 delete.py 均存在

    Tool: Calling tool: run_powershell  
    Tool: Tool result: File info.txt deleted successfully.

    Tool: Calling tool: run_powershell
    Tool: Tool result: 确认 info.txt 已从目录中消失

    已完成：使用 delete.py 脚本删除 info.txt 文件  
    - 执行结果：File info.txt deleted successfully.  
    - 验证：info.txt 已成功删除
```
