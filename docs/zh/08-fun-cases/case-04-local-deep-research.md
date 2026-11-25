# Case 04：全本地万字深度调研报告专家 —— 用 `Local Deep Research` 零账单跑满多轮深挖

> 🏷️ **场景分类**：报告生成  
> ⏱️ **预计耗时**：35 分钟 ｜ 🎮 **最低硬件档位**：🔵 甜点档 16G 独显 / 🟣 旗舰 AI PC / ☁️ Radeon Cloud（多轮长文本综合对显存和推理力要求较高）

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：一句话研究课题（例如 *“对比分析 2026 年主流端侧大模型量化方案 AWQ、GPTQ、GGUF K-Quant 与 FP8 的原理及适用硬件”*，或挂载一个本地私有文献目录）；
  - **AI 自动执行**：开源深度调研智能体 **[`LearningCircuit/local-deep-research`](https://github.com/LearningCircuit/local-deep-research)** 驱动你的本地 AMD 模型，自动将大课题拆解为 4～6 个子问题 → 多轮迭代检索网页/arXiv/本地私有文档 → 交叉核对事实并补搜盲区 → 汇总撰写长篇结构化研报；
  - **最终交付物**：一篇逻辑严密、带真实可点击 `[1][2]` 引用文献角标的 **5,000～10,000 字 Markdown / PDF 深度调研报告**！
- **为什么它是本地 AMD 显卡的绝佳秀场？**  
  跑完一篇真正的 Deep Research 往往需要在后台循环调用大模型 **30～60 次、吞吐数十万 Token**！在商业 API 上跑几篇几十块钱就没了，而在本地 Radeon GPU 上开启 Prefix Caching 后**零成本随便跑**！

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Ubuntu 24.04 / Windows 11（🔵 16G～24G 独显 / 🟣 Max+ 395 128G / ☁️ Radeon Cloud） |
| **底层推理服务引擎** | 强烈推荐 **`vLLM`（`:8000`）** 或 **`llama-server`（`:8080`）**，**务必开启 Prefix Caching**！ |
| **推荐主体模型** | 🔵 **`Qwen3.5-9B`** 或 **`Gemma 4 12B`**；🟣 旗舰档 **`Qwen3.6-35B-A3B`**（万字长文综述质感极佳！） |
| **必带服务端启动参数** | `--ctx-size 32768 --cache-reuse 256`（`llama-server`）或 `--max-model-len 32768 --enable-prefix-caching`（`vLLM`） |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

### Step 1：确认本地长上下文服务已就绪

```bash
# 确认本地 8080 (llama-server) 或 8000 (vLLM) 在线
curl -s http://127.0.0.1:8080/v1/models
```

### Step 2：在独立虚拟环境中安装 `local-deep-research`

```bash
cd ~/ai_workspace
uv venv .venv-ldr --python 3.12
source .venv-ldr/bin/activate
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple local-deep-research
```

### Step 3：指向本机 `/v1` 接口启动可视化调研工作台

```bash
# 通过环境变量直接绑定本机 AMD 模型服务（以 8080 端口 qwen3.5-9b 为例，密钥仅存本地环境）
export LDR_LLM__PROVIDER="openai_endpoint"
export LDR_LLM__OPENAI_ENDPOINT_URL="http://127.0.0.1:8080/v1"
export LDR_LLM__MODEL="qwen3.5-9b"
export OPENAI_API_KEY="local-rocm"

# 一键启动本地 Web 调研控制台（默认监听 http://127.0.0.1:5000）
ldr-web
```

打开浏览器访问 **`http://127.0.0.1:5000`**：
1. 在设置页确认 LLM Provider 为 `openai_endpoint`，检索源可选免 Key 开箱即用的 `DuckDuckGo / Wikipedia / arXiv` **或直接添加你的本地私有文档文件夹（Local Collections）**；
2. 将迭代轮数 `Iterations` 设为 `2`、每轮子问题 `Questions per iteration` 设为 `3`；
3. 输入调研题目：*“系统对比 LoRA、QLoRA 与全参数微调在显存开销、适用场景与防遗忘策略上的异同”*，点击 **Start Research**！

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **多步规划可见性验证**：在进度面板亲眼看到本地模型先列出 3 个子查询方向，并在第 1 轮阅读摘要后自动提出第 2 轮补充问题。
- [x] **引用真实性抽查**：报告生成完毕后，随机点击文末 `Sources / References` 中的 3 条文献链接（或核对本地文档名），确认标题与正文论据真实对应、非凭空编造。
- [x] **导出归档验证**：将报告一键导出为 Markdown 保存至你的 Obsidian 仓库 `02-论文与读书/` 目录。

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（20 分钟）**：把检索源切为纯本地模式（Local Folder），指向你攒了十几篇行业研报 PDF 的文件夹，生成一份 100% 断网保密的行业竞品综述。
- **⭐⭐ 白银改造（40 分钟）**：把生成的深度调研 `.md` 报告直接丢给 [Case 01（`dashi-ppt-skill`）](./case-01-dashi-ppt.md)，实现 **“一句话输入课题 → 出万字研报 → 再出 10 页汇报 PPT”** 王炸连招！
- **⭐⭐⭐ 黄金改造（进阶深造）**：想自己用 Python 从零写出一个支持 Plan-and-Solve 子问题拆解与混合重排检索的 Deep Research 智能体？
  - 🔗 **进阶学习入口**：**[Datawhale `all-in-rag`](https://github.com/datawhalechina/all-in-rag)** + **[Datawhale `hello-agents`](https://github.com/datawhalechina/hello-agents)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：调研跑到最后一轮汇总长文时报上下文超限错误？**
  - **原因**：多轮累积的网页摘要超过了默认 4K/8K 窗口。
  - **一行解决**：确保第 1 章启动引擎时带了 `--ctx-size 32768`（显存充裕可开到 `65536`），并将 `Iterations` 控制在 `2～3` 轮。

---

👉 下一篇案例：**[Case 05｜多篇英文文献横评对比表 & 组会汇报 PPT 串联生成](./case-05-paper-review-ppt.md)**
