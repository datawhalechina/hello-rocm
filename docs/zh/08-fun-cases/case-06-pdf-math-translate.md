# Case 06：公式图表纹丝不乱 —— 双语学术论文精翻神器（`PDFMathTranslate` + 本地高并发）

> 🏷️ **场景分类**：论文阅读  
> ⏱️ **预计耗时**：20 分钟 ｜ 🎮 **最低硬件档位**：🟢 入门档 8G（`llama-server` 稳健单翻）/ 🔵 甜点档 16G（搭配 `vLLM` 多线程并发，15 页顶会论文 1 分钟翻完！）

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：任意一篇满是复杂 LaTeX 行间数学公式、双栏排版、算法伪代码框与精细矢量折线图的英文学术 PDF 论文（或英文原版教材章节）；
  - **AI 自动执行**：开源明星神器 **[`Byaidu/PDFMathTranslate`（`pdf2zh`）](https://github.com/Byaidu/PDFMathTranslate)** 智能识别保留所有公式、图表、角标排版坐标不动，仅将正文段落并发发送给你本机的 AMD 显卡免费精翻后再原位回填；
  - **最终交付物**：同时输出两份排版堪称艺术品的 PDF —— **`paper-mono.pdf`（纯中文沉浸阅读版）** 与 **`paper-dual.pdf`（一页英文原版对照一页中文译版，读顶会论文爽到飞起！）**，且 **0 元翻译字数费**！

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Ubuntu 24.04 / Windows 11（全档位支持） |
| **底层推理服务引擎** | • 追求极速整本秒翻：强烈推荐 **`vLLM`（`:8000`）** 配合 `-t 8` 八线程并发批处理！<br/>• 个人轻量单篇翻译：**`llama-server`（`:8080`）** 或 **`lemonade-server`（`:13305`）** 配合 `-t 1` 或 `-t 2` |
| **推荐主体模型** | 🔵 **`Qwen3.5-9B`** 或 **`Gemma 4 12B`**（中英双语学术术语地道流畅）；🟢 **`Gemma 4 E4B`** |
| **客户端与扩展生态** | 命令行 `pdf2zh` / 内置图形网页 `pdf2zh -i` / Zotero 文献管理插件 |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

### Step 1：安装 `pdf2zh` 并配置国内版面分析模型镜像加速

`pdf2zh` 首次运行时会自动下载一个几十 MB 的文档版面识别小模型（DocLayout-YOLO），提前加上 `HF_ENDPOINT` 国内镜像可秒速下完：

```bash
cd ~/ai_workspace
uv venv .venv-pdf2zh --python 3.12
source .venv-pdf2zh/bin/activate
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pdf2zh

export HF_ENDPOINT="https://hf-mirror.com"
```

### Step 2：一行命令调用本地 AMD 显卡精翻整篇 PDF！

假设你有一篇待读论文 `attention_paper.pdf`，将 `OPENAI_BASE_URL` 指向你第 1 章启动的服务端口：

```bash
# 若使用 llama-server (:8080)，线程数 -t 设为 1～2；若使用 vLLM (:8000)，直接把端口改为 8000 并把 -t 开到 8 享受起飞速度！
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="local-rocm"

# 执行翻译（将 qwen3.5-9b 替换为你当前加载的模型别名，如 gemma-4-e4b；可用 -p 1-3 先试翻前 3 页！）
pdf2zh attention_paper.pdf -s openai:qwen3.5-9b -t 2 -p 1-3
```

### Step 3（零命令行党福利）：启动拖拽式图形网页界面！

如果你更喜欢用鼠标把 PDF 拖进网页窗口翻译并实时预览，只需输入：

```bash
pdf2zh -i
```
在浏览器打开提示的本地端口（如 `http://127.0.0.1:7860`），Service 选 `OpenAI`，填入本地 `/v1` 地址与模型名即可无限畅翻！

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **公式与图表无损验收**：打开生成的 `attention_paper-dual.pdf`，重点检查带上下标的复杂数学公式（如 $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$）、Table 三线表与 Figure 矢量图是否 **100% 保持原字体原位置不乱码、不重叠**。
- [x] **专业术语连贯性验收**：核对诸如 *Transformer / Residual Connection / Ablation Study* 等高频术语是否翻译得自然专业（而非生硬机翻）。

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（10 分钟）**：在 Zotero 文献管理软件中安装开源 `Zotero PDF2zh` 插件，后端指向本机 `:8080/v1`，实现在 Zotero 里右键任何新入库论文一键原地双语化！
- **⭐⭐ 白银改造（30 分钟）**：做一次控制变量对比实验——同一篇 10 页论文，分别测试 `llama-server (-t 1)` 与 `vLLM (-t 8)` 的整篇翻译总耗时，直观感受第 1.5 节讲过的 **Continuous Batching（连续批处理）** 威力！
- **⭐⭐⭐ 黄金改造（进阶深造）**：想进一步尝试专门针对多语种翻译优化的垂直开源模型部署？
  - 🔗 **进阶学习入口**：**[Datawhale `self-llm`（开源大模型食用指南）](https://github.com/datawhalechina/self-llm)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：翻译出来的中文框里，模型多输出了一句“以下是这段英文的中文翻译：”？**
  - **原因**：极个别小模型开启了过多寒暄或 Thinking 思考标签未过滤。
  - **一行解决**：升级至最新版 `pdf2zh`（已内置思考标签剥离），或优先使用指令遵循极干净的 `Qwen3.5-9B` / `Gemma 4 12B`。
- **Q2：首次运行卡在下载 `doclayout_yolo` 模型？**
  - **原因**：当前终端未设置 `HF_ENDPOINT` 国内镜像。
  - **一行解决**：先执行 `export HF_ENDPOINT=https://hf-mirror.com`（Windows PowerShell 执行 `$env:HF_ENDPOINT="https://hf-mirror.com"`）后重跑。

---

👉 下一篇案例：**[Case 07｜`WorkBuddy + Obsidian` 双链第二大脑 & 可共享「知识管理 MCP」](./case-07-obsidian-knowledge-mcp.md)**
