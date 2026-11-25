# Case 05：多篇英文文献横评对比表 & 组会汇报 PPT 一键串联生成

> 🏷️ **场景分类**：论文阅读（研究生与毕业设计开题救星！）  
> ⏱️ **预计耗时**：30 分钟 ｜ 🎮 **最低硬件档位**：🔵 甜点档 16G 独显 / 🟣 旗舰 AI PC / ☁️ Radeon Cloud

---

## 1. 🎬 效果演示（Demo）

- **你将亲手做出什么**：
  - **输入**：一个放了 **3～5 篇同领域英文顶会 arXiv 论文 PDF**（或提取好的摘要与方法章节）的本地文件夹；
  - **AI 自动执行**：WorkBuddy 驱动本地 **`Gemma 4 12B`**（英文文献理解与推理强项！）或 **`Qwen3.5-9B`**，先逐篇精读提取研究动机、核心架构创新、实验基准得分与局限性，拼成一张一目了然的 **《Related Work 文献横评矩阵表.md》**，紧接着无缝唤醒 **`dashi-ppt-skill`（大师 PPT）** 直出双语组会汇报课件；
  - **最终交付物**：可直接塞进毕业论文第一章综述的 Markdown 横评表 + 一份 8～10 页原生可编辑的 `本周文献精读组会.pptx`！

---

## 2. ✅ 支持配置清单（Verified Configurations）

| 配置项 | 已验证推荐组合 |
|:---|:---|
| **验证硬件与系统** | Windows 11 / Ubuntu 24.04（🔵 16G+ 独显容纳多篇论文长上下文） |
| **底层推理服务引擎** | `llama-server`（`:8080`）/ `lemonade-server`（`:13305`）/ `vLLM`（`:8000`） |
| **推荐主体模型** | 🔵 **`Gemma 4 12B`**（英文学术术语与实验表格解析表现极佳！）或 **`Qwen3.5-9B`**；🟣 **`Qwen3.6-35B-A3B`** |
| **必带服务端启动参数** | `--ctx-size 32768 -fa on --jinja`（显存 ≥20G 可开 `--ctx-size 65536`） |
| **配套技能组件** | WorkBuddy（Craft 模式）+ `dashi-ppt-skill` + Python `pypdf`/`pymupdf`（提取 PDF 文本） |

---

## 3. 💻 完整代码与分步实操（Code & Step-by-Step）

### Step 1：准备文献沙盒并安装轻量 PDF 解析依赖

```bash
PAPER_BOX=~/ai_workspace/sandbox/case05_papers
mkdir -p "$PAPER_BOX/pdfs" "$PAPER_BOX/output"
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pymupdf
```

将你手头正在读的 **3 篇英文论文 `.pdf`** 复制放进 `~/ai_workspace/sandbox/case05_papers/pdfs/` 目录（如果手头暂无论文，也可以从 arXiv 随便下载 3 篇关于 LoRA / RAG / MoE 的经典短论文放进去）。

### Step 2：在 WorkBuddy（Craft 模式）发送「两阶段分步核验 + 出 PPT」金牌指令

绑定工作区 `case05_papers`，发送以下遵循第 3 章防幻觉原则的结构化指令：

```text
请帮我完成 pdfs/ 目录下全部英文学术论文的精读横评与组会 PPT 制作，严格分两步执行：

第一步【精准提取与横评制表】：
1. 使用 pymupdf 提取 pdfs/ 下每篇论文的标题、Abstract、Method 核心段落与主实验结论段落；
2. 生成 `output/文献横评矩阵.md`，表格必须包含 6 列：【论文简称与年份 | 针对的核心痛点 | 核心方法创新点 | 实验数据集与关键提升数字 | 作者承认的局限性(Limitations) | 对我课题的启发】；
3. 防幻觉要求：实验数字必须与原文一致；若某篇未写 Limitations 则填“原文未单列说明”，严禁编造指标！

第二步【调用 dashi-ppt-skill 生成组会 PPT】：
基于核对好的 `output/文献横评矩阵.md`，调用 dashi-ppt-skill 生成一份 8 页学术清爽风组会汇报 PPT，保存至 `output/本周文献组会汇报.pptx`（包含：研究脉络总览页、每篇核心方法拆解卡片页、多篇横向对比表格页与未来改进切入点页）。
```

---

## 4. 🔍 验证方法与验收标准（Verification）

- [x] **数值零幻觉核查**：打开其中 1 篇原英文 PDF 的实验 Table，与 `output/文献横评矩阵.md` 中提取的提升百分比逐字比对，确认数字分毫不差。
- [x] **双产物闭环验证**：确认 `output/` 目录下同时躺着结构规整的 `.md` 横评表与可在 WPS/PowerPoint 中自由编辑公式备注的 `.pptx` 课件。

---

## 5. 🚀 扩展任务与专精课程导流（Extension Tasks）

- **⭐ 青铜改造（15 分钟）**：配合 [Case 06（`PDFMathTranslate`）](./case-06-pdf-math-translate.md)，把横评筛选出最值得细读的那 1 篇核心论文一键生成左英右中双语对照 PDF！
- **⭐⭐ 白银改造（30 分钟）**：调用第 4 章你的 `my-knowledge-vault` MCP，把每次组会生成的 `文献横评矩阵.md` 自动打上 `#论文精读` 标签归入 Obsidian，半年攒出一座无敌开题综述库！
- **⭐⭐⭐ 黄金改造（进阶深造）**：想处理包含复杂跨页实验大图与双栏公式的百篇文献库自动检索问答？
  - 🔗 **进阶学习入口**：**[Datawhale `all-in-rag`](https://github.com/datawhalechina/all-in-rag)** + **[Datawhale WorkBuddy 教程](https://www.datawhale.cn/learn/content/382/8712?from=homepage)**

---

## ❓ 新手试学常见问题（FAQ）

- **Q1：一次放进去 5 篇长达 30 页带大量附录的论文，提示上下文超长怎么办？**
  - **原因**：论文末尾几十页的 References 参考文献列表和 Appendix 占掉了大量无用 Token。
  - **一行解决**：在指令里加一句*“提取 PDF 文本时仅截取正文（遇到 `References` 标题即停止往后读取）”*，立刻立省 40% 上下文空间！

---

👉 下一篇案例：**[Case 06｜公式图表纹丝不乱：双语学术论文精翻神器（`PDFMathTranslate`）](./case-06-pdf-math-translate.md)**
