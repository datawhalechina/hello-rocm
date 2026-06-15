# 内容规范指南

本文档约定 **hello-rocm** 中教程类 Markdown、示例代码与配套资源的目录、命名、排版和提交前自检要求。当前仓库以 `docs/zh/01-deploy/qwen3/`、`docs/zh/01-deploy/gemma4/` 以及对应 `docs/en/` 目录为主要参考实现；新内容请尽量对齐现有组织方式，降低读者切换模型、语言或项目时的认知成本。

对应英文版见 **[CONTENT_GUIDE_en.md](./CONTENT_GUIDE_en.md)**；贡献流程见 **[CONTRIBUTING.md](./CONTRIBUTING.md)**。

## 1. 目录与命名

| 类型 | 约定 | 示例 |
|:---|:---|:---|
| 中文部署教程 | `docs/zh/01-deploy/<model>/`，模型名使用小写目录名 | `docs/zh/01-deploy/qwen3/`、`docs/zh/01-deploy/gemma4/` |
| 英文部署教程 | `docs/en/01-deploy/<model>/`，与中文目录保持同名结构 | `docs/en/01-deploy/qwen3/` |
| 中文微调教程 | `docs/zh/02-fine-tune/<model>/` | `docs/zh/02-fine-tune/qwen3/` |
| 英文微调教程 | `docs/en/02-fine-tune/<model>/` | `docs/en/02-fine-tune/gemma4/` |
| 微调代码 / Notebook | `src/fine-tune/models/<model>/` | `src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb` |
| ROCm 算子 / Infra 示例 | 文档放 `docs/zh/03-infra/` 或 `docs/en/03-infra/`；代码放 `src/infra/` | `docs/zh/03-infra/custom-pytorch-operator.md` |
| AMD 实践案例文档 | `docs/zh/05-amd-yes/<project>/` 或单篇 `docs/zh/05-amd-yes/<project>.md` | `docs/zh/05-amd-yes/hello-agents/`、`docs/zh/05-amd-yes/openclaw.md` |
| AMD 实践案例源码 | `src/amd-yes/<project>/` | `src/amd-yes/hello-agents/` |
| 配图目录 | 使用 `docs/public/images/` 中既有章节分类；文中引用时使用相对当前 Markdown 的路径 | `docs/public/images/01-deploy/qwen3/image11.png`，引用为 `../../../public/images/01-deploy/qwen3/image11.png` |

部署教程常用文件名请与现有 Qwen3 / Gemma4 教程保持一致：

- `env-prepare-ubuntu24-rocm7.md`：环境准备与验证。
- `lm-studio-rocm7-deploy.md`：LM Studio 部署。
- `vllm-rocm7-deploy.md`：vLLM 部署。
- `ollama-rocm7-deploy.md`：Ollama 部署。
- `llamacpp-rocm7-deploy.md`：llama.cpp 部署。

## 2. 文档结构（与现有教程对齐）

1. **一级标题 / 开篇**
   - 使用现有文件中的标题层级风格，不为单篇文档引入额外复杂结构。
   - 首段说明：**系统**（如 Ubuntu 24.04）、**ROCm 大版本**（如 ROCm 7+）、**主题**（某框架 + 某模型示例）。
   - 示例参考：`docs/zh/01-deploy/qwen3/lm-studio-rocm7-deploy.md`。

2. **前置条件**
   - 明确依赖的环境准备文档或已有章节，例如同目录下 `env-prepare-ubuntu24-rocm7.md`。
   - 如果依赖外部模型权重、数据集、账号权限或特定 GPU 型号，请在开头说明。

3. **分节**
   - 可使用「一、二、」或「### 1.」「#### 1.1」层级，**同一篇内保持一种风格**。
   - 大步骤之间可使用 `---` 分隔；不要为了统一而重排无关章节。

4. **命令与代码**
   - Shell 使用 ` ```bash ` 围栏；仅展示输出时用纯文本或注释说明。
   - 官方文档、脚本 URL 使用完整 `https://` 链接。
   - 命令涉及版本号时，尽量说明适用的 ROCm / PyTorch / 框架版本。

5. **图片**
   - 图片路径必须与仓库真实路径一致。
   - 当前项目的教程图片主要集中在 `docs/public/images/` 下，请按既有章节分类放置，例如 `docs/public/images/01-deploy/qwen3/`、`docs/public/images/05-amd-yes/toy-cli/`。
   - 图片引用必须使用**相对当前 Markdown 文件的相对路径**指向 `docs/public/images/` 中的真实文件；不要使用从仓库根开始的路径。这样在 GitHub Markdown 预览和 VitePress 页面中都能正常渲染。
   - 示例：若文档为 `docs/zh/01-deploy/qwen3/vllm-rocm7-deploy.md`，图片为 `docs/public/images/01-deploy/qwen3/image11.png`，推荐写法为：

     ```markdown
     ![vLLM ROCm 部署截图](../../../public/images/01-deploy/qwen3/image11.png)
     ```

     如需控制宽度，可使用 HTML，但仍保持相对路径：

     ```html
     <img src="../../../public/images/01-deploy/qwen3/image11.png" alt="vLLM ROCm 部署截图" width="90%" />
     ```

   - 另一个常见示例：`docs/zh/05-amd-yes/toy-cli.md` 引用 `docs/public/images/05-amd-yes/toy-cli/toy_cli_rocm_agent_flow_zh.png` 时，应写成 `../../public/images/05-amd-yes/toy-cli/toy_cli_rocm_agent_flow_zh.png`。
   - 不要写 `./images/media/` 等不存在的中间层级。

6. **交叉引用**
   - 同目录文件：使用 `./xxx.md`。
   - 同一语言文档内跨章节：使用相对路径或 VitePress 路由中现有写法。
   - 根目录文件：使用从当前文件出发可在 GitHub 上点击的相对路径。

## 3. README 与「已支持模型」

- 新增一整条模型线或改变模型展示方式时，应检查根目录 **[README.md](./README.md)** 与 **[README_en.md](./README_en.md)** 的相关表格或章节是否需要同步更新。
- 仅增加单篇子教程时，若适合暴露在首页或章节索引，可补充对应链接；否则至少确保所在章节的索引页或同目录导航能找到该内容。
- 多语言 README 位于 `docs-readme/`，通常链接到英文贡献指南和英文内容规范，避免每个翻译版本维护不同的贡献规则。

## 4. 中英文同步

- 中文贡献指南：`CONTRIBUTING.md`
- 英文贡献指南：`CONTRIBUTING_en.md`
- 中文内容规范：`CONTENT_GUIDE.md`
- 英文内容规范：`CONTENT_GUIDE_en.md`

中文与英文文档不要求逐字翻译，但目录结构、路径约定、提交要求和自检清单应保持一致。若本次 PR 只更新一种语言，请在 PR 描述中说明另一种语言是否需要后续同步。

## 5. 语言与风格

- 中文正文以**简体中文**为主；英文正文使用清晰直接的技术写作风格。
- 专有名词（ROCm、vLLM、Ollama、PyTorch、CUDA、LoRA 等）保持业界通用写法。
- 避免歧义：写清「宿主机 / 容器」「系统级安装 / 虚拟环境」「Windows / Ubuntu」等边界。
- 版本号、命令与官方文档变更较快，修改时尽量注明**针对的 ROCm 或框架版本**。

## 6. 自检清单（提交前）

- [ ] 文内所有相对链接在仓库中真实存在。
- [ ] 所有图片路径指向真实文件或现有公共图片目录。
- [ ] 新增模型线或重要入口已检查 README / README_en 是否需要更新。
- [ ] 中文与英文版本的结构和关键要求保持一致，或在 PR 中说明未同步原因。
- [ ] 无仅本地有效的绝对路径、账号信息、API key 或个人隐私内容。

---

维护者可根据仓库演进修订本页；有重大约定变更时，建议在 **[CONTRIBUTING.md](./CONTRIBUTING.md)** 中同步提示。
