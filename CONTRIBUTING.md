# 贡献指南

感谢你愿意为 **hello-rocm** 添砖加瓦。本指南说明如何高效提交 Issue / Pull Request，并与仓库内 **Qwen3 / Gemma4** 等模型教程的组织方式保持一致。

对应英文版见 **[CONTRIBUTING_en.md](./CONTRIBUTING_en.md)**；更细的内容与排版要求见 **[CONTENT_GUIDE.md](./CONTENT_GUIDE.md)**。

## 项目结构概览

hello-rocm 以文档教程、示例代码和多语言 README 共同组织内容。贡献前建议先判断内容应该放在哪一类目录中：

```text
hello-rocm/
├── docs/                   # VitePress 文档源文件
│   ├── zh/                 # 中文文档
│   │   ├── 00-environment/ # ROCm 基础环境安装与配置
│   │   ├── 01-deploy/      # ROCm 大模型部署实践
│   │   ├── 02-fine-tune/   # ROCm 大模型微调实践
│   │   ├── 03-infra/       # ROCm 算子优化实践
│   │   ├── 04-references/  # ROCm 优质参考资料
│   │   └── 05-amd-yes/     # AMD 实践案例集合
│   └── en/                 # English docs，与中文文档保持对应结构
├── src/                    # 示例代码、Notebook、Skill 与项目源码
├── docs-readme/            # 多语言 README
├── scripts/                # 项目脚本
├── CONTENT_GUIDE.md        # 内容规范指南
└── CONTRIBUTING.md         # 贡献指南
```

## 我们欢迎哪些贡献

- **教程**：部署、微调、环境准备、排错与版本更新（ROCm / 框架 / 模型）。
- **修正**：错别字、失效链接、过时命令、图片无法显示等。
- **结构**：在不影响读者理解的前提下，优化目录或交叉引用。
- **讨论**：在 Issue 中反馈使用环境、硬件型号与复现步骤。

更细的内容、命名、图片与链接要求见 **[CONTENT_GUIDE.md](./CONTENT_GUIDE.md)**。

## 开始之前

1. 请先阅读 **[CONTENT_GUIDE.md](./CONTENT_GUIDE.md)**，避免路径、命名与图片引用与现有教程不一致。
2. 大型改动或新增模型系列，建议先开 **Issue** 简要说明计划，便于维护者对齐目录与 README 展示方式。

## 内容应该放在哪里

| 贡献类型 | 推荐位置 |
|:---|:---|
| ROCm / PyTorch / 基础环境说明 | `docs/zh/00-environment/`，英文同步放 `docs/en/00-environment/` |
| 模型部署教程 | `docs/zh/01-deploy/<model>/`，英文同步放 `docs/en/01-deploy/<model>/` |
| 模型微调教程 | `docs/zh/02-fine-tune/<model>/`，英文同步放 `docs/en/02-fine-tune/<model>/` |
| 微调 Notebook / 训练脚本 | `src/fine-tune/models/<model>/` |
| ROCm 算子 / HIP / PyTorch extension | 文档放 `docs/zh/03-infra/` 或 `docs/en/03-infra/`；代码放 `src/infra/` |
| 完整应用案例 / 社区项目 | 文档放 `docs/zh/05-amd-yes/<project>/` 或 `docs/en/05-amd-yes/<project>/`；源码放 `src/amd-yes/<project>/` |
| 参考资料 / 外部链接 | `docs/zh/04-references/` 与 `docs/en/04-references/` |

## 提交 Issue

- **Bug / 文档错误**：说明页面路径、预期与实际表现；若涉及命令，请贴完整报错与环境（系统、ROCm 版本、GPU 型号）。
- **内容建议**：希望增加的模型、框架或章节；如有可参考的上游文档请附链接。

## 提交 Pull Request

1. **Fork** 本仓库，从最新默认分支创建功能分支（示例：`fix/qwen3-image-path`、`feat/model-xxx-deploy`）。
2. 在分支上完成修改，**一次 PR 聚焦一类变更**（例如只修链接，或只新增一篇教程），便于审查。
3. PR 描述中请写清：
   - **动机**：解决什么问题或新增什么能力；
   - **变更摘要**：改了哪些文件；
   - **验证**：本地是否按文档步骤跑通（简要即可）。
4. 若新增或调整 **模型专项教程**，请检查根目录 **[README.md](./README.md)** 与 **[README_en.md](./README_en.md)** 的相关入口是否需要同步更新。

## 新增「模型专项」教程时的推荐结构

以下与 **[`docs/zh/01-deploy/qwen3/`](./docs/zh/01-deploy/qwen3/)** 和 **[`docs/zh/01-deploy/gemma4/`](./docs/zh/01-deploy/gemma4/)** 对齐，便于读者与维护者复用同一套组织方式。

```text
docs/zh/01-deploy/<model>/
├── env-prepare-ubuntu24-rocm7.md   # 环境准备（ROCm 安装与验证）
├── <model>_model.md                # 模型介绍（如适用）
├── lm-studio-rocm7-deploy.md
├── vllm-rocm7-deploy.md
├── ollama-rocm7-deploy.md
└── llamacpp-rocm7-deploy.md

docs/public/images/01-deploy/<model>/
├── image1.png                      # 部署教程配图统一放在 public 图片目录
├── image2.png
└── ...
```

- **微调文档**通常放在 **`docs/zh/02-fine-tune/<model>/`**，与部署目录区分。
- **微调代码 / Notebook** 通常放在 **`src/fine-tune/models/<model>/`**。
- 单篇教程内引用环境准备时，优先使用**相对路径**指向同目录下的 `env-prepare-ubuntu24-rocm7.md`。
- 教程配图不要放在 `docs/zh/01-deploy/<model>/images/`，应放在 `docs/public/images/01-deploy/<model>/`，并在 Markdown 中使用相对当前文件的路径引用，例如 `../../../public/images/01-deploy/qwen3/image11.png`。

## 行为准则

请保持友善、尊重与建设性沟通；对审查意见耐心迭代。我们重视可复现、可验证的技术表述。

## 许可

向本仓库贡献的内容，默认遵循仓库根目录 **[LICENSE](./LICENSE)**（MIT）的约定；若你引入第三方资源，请在 PR 中说明来源与许可。

---

**再次感谢你的时间与专业分享。**
