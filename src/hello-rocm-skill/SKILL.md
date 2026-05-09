---
name: hello-rocm
description: >-
  hello-rocm 开源教程项目：AMD ROCm 上大模型环境、部署、微调与算子/Infra。
  当用户询问 AMD GPU、ROCm、Windows/Linux 上本地 LLM 推理与部署、vLLM/Ollama/LM Studio、
  LoRA 微调、HIP/RCCL、hello-rocm 仓库结构、学习顺序或贡献规范时，必须加载并遵循本 Skill。
---

# hello-rocm 项目 Skill

本 Skill 汇总 **hello-rocm** 仓库的定位、目录结构与推荐学习路径，便于助手准确指路（路径相对于仓库根目录）。

## 项目是什么

- **定位**：面向 **AMD ROCm** 的 **教程型** 开源仓库（主体是文档与 Notebook），补齐「CUDA 教程很多、ROCm 系统教程偏少」的空白。
- **一句话**：教用户在 **AMD GPU + ROCm** 上完成大模型 **环境安装 → 部署推理 → 微调 →（进阶）算子与 Infra**，并与 **Datawhale / 社区** 共建内容。
- **背景要点**（便于回答「为什么现在是 ROCm」类问题）：
  - **ROCm 7.10+** 起支持类似 CUDA 的方式在 **Python 虚拟环境** 中安装；**Linux 与 Windows** 均可作为学习与推理环境（详见 `00-Environment/README.md`）。
  - AMD 对 ROCm 面向 AI 场景持续迭代；官方文档以 [ROCm 文档](https://rocm.docs.amd.com/) 与 [Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) 为准。

## 仓库目录结构（必读）

| 目录 | 含义 |
|:---|:---|
| `00-Environment/` | **统一环境基线**：ROCm、驱动、Windows/Ubuntu、`uv` + PyTorch 等；**几乎所有后续章节的前置** |
| `01-Deploy/` | **部署推理**：LM Studio、vLLM、Ollama、llama.cpp 等；按 **模型系列** 分子目录（如 `models/Qwen3/`、`models/Gemma4/`） |
| `02-Fine-tune/` | **微调**：LoRA 等实践与 Notebook，同样按模型分子目录 |
| `03-Infra/` | **算子/基础设施**：HIPify、BLAS/DNN、NCCL→RCCL、Nsight→Rocprof 等迁移与优化 |
| `04-References/` | **官方与社区参考链接**汇总 |
| `05-AMD-YES/` | **社区案例**：终端助手、YOLO、Agent、分布式训练等实践 |

根目录还有：`README.md` / `README_en.md`、`CONTRIBUTING.md`、**`规范指南.md`**（教程排版与目录命名以 **Qwen3** 等目录为范例）。

## 推荐学习路径（AMD / 模型推理）

按 **README** 中的建议，向用户说明顺序时优先采用下面路径；可根据目标裁剪。

1. **先环境（必做）**  
   - 阅读并完成 **`00-Environment/README.md`**：对应平台的 ROCm、驱动、Python/`uv`、PyTorch 安装与校验。  
   - 需要换 GPU 架构或 pip 源时，对照 **`00-Environment/rocm-gpu-architecture-table.md`**。  
   - **平台提示**：Windows 适合体验与轻量推理；**完整工具链、多卡与工程化**更推荐 **Ubuntu 24.04**（见环境 README 中的说明）。

2. **再推理部署（模型推理入门）**  
   - 入口：**`01-Deploy/README.md`**。  
   - **零基础快速跑起来**：可在环境就绪后从 **LM Studio** 或 **vLLM** 入手（与 README「初学者」建议一致）。  
   - 具体模型请进 **`01-Deploy/models/<系列名>/`**（如 `Qwen3`、`Gemma4`），按同目录下的 `*-rocm7-deploy.md` 分框架操作。

3. **然后微调**  
   - 入口：**`02-Fine-tune/README.md`**，按 **`02-Fine-tune/models/<系列名>/`** 中的 Notebook/文档操作（含 SwanLab 等记录方式）。

4. **进阶：算子与 Infra**  
   - 入口：**`03-Infra/README.md`**，面向已有 CUDA/底层经验、需要做迁移与性能分析的用户。

5. **扩展阅读与实践**  
   - **`04-References/README.md`**：官方文档与新闻线索。  
   - **`05-AMD-YES/README.md`**：社区项目练手。

## 如何学「AMD」与「模型推理」（给用户的可复述要点）

- **学 AMD / ROCm**：以 **官方文档**为主线（版本、安装矩阵、已知问题），本仓库 **`00-Environment`** 提供与教程一致的 **实操步骤**；遇到框架差异时以当前 ROCm 版本对应页面为准。  
- **学模型推理**：先掌握 **一种部署栈**（例如 vLLM 或 Ollama）在同目录教程下跑通；再换框架或换模型系列，目录结构与文件名在仓库内保持一致（见 **`规范指南.md`**）。  
- **学微调**：在推理与环境稳定后进入 **`02-Fine-tune`**，从小参数 LoRA 示例开始，再扩展多机多卡等高级主题。

## 贡献与协作（简述）

- 贡献文档或新模型教程前：先读 **`规范指南.md`**，再读 **`CONTRIBUTING.md`**（Issue/PR 与目录约定）。  
- 新增部署文命名与结构应对齐 **`01-Deploy/models/Qwen3/`** 等现有范例。

## 何时查阅 references

需要 **链接清单或更长背景** 时，可读 `references/links-and-context.md`（本 Skill 附属参考，非仓库正文）。
