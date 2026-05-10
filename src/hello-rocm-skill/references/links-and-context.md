# hello-rocm — 扩展链接与上下文

## 官方与文档（优先以最新版为准）

- ROCm 文档首页：<https://rocm.docs.amd.com/>
- ROCm Release Notes：<https://rocm.docs.amd.com/en/latest/about/release-notes.html>
- AMD GitHub 组织：<https://github.com/amd>

## Skill 快速索引：框架与推理服务

当用户询问某个框架在 ROCm 上的安装方式、官方支持状态或实践案例时，优先索引仓库正文 `docs/zh/04-references/index.md` / `docs/en/04-references/index.md` 的“框架与推理服务（ROCm 快速安装入口）”表，再按用户目标跳转到项目内教程。

| 场景 | 官方快速入口 | AMD 官方实践参考 | 本项目入口 |
|:---|:---|:---|:---|
| PyTorch on ROCm | <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html> | <https://rocm.blogs.amd.com/search.html?q=PyTorch> | `docs/zh/00-environment/index.md` / `docs/en/00-environment/index.md` |
| TensorFlow on ROCm | <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html> | <https://rocm.blogs.amd.com/search.html?q=TensorFlow> | `docs/zh/00-environment/index.md` / `docs/en/00-environment/index.md` |
| JAX on ROCm | <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html> | <https://rocm.blogs.amd.com/search.html?q=JAX> | `docs/zh/00-environment/index.md` / `docs/en/00-environment/index.md` |
| vLLM on ROCm | <https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#amd-rocm> | <https://rocm.blogs.amd.com/search.html?q=vLLM> | `docs/zh/01-deploy/index.md` / `docs/en/01-deploy/index.md` |
| Ollama on AMD GPU | <https://github.com/ollama/ollama/blob/main/docs/gpu.md> | <https://rocm.blogs.amd.com/search.html?q=Ollama> | `docs/zh/01-deploy/index.md` / `docs/en/01-deploy/index.md` |
| llama.cpp HIP/ROCm | <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md> | <https://rocm.blogs.amd.com/search.html?q=llama.cpp> | `docs/zh/01-deploy/index.md` / `docs/en/01-deploy/index.md` |
| LM Studio GPU | <https://lmstudio.ai/docs/app/advanced/gpu> | <https://rocm.blogs.amd.com/search.html?q=LM%20Studio> | `docs/zh/01-deploy/index.md` / `docs/en/01-deploy/index.md` |
| ONNX Runtime on ROCm | <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/onnxruntime-install.html> | <https://rocm.blogs.amd.com/search.html?q=ONNX%20Runtime> | `docs/zh/00-environment/index.md` / `docs/en/00-environment/index.md` |

## Skill 快速索引：AMD GPU 架构资料

当用户询问“我的 GPU 属于什么架构”“CDNA / RDNA 有什么区别”“某个 gfx 目标对应什么架构”时，优先索引 `docs/zh/00-environment/rocm-gpu-architecture-table.md` / `docs/en/00-environment/rocm-gpu-architecture-table.md`，再补充下面 AMD 官方架构资料。

| 架构 | AMD 官方资料 |
|:---|:---|
| CDNA | <https://www.amd.com/en/technologies/cdna.html> |
| CDNA 2 | <https://www.amd.com/en/technologies/cdna-2.html> |
| CDNA 3 | <https://www.amd.com/en/technologies/cdna-3.html> |
| CDNA 4 | <https://www.amd.com/en/technologies/cdna-4.html> |
| RDNA | <https://www.amd.com/en/technologies/rdna.html> |
| RDNA 2 | <https://www.amd.com/en/technologies/rdna-2.html> |
| RDNA 3 | <https://www.amd.com/en/technologies/rdna-3.html> |
| RDNA 4 | <https://www.amd.com/en/technologies/rdna-4.html> |

## 社区讨论入口

- 故障排查与常见问题社区讨论（飞书）：<https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO>

## 本仓库中的「地图」文件

- 根目录 `README.md`：总览、已支持模型表、各章节入口。
- `docs/zh/00-environment/index.md` / `docs/en/00-environment/index.md`：ROCm + PyTorch + uv 基线与版本表。
- `docs/zh/01-deploy/index.md` / `docs/en/01-deploy/index.md`、`docs/zh/02-fine-tune/index.md` / `docs/en/02-fine-tune/index.md`、`docs/zh/03-infra/index.md` / `docs/en/03-infra/index.md`：各阶段入口。
- `规范指南.md`：教程 Markdown 与目录命名约定。

## 与 README 一致的初学者节奏（摘录）

1. 完成 `00-Environment`。
2. 再学部署与微调。
3. 最后探索 `03-Infra` 算子优化。
4. 环境就绪后，可从 LM Studio 或 vLLM 部署入门。

## License

hello-rocm 项目仓库通常为 MIT（见根目录 `LICENSE`）。
