# hello-rocm — 扩展链接与上下文

## 官方与文档（优先以最新版为准）

- ROCm 文档首页：<https://rocm.docs.amd.com/>
- ROCm Release Notes：<https://rocm.docs.amd.com/en/latest/about/release-notes.html>
- AMD GitHub 组织：<https://github.com/amd>

## 本仓库中的「地图」文件

- 根目录 `README.md`：总览、已支持模型表、各章节入口。
- `00-Environment/README.md`：ROCm + PyTorch + uv 基线与版本表。
- `01-Deploy/README.md`、`02-Fine-tune/README.md`、`03-Infra/README.md`：各阶段入口。
- `规范指南.md`：教程 Markdown 与目录命名约定。

## 与 README 一致的初学者节奏（摘录）

1. 完成 `00-Environment`。
2. 再学部署与微调。
3. 最后探索 `03-Infra` 算子优化。
4. 环境就绪后，可从 LM Studio 或 vLLM 部署入门。

## License

hello-rocm 项目仓库通常为 MIT（见根目录 `LICENSE`）。
