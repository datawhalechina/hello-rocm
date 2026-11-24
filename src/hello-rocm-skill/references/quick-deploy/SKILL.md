---
name: hello-rocm-quick-deploy
description: 新用户闪电部署检查表 —— 最短路径跑通四大主体模型与三大标准服务引擎
version: 2.0.0
---

# 闪电部署：4 步跑通第一个本地模型并接入办公

| 步骤 | 做什么 | 仓库教程位置 |
|:---:|:---|:---|
| 1 | 查显卡档位与 `device-gfxXXXX` 架构标签 | `docs/zh/00-guide/03-hardware-and-cloud.md` & `docs/zh/01-first-model/02-rocm-env-setup.md` |
| 2 | 用 `uv` 安装 ROCm 10 + PyTorch 2.13 并跑通三连体检 | `docs/zh/01-first-model/02-rocm-env-setup.md` |
| 3 | 任选 `llama-server` / `lemonade-server` / `vLLM` 拉起本地 `/v1` 服务（必开 `32768` 上下文） | `docs/zh/01-first-model/03-llama-server.md` ～ `05-vllm-server.md` |
| 4 | 复制 `src/configs/` 配置接入 WorkBuddy、`Qoder` 或 Codex | `docs/zh/02-connect-tools/` |
