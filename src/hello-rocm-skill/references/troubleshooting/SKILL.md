---
name: hello-rocm-troubleshooting
description: 跨画像常见问题排查 —— 错误模式识别和修复建议
version: 0.1.0
---

# 常见问题排查

当用户遇到报错时触发。先匹配错误模式，再给出修复方案。

## 错误模式

| 错误现象 | 可能原因 | 修复方向 |
|---------|---------|---------|
| `rocminfo` 找不到 GPU | ROCm 未正确安装或 GPU 不兼容 | 先查 `04-References/README.md` 硬件兼容表，再对照 env-prepare 教程重装 |
| `torch.cuda.is_available()` 返回 False | PyTorch 不是 ROCm 版本 | 用 `--index-url https://repo.amd.com/rocm/whl/gfx...` 重装 PyTorch |
| 显存不足 (OOM) | batch size 过大或模型过大 | 减小 batch size、开启 gradient accumulation、启用 DeepSpeed ZeRO、使用 LoRA |
| HIP 编译错误 | ROCm 版本与代码不匹配 | 确认 `hipcc --version`，参考 `03-Infra/` 对应章节的 API 版本 |
| `Permission denied` | 用户权限不足 | 检查 Docker 用户组（`sudo usermod -aG docker $USER`），或文件权限 |
| 多卡训练速度不理想 | 通信瓶颈 | `rocm-bandwidth-test` 检查带宽，确认 `NCCL_DEBUG=INFO`（或 HIP 等效环境变量） |

## 通用排查流程

1. 先确认硬件：`rocminfo` / `rocm-smi`
2. 再确认软件：`python -c "import torch; print(torch.__version__)"`
3. 检查项目 FAQ：大部分 README 文末有 `<details>` FAQ 折叠区
4. 查 AMD 官方：`https://rocm.docs.amd.com/` 或 ROCm GitHub Issues
