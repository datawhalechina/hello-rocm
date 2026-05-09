---
name: hello-rocm-quick-deploy
description: P1 新用户闪电部署检查表 —— 最简路径跑通第一个模型
version: 0.1.0
---

# 闪电部署：5 步跑通第一个模型

针对 P1（刚买 AMD 设备）用户的极简路径。只在用户说"我想最快跑起来"时触发。

## 检查表

| 步骤 | 做什么 | 参考文件 |
|------|--------|---------|
| 1 | 确认 GPU 支持 ROCm | `04-References/README.md` 硬件兼容表 |
| 2 | 安装 ROCm（pip 方式，最简单） | `01-Deploy/models/Gemma4/env-prepare-ubuntu24-rocm7.md` |
| 3 | 安装 LM Studio | `01-Deploy/models/Gemma4/lm-studio-rocm7-deploy.md` |
| 4 | 下载 Gemma4 模型并加载 | 同上 |
| 5 | 开始对话 | 同上 |

## 验证命令

```bash
rocminfo          # 确认 GPU 被 ROCm 识别
rocm-smi          # 查看 GPU 状态
python -c "import torch; print(torch.cuda.is_available())"  # PyTorch 确认 ROCm 可用
```

## 注意

- Windows 用户：确保 ROCm 7.12+ 版本（支持 Windows）
- Linux 用户：推荐 Ubuntu 24.04
- 如果 LM Studio 不工作，回退到 Ollama：`01-Deploy/models/Gemma4/ollama-rocm7-deploy.md`
