---
name: hello-rocm-troubleshooting
description: 本地环境、三大推理引擎与桌面 Agent 接入高频报错速查
version: 2.0.0
---

# 高频报错速查

| 报错现象 | 根因定位 | 修复路径 |
|:---|:---|:---|
| `torch.cuda.is_available()` 为 `False` | Linux 未加 `render,video` 组或 APU 缺 6.14 OEM 内核；Windows 驱动低于 `26.8.1` 或未关 WDAG/SAC | 对照 `docs/zh/01-first-model/02-rocm-env-setup.md` 排错表逐项修复 |
| WorkBuddy 回复一片空白 | 默认 4096 上下文被内置工具定义撑爆截断 | 服务端启动加上 `--ctx-size 32768 --jinja` 或 `--max-model-len 32768`（见 `docs/zh/02-connect-tools/02-workbuddy-setup.md`） |
| WorkBuddy 报 `404 Not Found` | 混淆了 UI `/v1` 与 `models.json` `/v1/chat/completions` | 直接使用 `src/configs/workbuddy-models.json` 已验证模板 |
| 多轮 Agent 首字等待超过 8 秒 | 未开启前缀缓存导致每轮重复计算万字工具说明 | 开启 `--cache-reuse 256` 或 `--enable-prefix-caching`（见 `docs/zh/06-optimize-speed/`） |
