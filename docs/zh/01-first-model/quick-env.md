# 1.2 极简环境准备（Windows 11 / Ubuntu 24.04 / 云算力基线）

本节为你提供**最低成本、不折腾内核**的快速起跑检查单。如果你的目的只是先跑起本地模型接入 WorkBuddy 和 `Qoder`，甚至无需先安装庞大的完整编译套件！

> 💡 如需查阅完整的 ROCm 10.0.0 / PyTorch 2.13 系统级安装与多架构 `pip` 源对照表，可随时参考仓库详细的 **[基础环境手册（00-Environment）](../00-environment/index.md)** 与 **[GPU 架构对照表](../00-environment/rocm-gpu-architecture-table.md)**。

---

## 一、Windows 11 用户（游戏本 / 台式机 / 锐龙 AI 轻薄本）

1. **更新显卡驱动**：确保已安装最新版 **AMD Software: Adrenalin Edition**（推荐 `26.x` 及以上版本）。
2. **安装极速 Python 包管理器 `uv`**（后续跑第 4 章知识管理 MCP 与评测脚本会用到）：
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. **准备非 C 盘工作与模型目录**（强烈建议！避免几十 GB 模型塞满系统盘）：
   ```powershell
   mkdir D:\AI_Models
   mkdir D:\AI_Workspace
   ```

---

## 二、Ubuntu 24.04 用户（本地工作站 / 双系统）

1. **一秒核对显卡与 ROCm 可见性**：
   ```bash
   amd-smi
   ```
   能正常看到你的 GPU 型号、温度与显存容量即代表底层驱动就绪。
2. **安装 `uv` 并创建干净的 Python 3.12 虚拟环境**：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   mkdir -p ~/ai_models ~/ai_workspace
   cd ~/ai_workspace
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   ```

---

## 三、手头暂无 AMD 显卡？免费云端算力 + 本地桌面工具无缝联调技巧！

如果你目前用的是轻薄办公本或 Mac，可以先免费申请 **[AMD Radeon Cloud](../cloud/amd-radeon-cloud.md)** 或 **[AUP Learning Cloud](../cloud/aup-learning-cloud.md)**。

**🌟 独家爽玩技巧（SSH 端口映射）**：在云端启动 `llama-server`（8080 端口）或 `vLLM`（8000 端口）后，在你自己的笔记本终端执行一条 SSH 本地端口转发命令：

```bash
# 将远程云主机的 8080 和 8000 端口映射到你本机的 127.0.0.1
ssh -L 8080:127.0.0.1:8080 -L 8000:127.0.0.1:8000 <你的云主机用户名>@<云主机IP>
```

这样一来，你本机桌面上安装的 **WorkBuddy** 和 **`Qoder`** 依然直接填写 `http://127.0.0.1:8080/v1`，就能像拥有一块本地旗舰显卡一样丝滑操作本机文件、做 PPT、填 Excel！

---

环境核对完毕，下一步进入 **[1.3～1.5 三大标准服务引擎极速拉起](./three-engines-deploy.md)**！
