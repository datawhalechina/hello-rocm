## MiniCPM-o Web Demo 全双工部署（Ubuntu + ROCm 7+）

本节介绍如何在 AMD GPU 上部署 **MiniCPM-o 4.5 Web Demo**，实现在浏览器中通过麦克风和摄像头与模型进行全双工实时对话。部署完成后可访问 4 种交互模式的 Web 界面：

| 路径 | 模式 |
|------|------|
| `/turnbased` | 轮询对话（最稳定，适合首次测试） |
| `/half_duplex` | 半双工语音交互 |
| `/omni` | **Omni 全双工**（语音 + 摄像头 + 实时语音回复） |
| `/audio_duplex` | 纯语音全双工 |

> **前置条件**：
> - 已完成 [ROCm 基础环境安装](/zh/00-environment/)。
> - 已按照 [llama.cpp-omni CLI 部署](./llamacpp-omni-rocm7-deploy.md) 完成编译，`llama-server` 二进制已就绪。
> - 已下载全部 GGUF 模型文件（约 8.3 GB，位于 `~/omni/models/`）。

---

### 一、架构概览

MiniCPM-o Web Demo 采用如下多进程架构：

```
浏览器（麦克风/摄像头）
    │  HTTPS / WebSocket
    ▼
Gateway（Python / FastAPI）          ← 对外提供 Web UI + API，端口 8040
    │  内部 HTTP
    ▼
Worker（Python / FastAPI）           ← 管理会话状态，端口 22440
    │  subprocess（os.environ.copy）
    ▼
llama-server（C++ 推理引擎）         ← llama.cpp-omni，端口 19080
    │  加载
    ▼
GGUF 模型文件（~/omni/models/）
```

- **Gateway** 负责路由和鉴权，提供前端 HTML/JS。
- **Worker** 懒启动 `llama-server` 子进程，通过 `/v1/stream/` 流式 API 与之通信。
- **llama-server** 同时加载 LLM + 视觉/音频/TTS 编码器，处理实际推理请求。

---

### 二、克隆 MiniCPM-o-Demo（Comni 分支）

官方仓库的 `main` 分支是纯 PyTorch + CUDA 版本，**无法在 AMD GPU 上运行**。需要使用 `Comni` 分支——该分支将后端换为 `llama.cpp-omni`，同一套前端和 gateway 保持不变。

```bash
cd ~/omni

# 克隆仓库
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git code/MiniCPM-o-Demo

# 方式一：直接在 Comni 分支工作
cd code/MiniCPM-o-Demo
git checkout Comni
cd ~/omni
cp -r code/MiniCPM-o-Demo MiniCPM-o-Demo-Comni

# 方式二（可选）：用 git worktree 同时保留 main 分支
# git worktree add ~/omni/MiniCPM-o-Demo-Comni Comni
```

---

### 三、准备 Python 环境

C++ 后端（`Comni` 分支）的 Worker 不加载 PyTorch，依赖极轻量：

```bash
# 创建虚拟环境
python3 -m venv ~/omni/venv
source ~/omni/venv/bin/activate

# 安装依赖（仅需以下包，无需 torch/transformers）
pip install fastapi uvicorn httpx numpy pydantic websockets requests python-multipart

# 验证
python -c "import fastapi,uvicorn,httpx,numpy,pydantic,websockets,multipart,requests; print('deps OK')"
```

然后将 venv 软链接到 Demo 仓库期望的位置：

```bash
cd ~/omni/MiniCPM-o-Demo-Comni
mkdir -p .venv
ln -sfn ~/omni/venv .venv/base
```

---

### 四、写入 config.json

在 Demo 根目录创建 `config.json`，告诉 Worker 在哪里找到 `llama-server` 和模型文件：

```bash
cat > ~/omni/MiniCPM-o-Demo-Comni/config.json << 'EOF'
{
    "backend": "cpp",

    "cpp_backend": {
        "llamacpp_root": "/home/<YOUR_USER>/omni/repo",
        "model_dir": "/home/<YOUR_USER>/omni/models",
        "llm_model": "MiniCPM-o-4_5-Q4_K_M.gguf",
        "cpp_server_port": 19080,
        "ctx_size": 8192,
        "n_gpu_layers": 99
    },

    "audio": {
        "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav",
        "playback_delay_ms": 200
    },

    "service": {
        "gateway_port": 8040,
        "worker_base_port": 22440,
        "num_workers": 1,
        "max_queue_size": 1000,
        "request_timeout": 300.0,
        "data_dir": "data"
    },

    "duplex": {
        "pause_timeout": 60.0
    }
}
EOF
```

> 将 `<YOUR_USER>` 替换为你的实际用户名（如 `jovyan`、`user` 等）。也可以直接使用 `$HOME` 绝对路径。

**关键配置说明**：

| 字段 | 说明 |
|------|------|
| `backend` | 必须为 `"cpp"`，使用 llama.cpp-omni 推理 |
| `llamacpp_root` | llama.cpp-omni 仓库根目录，Worker 在此目录下找 `build/bin/llama-server` |
| `model_dir` | GGUF 文件根目录，子模型按固定相对路径查找 |
| `gateway_port` | 对外服务端口（浏览器访问此端口） |
| `worker_base_port` | Worker 内部端口（不对外暴露） |

---

### 五、创建 AMD 启动脚本

仓库自带的 `start_all.sh` 用 `nohup env CUDA_VISIBLE_DEVICES=<id> python worker.py ...` 启动 Worker；Worker 在拉起 `llama-server` 子进程时通过 `os.environ.copy()` 继承父进程的环境变量。**这意味着只要在最外层正确设置 rocBLAS 环境变量，它就会一路传递到 llama-server。**

对于 gfx1151（或其他需要特殊 rocBLAS 路径的 GPU），请创建以下 AMD 专用启动包装脚本：

```bash
cat > ~/omni/MiniCPM-o-Demo-Comni/start_amd.sh << 'SCRIPT'
#!/bin/bash
# AMD GPU 启动脚本：注入正确的 rocBLAS 环境后再调用 start_all.sh
set -e

OMNI="$HOME/omni"

# ── gfx1151（Strix Halo）用户：使用 TheRock 7.12-alpha rocBLAS ──
# 如果你的 GPU 不受 Tensile 问题影响，将下面两行改为系统路径即可：
#   export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
#   unset ROCBLAS_TENSILE_LIBPATH
SDK_LIB="$OMNI/rocm712/_rocm_sdk_libraries_gfx1151"
SDK_CORE="$OMNI/rocm712/_rocm_sdk_core"
export LD_LIBRARY_PATH="$SDK_LIB/lib:$SDK_CORE/lib"
export ROCBLAS_TENSILE_LIBPATH="$SDK_LIB/lib/rocblas/library"
# ─────────────────────────────────────────────────────────────────

export HIP_VISIBLE_DEVICES=0
# start_all.sh 以 CUDA_VISIBLE_DEVICES 判断 GPU 列表（避免 nvidia-smi 报错）
export CUDA_VISIBLE_DEVICES=0

# 去掉代理：内部 HTTP（Worker <-> llama-server、gateway）必须直连
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

cd "$OMNI/MiniCPM-o-Demo-Comni"
exec bash start_all.sh "$@"
SCRIPT

chmod +x ~/omni/MiniCPM-o-Demo-Comni/start_amd.sh
```

> **其他 AMD GPU 用户**（gfx1100 / gfx1150 等）：将 `SDK_LIB` / `SDK_CORE` 两行替换为：
> ```bash
> export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
> ```
> 其余行保持不变。

---

### 六、启动服务

```bash
bash ~/omni/MiniCPM-o-Demo-Comni/start_amd.sh
```

启动顺序：
1. **Worker** 启动并开始加载模型（首次约 30–90 秒）
2. Worker 健康检查通过后，**Gateway** 启动（约 2 秒）

脚本会等待 Worker 加载完成，最终输出：

```
==================================================
  Service is running!
  Chat Demo:  https://localhost:8040
  Admin:      https://localhost:8040/admin
  API Docs:   https://localhost:8040/docs
==================================================
```

---

### 七、验证服务状态

```bash
# Worker 健康检查（期望 model_loaded: true）
curl -s http://localhost:22440/health | python3 -m json.tool

# Gateway 健康检查（自签证书用 -k 跳过验证）
curl -sk https://localhost:8040/health | python3 -m json.tool

# 检查各路由是否返回 200
for path in "" turnbased half_duplex omni audio_duplex admin docs; do
    echo -n "/$path -> "
    curl -sk -o /dev/null -w '%{http_code}\n' "https://localhost:8040/$path"
done
```

预期输出：

```
/           -> 200
/turnbased  -> 200
/half_duplex -> 200
/omni       -> 200
/audio_duplex -> 200
/admin      -> 200
/docs       -> 200
```

---

### 八、在浏览器中使用

在同一局域网的浏览器中访问（将 `<HOST_IP>` 替换为服务器 IP）：

```
https://<HOST_IP>:8040/
```

> **自签名证书警告**：浏览器会提示"不安全连接"，点击"高级"→"继续访问"即可。
>
> **麦克风/摄像头权限**：浏览器要求必须通过 HTTPS 才能访问媒体设备。自签证书满足此要求（在本地局域网中使用安全）。

**各模式入口**：

| 功能 | 地址 |
|------|------|
| 首页（模式选择） | `https://<HOST_IP>:8040/` |
| 轮询对话（推荐初次测试） | `https://<HOST_IP>:8040/turnbased` |
| Omni 全双工（语音+摄像头） | `https://<HOST_IP>:8040/omni` |
| 纯语音全双工 | `https://<HOST_IP>:8040/audio_duplex` |
| 半双工 | `https://<HOST_IP>:8040/half_duplex` |
| 管理界面 | `https://<HOST_IP>:8040/admin` |

---

### 九、停止服务

```bash
# 方式一：通过 PID 文件
kill $(cat ~/omni/MiniCPM-o-Demo-Comni/tmp/*.pid 2>/dev/null) 2>/dev/null

# 方式二：按进程名杀
pkill -f 'gateway.py|worker.py' 2>/dev/null
pkill -f llama-server 2>/dev/null
```

---

### 十、常见故障排查

<details>
<summary>llama-server 启动后立即退出，日志报 "hipErrorInvalidImage" 或 "Tensile" 错误</summary>

rocBLAS 环境没有正确注入。请检查：
1. 确认使用 `start_amd.sh` 而非直接调用 `start_all.sh`。
2. 确认 gfx1151 的 TheRock SDK 路径存在：

```bash
ls ~/omni/rocm712/_rocm_sdk_libraries_gfx1151/lib/rocblas/library/ | head
```

若目录为空或不存在，请按 [llama.cpp-omni 部署教程的「常见故障排查」](./llamacpp-omni-rocm7-deploy.md#五常见故障排查) 安装 TheRock SDK 修复。

</details>

<details>
<summary>Worker 一直不变成 idle 状态，日志报 "FileNotFoundError: GGUF"</summary>

Worker 找不到 GGUF 子模型。检查 `config.json` 中的 `model_dir` 路径是否存在，以及子模型文件名是否精确匹配（路径区分大小写）：

```bash
ls ~/omni/models/vision/
# 必须有：MiniCPM-o-4_5-vision-F16.gguf

ls ~/omni/models/audio/
# 必须有：MiniCPM-o-4_5-audio-F16.gguf
```

</details>

<details>
<summary>网关健康检查 OK，但浏览器打开页面空白或 JS 报错</summary>

前端静态文件未构建。检查 `static/` 目录是否存在 HTML 文件：

```bash
ls ~/omni/MiniCPM-o-Demo-Comni/static/
```

`Comni` 分支的 `static/` 目录中已包含预构建的前端资源，无需自行编译。若文件缺失，检查 git checkout 是否完整：

```bash
cd ~/omni/MiniCPM-o-Demo-Comni
git status
git checkout Comni -- static/
```

</details>

<details>
<summary>麦克风/摄像头权限被拒绝</summary>

全双工模式需要 HTTPS（非 HTTP）访问，且浏览器需授权媒体设备权限。确认：
1. 使用 `https://` 而非 `http://` 访问。
2. 浏览器地址栏点击🔒图标，手动允许麦克风和摄像头。
3. 若在外网通过 HTTP 反代访问，反代层必须升级为 HTTPS（可用 Tailscale `tailscale serve` 自动获取真实证书）。

</details>

<details>
<summary>代理环境下 Worker/Gateway 无法正常通信</summary>

系统设置的 HTTP(S) 代理（如 Cloudflare WARP）会导致 Worker 和 Gateway 的内部通信被路由到代理服务器，引发连接失败。`start_amd.sh` 已通过 `unset` 去除代理环境变量。若问题仍存在，检查 `~/.bashrc` 或 `~/.profile` 中是否有持久的代理设置，并在启动前手动 `unset`。

</details>

---

### 参考资源

- [OpenBMB/MiniCPM-o-Demo（Comni 分支）](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/Comni)
- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [ROCm 官方文档](https://rocm.docs.amd.com/)
- [GPU 架构对照表](/zh/00-environment/rocm-gpu-architecture-table)
