## MiniCPM-o Web Demo Full-Duplex Deployment (Ubuntu + ROCm 7+)

This section shows how to deploy the **MiniCPM-o 4.5 Web Demo** on AMD GPU, enabling full-duplex real-time conversation via microphone and camera in a browser. Four interaction modes are available once deployed:

| Path | Mode |
|------|------|
| `/turnbased` | Turn-based conversation (most stable, good for first test) |
| `/half_duplex` | Half-duplex voice interaction |
| `/omni` | **Omni full-duplex** (voice + camera + real-time voice reply) |
| `/audio_duplex` | Audio-only full-duplex |

> **Prerequisites**:
> - [ROCm environment setup](/00-environment/) completed.
> - [llama.cpp-omni CLI deployment](./llamacpp-omni-rocm7-deploy.md) completed — the `llama-server` binary is ready.
> - All GGUF model files downloaded (~8.3 GB in `~/omni/models/`).

---

### 1. Architecture Overview

The MiniCPM-o Web Demo uses a multi-process architecture:

```
Browser (microphone / camera)
    │  HTTPS / WebSocket
    ▼
Gateway (Python / FastAPI)          ← serves Web UI + API, port 8040
    │  internal HTTP
    ▼
Worker (Python / FastAPI)           ← manages session state, port 22440
    │  subprocess (os.environ.copy)
    ▼
llama-server (C++ inference)        ← llama.cpp-omni, port 19080
    │  loads
    ▼
GGUF model files (~/omni/models/)
```

- **Gateway** handles routing and serves frontend HTML/JS.
- **Worker** lazily spawns `llama-server` as a subprocess and proxies streaming API calls.
- **llama-server** loads LLM + vision/audio/TTS encoders and processes inference requests.

---

### 2. Clone MiniCPM-o-Demo (Comni branch)

The repository's `main` branch is PyTorch + CUDA only and **cannot run on AMD GPU**. Use the `Comni` branch, which replaces the backend with `llama.cpp-omni` while keeping the same frontend and gateway.

```bash
cd ~/omni

# Clone the repository
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git code/MiniCPM-o-Demo

# Option A: work directly on the Comni branch
cd code/MiniCPM-o-Demo
git checkout Comni
cd ~/omni
cp -r code/MiniCPM-o-Demo MiniCPM-o-Demo-Comni

# Option B (optional): use git worktree to keep both branches
# git worktree add ~/omni/MiniCPM-o-Demo-Comni Comni
```

---

### 3. Set Up the Python Environment

The `Comni` branch worker does not load PyTorch — dependencies are minimal:

```bash
# Create virtual environment
python3 -m venv ~/omni/venv
source ~/omni/venv/bin/activate

# Install dependencies (no torch/transformers needed)
pip install fastapi uvicorn httpx numpy pydantic websockets requests python-multipart

# Verify
python -c "import fastapi,uvicorn,httpx,numpy,pydantic,websockets,multipart,requests; print('deps OK')"
```

Symlink the venv to the path the Demo's `start_all.sh` expects:

```bash
cd ~/omni/MiniCPM-o-Demo-Comni
mkdir -p .venv
ln -sfn ~/omni/venv .venv/base
```

---

### 4. Write config.json

Create `config.json` in the Demo root to tell the Worker where to find `llama-server` and model files:

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

> Replace `<YOUR_USER>` with your actual username (e.g. `jovyan`, `user`). Use the absolute `$HOME` path if preferred.

**Key configuration fields**:

| Field | Description |
|-------|-------------|
| `backend` | Must be `"cpp"` to use llama.cpp-omni inference |
| `llamacpp_root` | llama.cpp-omni repo root; Worker finds `build/bin/llama-server` here |
| `model_dir` | GGUF file root directory; sub-models are looked up via fixed relative paths |
| `gateway_port` | External-facing port (browsers connect here) |
| `worker_base_port` | Internal worker port (not exposed externally) |

---

### 5. Create the AMD Launch Script

The bundled `start_all.sh` uses `nohup env CUDA_VISIBLE_DEVICES=<id> python worker.py ...` to start the Worker. The Worker spawns `llama-server` via `subprocess.Popen(..., env=os.environ.copy())`, inheriting all parent environment variables. **This means correctly setting the rocBLAS environment in the outermost process propagates all the way to `llama-server`.**

Create an AMD-specific wrapper script:

```bash
cat > ~/omni/MiniCPM-o-Demo-Comni/start_amd.sh << 'SCRIPT'
#!/bin/bash
# AMD GPU launch wrapper: inject the correct rocBLAS environment, then call start_all.sh
set -e

OMNI="$HOME/omni"

# ── gfx1151 (Strix Halo) users: use TheRock 7.12-alpha rocBLAS ──
# For other AMD GPUs not affected by the Tensile issue, replace the two lines below with:
#   export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
#   unset ROCBLAS_TENSILE_LIBPATH
SDK_LIB="$OMNI/rocm712/_rocm_sdk_libraries_gfx1151"
SDK_CORE="$OMNI/rocm712/_rocm_sdk_core"
export LD_LIBRARY_PATH="$SDK_LIB/lib:$SDK_CORE/lib"
export ROCBLAS_TENSILE_LIBPATH="$SDK_LIB/lib/rocblas/library"
# ─────────────────────────────────────────────────────────────────

export HIP_VISIBLE_DEVICES=0
# start_all.sh uses CUDA_VISIBLE_DEVICES to enumerate GPUs (avoids nvidia-smi)
export CUDA_VISIBLE_DEVICES=0

# Strip proxy: internal HTTP (Worker <-> llama-server, Gateway) must be direct
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

cd "$OMNI/MiniCPM-o-Demo-Comni"
exec bash start_all.sh "$@"
SCRIPT

chmod +x ~/omni/MiniCPM-o-Demo-Comni/start_amd.sh
```

> **Other AMD GPU users** (gfx1100 / gfx1150 etc.): Replace the `SDK_LIB` / `SDK_CORE` block with:
> ```bash
> export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
> ```

---

### 6. Start the Services

```bash
bash ~/omni/MiniCPM-o-Demo-Comni/start_amd.sh
```

Startup sequence:
1. **Worker** starts and begins loading models (~30–90 seconds on first launch)
2. Once Worker health check passes, **Gateway** starts (~2 seconds)

The script waits for the Worker to finish loading and prints:

```
==================================================
  Service is running!
  Chat Demo:  https://localhost:8040
  Admin:      https://localhost:8040/admin
  API Docs:   https://localhost:8040/docs
==================================================
```

---

### 7. Verify Service Status

```bash
# Worker health check (expect model_loaded: true)
curl -s http://localhost:22440/health | python3 -m json.tool

# Gateway health check (-k skips self-signed cert verification)
curl -sk https://localhost:8040/health | python3 -m json.tool

# Check all routes return 200
for path in "" turnbased half_duplex omni audio_duplex admin docs; do
    echo -n "/$path -> "
    curl -sk -o /dev/null -w '%{http_code}\n' "https://localhost:8040/$path"
done
```

Expected output:

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

### 8. Using the Web Demo

Open a browser on the same LAN and navigate to (replace `<HOST_IP>` with the server IP):

```
https://<HOST_IP>:8040/
```

> **Self-signed certificate warning**: The browser will warn about an insecure connection. Click "Advanced" → "Proceed to site".
>
> **Microphone/camera permissions**: Browsers require HTTPS to access media devices. The self-signed certificate satisfies this requirement for local LAN use.

**Mode entry points**:

| Feature | URL |
|---------|-----|
| Home (mode selection) | `https://<HOST_IP>:8040/` |
| Turn-based chat (recommended for first test) | `https://<HOST_IP>:8040/turnbased` |
| Omni full-duplex (voice + camera) | `https://<HOST_IP>:8040/omni` |
| Audio-only full-duplex | `https://<HOST_IP>:8040/audio_duplex` |
| Half-duplex | `https://<HOST_IP>:8040/half_duplex` |
| Admin panel | `https://<HOST_IP>:8040/admin` |

---

### 9. Stopping the Services

```bash
# Option A: via PID files
kill $(cat ~/omni/MiniCPM-o-Demo-Comni/tmp/*.pid 2>/dev/null) 2>/dev/null

# Option B: by process name
pkill -f 'gateway.py|worker.py' 2>/dev/null
pkill -f llama-server 2>/dev/null
```

---

### 10. Troubleshooting

<details>
<summary>llama-server exits immediately; logs show "hipErrorInvalidImage" or "Tensile" errors</summary>

The rocBLAS environment was not correctly injected. Check:
1. Confirm you're using `start_amd.sh`, not calling `start_all.sh` directly.
2. Verify the gfx1151 TheRock SDK paths exist:

```bash
ls ~/omni/rocm712/_rocm_sdk_libraries_gfx1151/lib/rocblas/library/ | head
```

If empty or missing, install the TheRock SDK fix as described in the [llama.cpp-omni tutorial "Troubleshooting" section](./llamacpp-omni-rocm7-deploy.md#5-troubleshooting).

</details>

<details>
<summary>Worker stays in non-idle state; logs show "FileNotFoundError: GGUF"</summary>

The Worker cannot find GGUF sub-model files. Check that `config.json`'s `model_dir` path exists and that sub-model file names match exactly (paths are case-sensitive):

```bash
ls ~/omni/models/vision/
# Must contain: MiniCPM-o-4_5-vision-F16.gguf

ls ~/omni/models/audio/
# Must contain: MiniCPM-o-4_5-audio-F16.gguf
```

</details>

<details>
<summary>Gateway health check OK but browser shows blank page or JS errors</summary>

Check that the `static/` directory contains HTML files:

```bash
ls ~/omni/MiniCPM-o-Demo-Comni/static/
```

The `Comni` branch includes pre-built frontend assets in `static/` — no local build step required. If files are missing, check the git checkout:

```bash
cd ~/omni/MiniCPM-o-Demo-Comni
git status
git checkout Comni -- static/
```

</details>

<details>
<summary>Microphone/camera permission denied</summary>

Full-duplex modes require HTTPS (not plain HTTP) and browser media permission. Verify:
1. Access via `https://` not `http://`.
2. Click the 🔒 icon in the address bar and manually allow microphone and camera.
3. If accessing through an HTTP reverse proxy over the internet, the proxy must terminate with real HTTPS (e.g. `tailscale serve` for automatic certificates).

</details>

<details>
<summary>Worker and Gateway cannot communicate in a proxy environment</summary>

A system HTTP(S) proxy (e.g. Cloudflare WARP) may route internal service calls through the proxy, causing failures. `start_amd.sh` already `unset`s proxy variables. If the issue persists, check `~/.bashrc` or `~/.profile` for persistent proxy settings and `unset` them before launching.

</details>

---

### References

- [OpenBMB/MiniCPM-o-Demo (Comni branch)](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/Comni)
- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [ROCm Official Documentation](https://rocm.docs.amd.com/)
- [GPU Architecture Table](/00-environment/rocm-gpu-architecture-table)
