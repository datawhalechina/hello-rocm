## Ubuntu 24.04 / Windows 11 Environment Setup: ROCm 7.13 + PyTorch + vLLM (gfx1151 Example)

**ROCm 7.13.0-preview environment setup guide for Gemma 4 deployment.**

This guide uses **Ryzen AI Max / Ryzen AI Max+ (gfx1151)** as the reference GPU architecture and summarizes the key setup steps under ROCm 7.13 / TheRock.

> Official references:
> - [ROCm 7.13 installation guide (gfx1151)](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html?fam=ryzen&w=compute&os=windows&windows-ver=11&i=pip&gpu=max-pro-390&gfx=gfx1151)
> - [PyTorch 2.11.0 on ROCm 7.13 (gfx1151)](https://rocm.docs.amd.com/en/7.13.0-preview/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.11.0&w=compute&gpu=max-pro-390&gfx=gfx1151)
> - [vLLM 0.19.1 on ROCm 7.13 (gfx1151)](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/vllm.html?fam=ryzen&vllm-ver=0.19.1&i=docker&w=compute&gpu=max-pro-390&gfx=gfx1151)
> - [TheRock transition guide](https://rocm.docs.amd.com/en/7.13.0-preview/about/transition-guide-TheRock.html)

---

### 1. ROCm 7.13 / TheRock Notes

ROCm 7.13 moves into the TheRock / Core SDK packaging model:

| Item | Legacy ROCm | ROCm 7.13 |
|:---|:---|:---|
| Core path | `/opt/rocm/` | `/opt/rocm/core` |
| Package prefix | `rocm-*`, `hip*`, `roc*` | `amdrocm-*` |
| Compatibility | Legacy ROCm layout | Core SDK with ABI / API compatibility and common symlinks |

When installed through package managers, common `/opt/rocm` symlinks are preserved. With tarball or custom installs, check `PATH`, `LD_LIBRARY_PATH`, and `ROCM_PATH` manually.

---

### 2. Clean Existing ROCm / AMD Components

If an older ROCm stack, HIP SDK, or `amdgpu-dkms` has been installed, clean it first to avoid conflicts with ROCm 7.13 / TheRock components:

```bash
sudo apt remove 'rocm*' 'amdrocm*' 'amdgpu-dkms*' -y
sudo apt autoremove -y
```

Also check `~/.bashrc`, `~/.zshrc`, and `/etc/profile.d/` for stale ROCm environment variables.

---

### 3. Ubuntu 24.04 + gfx1151 Setup

#### 2.1 Install OEM kernel 6.14

```bash
sudo apt update
sudo apt install -y linux-image-6.14.0-1018-oem
sudo reboot
```

#### 2.2 Install basic dependencies

```bash
sudo apt update
sudo apt install -y \
  python3.13 python3.13-venv \
  libatomic1 libquadmath0 \
  build-essential git curl wget jq pciutils
```

#### 2.3 Configure GPU permissions

```bash
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

Or use udev rules:

```bash
sudo tee /etc/udev/rules.d/70-amdgpu.rules <<'EOF'
KERNEL=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="render", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
sudo reboot
```

---

### 4. Install PyTorch 2.11.0 (ROCm 7.13 / gfx1151)

The project recommends using [uv](https://docs.astral.sh/uv/) to manage Python environments and dependencies instead of the traditional `pip + venv` flow.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.13
uv python install 3.13

# Create and activate the virtual environment
uv venv --python 3.13
source .venv/bin/activate

# Fallback: Python standard-library venv
# python3.13 -m venv .venv
# source .venv/bin/activate
# python -m pip install --upgrade pip
```

```bash
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  "torch==2.11.0+rocm7.13.0" \
  "torchvision==0.26.0+rocm7.13.0" \
  "torchaudio==2.11.0+rocm7.13.0"
```

Verify:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

---

### 5. Windows 11 + pip Path (ROCm 7.13)

On Windows 11, ROCm 7.13 uses the pip / TheRock flow. Before installation, uninstall any existing HIP SDK, disable WDAG / SAC, and install AMD Software: Adrenalin Edition 26.5.1 or newer.

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv python install 3.13
uv venv --python 3.13
.venv\Scripts\activate

# Fallback:
# py -3.13 -m venv .venv
# .venv\Scripts\activate

uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ `
  "torch==2.11.0+rocm7.13.0" `
  "torchvision==0.26.0+rocm7.13.0" `
  "torchaudio==2.11.0+rocm7.13.0"

python -c "import torch; print(torch.cuda.is_available())"
```

---

### 6. vLLM Environment Check (Docker)

The official ROCm 7.13 vLLM image for gfx1151 is:

```bash
docker pull rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

> Note: the vLLM 0.19.1 Docker image includes PyTorch 2.10.0. PyTorch 2.11.0 belongs to the pip installation path above.

```bash
docker run -it --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --network=host \
  --ipc=host \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ~/models:/app/models \
  -e HF_HOME="/app/models" \
  rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  bash
```

Inside the container:

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'HIP:', torch.cuda.is_available())"
```

---

### 7. Next Steps

- [LM Studio Deployment](./lm-studio-rocm7-deploy.md)
- [Ollama Deployment](./ollama-rocm7-deploy.md)
- [llama.cpp Deployment](./llamacpp-rocm7-deploy.md)
- [vLLM Deployment](./vllm-rocm7-deploy.md)
