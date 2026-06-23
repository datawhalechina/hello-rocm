## Ubuntu 24.04 / Windows 11 Environment Setup: ROCm 7.13 + PyTorch + vLLM (gfx1151 Example)

**ROCm 7.13.0-preview environment setup guide for Qwen3.5 deployment.**

This guide uses **Ryzen AI Max / Ryzen AI Max+ (gfx1151)** as the reference GPU architecture and summarizes the key setup steps under ROCm 7.13 / TheRock.

> Official references:
> - [ROCm 7.13 installation guide (gfx1151)](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html?fam=ryzen&w=compute&os=windows&windows-ver=11&i=pip&gpu=max-pro-390&gfx=gfx1151)
> - [PyTorch 2.11.0 on ROCm 7.13 (gfx1151)](https://rocm.docs.amd.com/en/7.13.0-preview/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.11.0&w=compute&gpu=max-pro-390&gfx=gfx1151)
> - [vLLM 0.19.1 on ROCm 7.13 (gfx1151)](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/vllm.html?fam=ryzen&vllm-ver=0.19.1&i=docker&w=compute&gpu=max-pro-390&gfx=gfx1151)
> - [TheRock transition guide](https://rocm.docs.amd.com/en/7.13.0-preview/about/transition-guide-TheRock.html)

---

### 1. ROCm 7.13 / TheRock Notes

ROCm 7.13 uses the TheRock / Core SDK packaging model. Core components live under `/opt/rocm/core`, package names use the `amdrocm-*` prefix, and package-manager installs preserve common compatibility symlinks.

Qwen3.5 is a newer model family, so deployment also depends on recent vLLM / Transformers support for the `qwen3_5` model type.

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

```bash
sudo apt update
sudo apt install -y linux-image-6.14.0-1018-oem
sudo reboot
```

```bash
sudo apt update
sudo apt install -y \
  python3.13 python3.13-venv \
  libatomic1 libquadmath0 \
  build-essential git curl wget jq pciutils
```

```bash
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

---

### 4. Install PyTorch 2.11.0 (ROCm 7.13 / gfx1151)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.13 and create the virtual environment
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate

# Fallback:
# python3.13 -m venv .venv
# source .venv/bin/activate
# python -m pip install --upgrade pip

uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  "torch==2.11.0+rocm7.13.0" \
  "torchvision==0.26.0+rocm7.13.0" \
  "torchaudio==2.11.0+rocm7.13.0"
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

### 5. vLLM Environment Check

```bash
docker pull rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

> The Docker image includes PyTorch 2.10.0; PyTorch 2.11.0 is used by the pip path.

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

---

### 6. Next Steps

- [vLLM Deployment](./vllm-rocm7-deploy.md)
