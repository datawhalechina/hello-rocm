## Ubuntu 24.04 Environment Setup: Installing ROCm 7.12 + PyTorch + vLLM (Using Ryzen AI Max+ PRO 395 / gfx1151 as Example)

**Ubuntu 24.04 (Linux) Inference Framework Deployment Guide with ROCm 7.12 Support — Environment Setup Section**

This section uses **AMD Ryzen AI Max+ PRO 395 (APU, gfx1151 architecture)** as a reference to walk through the following steps on Ubuntu 24.04:

- Clean up existing ROCm / AMD related environments
- Install **ROCm 7.12.0** using the official `apt` source
- Install **PyTorch 2.9.1** based on ROCm 7.12.0
- Pull up an inference service directly via the official **ROCm vLLM Docker image**

> Official documentation references:
> - [Install AMD ROCm 7.12.0](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html?fam=ryzen&gpu=max-pro-395&os=ubuntu&os-version=24.04&i=pkgman)
> - [Install PyTorch on ROCm 7.12.0](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/pytorch.html?fam=ryzen&gpu=max-pro-395&os=linux&os-version=24.04&i=pkgman)
> - [vLLM inference](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/vllm.html?fam=ryzen&gpu=max-pro-395&i=docker&os=linux&os-version=24.04)

> For other GPU architectures (e.g., Instinct MI350X=gfx950, MI300X=gfx94X, RX 9070=gfx120X, RX 7900=gfx110X, etc.), refer to the hardware selector in the official links above and simply replace `gfx1151` with the corresponding architecture name.

---

### 1. Clean Up Existing ROCm / AMD Related Software

If the system already has an older version of ROCm installed, it is recommended to clean it up first to avoid conflicts with ROCm 7.12:

```bash
sudo apt remove 'rocm*' 'amdrocm*' 'amdgpu-dkms*' -y
sudo apt autoremove -y
```

---

### 2. Prepare the System Environment (Ryzen APU Specific)

#### 2.1 Install the OEM Kernel (6.14)

Ryzen AI APUs on Ubuntu 24.04 require the OEM kernel 6.14 to properly drive the iGPU:

```bash
sudo apt update
sudo apt install -y linux-image-6.14.0-1018-oem
# Reboot after installation
sudo reboot
```

#### 2.2 Install ROCm Dependencies and Python

```bash
sudo apt update
sudo apt install -y libatomic1 libquadmath0
sudo apt install -y python3.12 python3.12-venv
```

#### 2.3 Configure GPU Access Permissions

Add the current user to the `render` and `video` groups (takes effect after reboot or re-login):

```bash
sudo usermod -a -G render,video "$LOGNAME"
```

(Optional) Use udev rules to grant system-level GPU access permissions:

```bash
sudo tee /etc/udev/rules.d/70-amdgpu.rules <<'EOF'
KERNEL=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="render", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

### 3. Install ROCm 7.12.0 (apt / pkgman Method)

> Note: Ryzen AI APUs use the inbox kernel driver bundled with Ubuntu 24.04 and do not require a separate `amdgpu-dkms` installation; for Instinct / Radeon discrete GPUs, please install the `amdgpu` driver separately following the official documentation.

#### 3.1 Register the ROCm apt Repository

```bash
# Download and write the GPG key
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.amd.com/rocm/packages/gpg/rocm.gpg -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

# Register the ROCm 7.12 source for Ubuntu 24.04
sudo tee /etc/apt/sources.list.d/rocm.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages/ubuntu2404 stable main
EOF

sudo apt update
```

#### 3.2 Install ROCm 7.12 Core Packages (gfx1151)

```bash
sudo apt install -y amdrocm7.12-gfx1151
```

> If you are using a different architecture, replace with the corresponding meta package, for example:
> - Instinct MI350X: `amdrocm7.12-gfx950`
> - Instinct MI300X / MI325X: `amdrocm7.12-gfx94x`
> - Radeon RX 9000 series: `amdrocm7.12-gfx120x`
> - Radeon RX 7000 series: `amdrocm7.12-gfx110x`
> - Ryzen AI 300 / Strix Halo (gfx1150): `amdrocm7.12-gfx1150`

#### 3.3 Configure Environment Variables

Configure for the current user (recommended):

```bash
tee --append ~/.bashrc <<'EOF'

# BEGIN ROCm environment configuration
export LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib
# END ROCm environment configuration
EOF

source ~/.bashrc
```

For system-level configuration:

```bash
sudo tee /etc/profile.d/set-rocm-env.sh <<'EOF'
export LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib
EOF
sudo chmod +x /etc/profile.d/set-rocm-env.sh
source /etc/profile.d/set-rocm-env.sh
```

#### 3.4 Verify Installation

Run the following commands in order; you should see GPU information and the ROCm version:

```bash
rocminfo
amd-smi version
amd-smi monitor
```

Example output of `amd-smi version`:

```
AMDSMI Tool: 26.3.0+2bd1678d3d | AMDSMI Library version: 26.3.0 | ROCm version: 7.12.0 | amdgpu version: 6.16.13 | ...
```

In `rocminfo`, you should see an entry like `AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S`.

---

### 4. Install PyTorch (ROCm 7.12 Version)

The official recommendation is to use a Python virtual environment + pip to install the ROCm PyTorch for `gfx1151`.

#### 4.1 Create and Activate a Virtual Environment

```bash
python3.12 -m venv ~/rocm-venv
source ~/rocm-venv/bin/activate
```

#### 4.2 Install ROCm Version of PyTorch

```bash
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  torch torchvision torchaudio
```

> To match vLLM 0.16 version requirements, you can use:
> ```bash
> python -m pip install \
>   --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
>   "torch==2.9.1+rocm7.12.0" \
>   "torchaudio==2.9.0+rocm7.12.0" \
>   "torchvision==0.24.0+rocm7.12.0"
> ```

#### 4.3 Verify PyTorch Can Use ROCm

```bash
python -c "import torch; print(torch.__version__); print('HIP available:', torch.cuda.is_available())"
```

If the expected output shows `True`, PyTorch + ROCm installation is successful.

---

### 5. Install vLLM (Docker Method, Recommended)

vLLM officially provides ROCm 7.12 Docker images for `gfx1151`, ready to use out of the box, avoiding the complexity of manually compiling Triton / FlashAttention.

> Prerequisite: Docker is already installed on the system.
> Reference: <https://docs.docker.com/engine/install/ubuntu/>

#### 5.1 Pull the ROCm vLLM Image (gfx1151)

```bash
docker pull rocm/vllm:rocm7.12.0_gfx1151_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0
```

> Image naming for other architectures follows the same convention — just replace `gfx1151`, for example:
> - `rocm/vllm:rocm7.12.0_gfx950-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0` (MI350X)
> - `rocm/vllm:rocm7.12.0_gfx94X-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0` (MI300X / MI325X)
> - `rocm/vllm:rocm7.12.0_gfx120X-all_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0` (RX 9000 series)

#### 5.2 Start the Container

```bash
mkdir -p ~/models

docker run -it --rm \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/models:/app/models \
  -e HF_HOME="/app/models" \
  -e HF_TOKEN="hf_***" \
  rocm/vllm:rocm7.12.0_gfx1151_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0 \
  bash
```

> Replace `HF_TOKEN` with a token generated on [Hugging Face](https://huggingface.co/settings/tokens) (for gated models like Gemma / Llama, you must first click **Agree & Access** on the model page).

#### 5.3 Temporary Workaround for Known Issues in the Container

The ROCm 7.12 Docker image may fail when starting vLLM due to path resolution issues. As recommended in the official Release Notes, add the following environment variable before starting vLLM:

```bash
export LD_LIBRARY_PATH=/opt/python/lib/python3.12/site-packages/_rocm_sdk_core/lib:$LD_LIBRARY_PATH
```

#### 5.4 Verify vLLM Inside the Container

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'HIP:', torch.cuda.is_available())"
```

#### 5.5 Start a Gemma 4 E4B Inference Service (Optional)

```bash
vllm serve google/gemma-4-E4B-it \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 32
```

Once the service is ready, you can use `curl http://127.0.0.1:8000/v1/models` to check the model list. For performance testing scripts, refer to the [vLLM Deployment Tutorial](./vllm-rocm7-deploy.md).

---

### 6. FAQ

<details>
<summary>Q1: After installing ROCm, `rocminfo` does not list any GPU?</summary>

1. Confirm that the OEM kernel 6.14 has been installed and the system has been rebooted;
2. Confirm that the current user is in the `render` / `video` groups — verify with the `groups` command;
3. Re-login or reboot to let the group permissions take effect.

</details>

<details>
<summary>Q2: `/dev/kfd` does not exist inside Docker?</summary>

1. On the host machine, first run `ls /dev/kfd /dev/dri` to confirm the devices exist;
2. Confirm that the Docker service is running and check the `docker info` output for the `Runtimes` section;
3. For Ryzen APUs using the inbox driver, `/dev/kfd` only appears with the OEM kernel.

</details>

<details>
<summary>Q3: Pulling the ROCm vLLM image is very slow?</summary>

You can configure a domestic mirror for Docker (`/etc/docker/daemon.json`), or use a proxy when running `docker pull`;  
Alternatively, you can `docker save` the image in an environment with good network access, then `docker load` it on the target machine.

</details>

---

After completing the steps above, you can proceed to model-specific tutorials:

- [LM Studio Deployment Tutorial](./lm-studio-rocm7-deploy.md)
- [Ollama Deployment Tutorial](./ollama-rocm7-deploy.md)
- [llama.cpp Deployment Tutorial](./llamacpp-rocm7-deploy.md)
- [vLLM Deployment Tutorial](./vllm-rocm7-deploy.md)
