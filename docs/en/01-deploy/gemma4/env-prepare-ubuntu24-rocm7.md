## Ubuntu 24.04 / Windows 11 Environment Preparation: ROCm 10.0.0 + PyTorch + vLLM (gfx1151)

**ROCm 10.0.0 environment guide for deploying Gemma 4 inference frameworks.**

This section uses **Ryzen AI Max / Ryzen AI Max+ (gfx1151)** as the reference. For the full baseline and ROCm.AI notes, see [00-Environment](/00-environment/) and the [ROCm 10.0.0 release notes](/00-environment/rocm-10-0-0-release-notes).

> Official references:
> - [ROCm 10.0.0 docs home](https://rocm.docs.amd.com/en/latest/)
> - [Install ROCm 10.0.0](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&w=compute&os=windows&windows-ver=11&i=pip&gpu=max-395&gfx=gfx1151)
> - [PyTorch 2.13.0 on ROCm 10.0.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.13.0&w=compute&gpu=max-395&gfx=gfx1151)
> - [vLLM 0.27.0 on ROCm 10.0.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)
> - [TheRock transition guide](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html)

---

### 1. What changed from 7.13 / 7.14.0

| Item | ROCm 7.13 / 7.14.0 | ROCm 10.0.0 |
|:---|:---|:---|
| pip index | `repo.amd.com/rocm/whl/` or `whl-multi-arch/` | `https://stable.repo.amd.com/rocm/whl-next/` |
| PyTorch | 2.11.0 / 2.12.0 | **2.13.0** |
| vLLM | 0.19.1 / 0.23.0 | **0.27.0** |
| Windows driver | Adrenalin 26.5.1 | **Adrenalin 26.8.1** |
| One-command install | None | **ROCm CLI**: `rocm install sdk` |

`/opt/rocm/core` and the `amdrocm-*` prefix continue from 7.14.0. AMD Skills, Hyperloom, and ROCm CLI are new in 10.0.0.

---

### 2. Clean existing ROCm / AMD software

```bash
sudo apt remove 'rocm*' 'amdrocm*' 'amdgpu-dkms*' -y
sudo apt autoremove -y
```

On Windows, uninstall the old HIP SDK and install [Adrenalin 26.8.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads). Disable WDAG / SAC.

---

### 3. Ubuntu 24.04 + gfx1151

```bash
sudo apt update
sudo apt install -y linux-oem-24.04c libatomic1 libquadmath0 build-essential git curl wget jq pciutils
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" \
  "torchaudio==2.11.0.2+rocm10.0.0"
```

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `True` and a version like `2.13.0+rocm10.0.0`. Or skip extras entirely: `rocm examine` / `rocm install sdk`.

---

### 4. Windows 11 + pip

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\activate

uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ `
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" `
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" `
  "torchaudio==2.11.0.2+rocm10.0.0"
```

---

### 5. vLLM Docker (0.27.0)

```bash
docker pull rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0

docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --network=host --ipc=host --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v ~/models:/app/models -e HF_HOME="/app/models" \
  rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
  bash
```

> The image bundles PyTorch 2.12.0 + vLLM 0.27.0. The pip path above is PyTorch 2.13.0. Do not mix them.

---

### 6. Next tutorials

- [LM Studio](./lm-studio-rocm7-deploy.md)
- [Ollama](./ollama-rocm7-deploy.md)
- [llama.cpp](./llamacpp-rocm7-deploy.md)
- [vLLM](./vllm-rocm7-deploy.md)
