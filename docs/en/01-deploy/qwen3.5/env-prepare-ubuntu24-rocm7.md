## Ubuntu 24.04 / Windows 11 Environment Preparation: ROCm 10.0.0 + PyTorch + vLLM (gfx1151)

**ROCm 10.0.0 environment guide for deploying Qwen3.5 inference frameworks.**

This section uses **Ryzen AI Max / Ryzen AI Max+ (gfx1151)** as the reference. Qwen3.5 is a newer architecture — confirm that `vLLM` / `transformers` recognize the `qwen3_5` model type. Full baseline: [00-Environment](/00-environment/). Version delta: [ROCm 10.0.0 release notes](/00-environment/rocm-10-0-0-release-notes).

> Official references:
> - [ROCm 10.0.0 docs home](https://rocm.docs.amd.com/en/latest/)
> - [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&w=compute&os=windows&windows-ver=11&i=pip&gpu=max-395&gfx=gfx1151)
> - [PyTorch 2.13.0 on ROCm 10.0.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.13.0&w=compute&gpu=max-395&gfx=gfx1151)
> - [vLLM 0.27.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)

---

### 1. Clean existing software

```bash
sudo apt remove 'rocm*' 'amdrocm*' 'amdgpu-dkms*' -y
sudo apt autoremove -y
```

Windows: uninstall the old HIP SDK and install [Adrenalin 26.8.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads).

---

### 2. Ubuntu 24.04 + gfx1151

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

---

### 3. Windows 11 + pip

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

### 4. vLLM Docker (0.27.0)

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

---

### 5. Qwen3.5 checklist

1. Use vLLM 0.27.0 (official image or matching wheel).
2. Confirm `transformers` recognizes `qwen3_5`.
3. Keep `--max-model-len` within VRAM.
4. Set `enable_thinking` in API requests when needed.

---

### 6. Next tutorials

- [LM Studio](./lm-studio-rocm7-deploy.md)
- [Ollama](./ollama-rocm7-deploy.md)
- [llama.cpp](./llamacpp-rocm7-deploy.md)
- [vLLM](./vllm-rocm7-deploy.md)
