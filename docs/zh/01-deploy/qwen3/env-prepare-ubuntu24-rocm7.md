## Ubuntu 24.04 / Windows 11 环境准备：ROCm 10.0.0 + PyTorch + vLLM（以 gfx1151 为例）

**ROCm 10.0.0 部署 Qwen3 推理框架环境准备指南。**

本节以 **Ryzen AI Max / Ryzen AI Max+（gfx1151）** 为参考。完整基线见 [00-Environment](/zh/00-environment/)，10.0.0 与 7.14.0 的差异见 [ROCm 10.0.0 版本说明](/zh/00-environment/rocm-10-0-0-release-notes)。

> 官方参考：
> - [ROCm 10.0.0 文档首页](https://rocm.docs.amd.com/en/latest/)
> - [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&w=compute&os=windows&windows-ver=11&i=pip&gpu=max-395&gfx=gfx1151)
> - [PyTorch 2.13.0 on ROCm 10.0.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.13.0&w=compute&gpu=max-395&gfx=gfx1151)
> - [vLLM 0.27.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)

---

### 一、清理旧环境

```bash
sudo apt remove 'rocm*' 'amdrocm*' 'amdgpu-dkms*' -y
sudo apt autoremove -y
```

Windows：卸载旧 HIP SDK，安装 [Adrenalin 26.8.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads)，关闭 WDAG / SAC。

---

### 二、Ubuntu 24.04 + gfx1151

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

### 三、Windows 11 + pip

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

### 四、vLLM Docker（0.27.0）

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

> 镜像内置 PyTorch 2.12.0；pip 路线是 2.13.0。不要混写。一条命令装环境也可以用 `rocm install sdk`。

---

### 五、后续部署教程

- [LM Studio 部署教程](./lm-studio-rocm7-deploy.md)
- [Ollama 部署教程](./ollama-rocm7-deploy.md)
- [llama.cpp 部署教程](./llamacpp-rocm7-deploy.md)
- [vLLM 部署教程](./vllm-rocm7-deploy.md)
