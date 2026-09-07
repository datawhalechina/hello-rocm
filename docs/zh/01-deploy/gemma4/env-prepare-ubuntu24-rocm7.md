## Ubuntu 24.04 / Windows 11 环境准备：ROCm 10.0.0 + PyTorch + vLLM（以 gfx1151 为例）

**ROCm 10.0.0 部署 Gemma 4 推理框架环境准备指南。**

本节以 **Ryzen AI Max / Ryzen AI Max+（gfx1151）** 为参考，说明在 ROCm 10.0.0 / TheRock 体系下准备 Gemma 4 部署环境的关键步骤。完整基线与 ROCm.AI 说明见 [00-Environment](/zh/00-environment/) 与 [ROCm 10.0.0 版本说明](/zh/00-environment/rocm-10-0-0-release-notes)。

> 官方参考：
> - [ROCm 10.0.0 文档首页](https://rocm.docs.amd.com/en/latest/)
> - [ROCm 10.0.0 安装指南](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=ryzen&w=compute&os=windows&windows-ver=11&i=pip&gpu=max-395&gfx=gfx1151)
> - [PyTorch 2.13.0 on ROCm 10.0.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.13.0&w=compute&gpu=max-395&gfx=gfx1151)
> - [vLLM 0.27.0 on ROCm 10.0.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)
> - [TheRock transition guide](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html)

---

### 一、相对 7.14.0 / 7.13 的变化

| 项目 | ROCm 7.13 / 7.14.0 | ROCm 10.0.0 |
|:---|:---|:---|
| pip 索引 | `repo.amd.com/rocm/whl/` 或 `whl-multi-arch/` | `https://stable.repo.amd.com/rocm/whl-next/` |
| PyTorch | 2.11.0 / 2.12.0 | **2.13.0** |
| vLLM | 0.19.1 / 0.23.0 | **0.27.0** |
| Windows 驱动 | Adrenalin 26.5.1 | **Adrenalin 26.8.1** |
| 一条命令安装 | 无 | **ROCm CLI**：`rocm install sdk` |

核心路径 `/opt/rocm/core`、包名前缀 `amdrocm-*` 从 7.14.0 延续下来。10.0.0 新增 AMD Skills、Hyperloom、ROCm CLI，见版本说明。

---

### 二、清理已有的 ROCm / AMD 相关软件

```bash
sudo apt remove 'rocm*' 'amdrocm*' 'amdgpu-dkms*' -y
sudo apt autoremove -y
```

同时检查 `~/.bashrc`、`~/.zshrc`、`/etc/profile.d/` 里是否还指向旧的 `/opt/rocm`。Windows 请先卸载旧 HIP SDK，并安装 [Adrenalin 26.8.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html#Downloads)。

---

### 三、Ubuntu 24.04 + gfx1151 准备步骤

#### 2.1 安装 OEM kernel 6.14

gfx1151 在 Ubuntu 24.04 上需要 OEM kernel 6.14：

```bash
sudo apt update
sudo apt install -y linux-oem-24.04c
sudo reboot
```

#### 2.2 安装基础依赖

```bash
sudo apt update
sudo apt install -y \
  libatomic1 libquadmath0 \
  build-essential git curl wget jq pciutils
```

#### 2.3 配置 GPU 权限

```bash
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

#### 2.4 验证 GPU 设备

```bash
ls -l /dev/kfd /dev/dri
```

---

### 四、安装 PyTorch 2.13.0（ROCm 10.0.0 / gfx1151）

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

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

预期 `torch.cuda.is_available()` 为 `True`，版本类似 `2.13.0+rocm10.0.0`。

> 不想手填 extras 时：`curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh`，然后 `rocm examine` / `rocm install sdk`。

---

### 五、Windows 11 + pip 路线（ROCm 10.0.0）

开始前：卸载 HIP SDK、关闭 WDAG / SAC、安装 Adrenalin 26.8.1。

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\activate

uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ `
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" `
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" `
  "torchaudio==2.11.0.2+rocm10.0.0"

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

### 六、vLLM 环境验证（Docker 方式）

```bash
docker pull rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
```

> 镜像内置 PyTorch 2.12.0 + vLLM 0.27.0；上面的 pip 路线是 PyTorch 2.13.0。两条路线不要混写。

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
  rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
  bash
```

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'HIP:', torch.cuda.is_available())"
```

---

### 七、后续部署教程

- [LM Studio 部署教程](./lm-studio-rocm7-deploy.md)
- [Ollama 部署教程](./ollama-rocm7-deploy.md)
- [llama.cpp 部署教程](./llamacpp-rocm7-deploy.md)
- [vLLM 部署教程](./vllm-rocm7-deploy.md)
