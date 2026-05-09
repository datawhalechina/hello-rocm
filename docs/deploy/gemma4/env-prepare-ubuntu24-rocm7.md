## Ubuntu 24.04 环境准备：安装 ROCm 7.12 + PyTorch + vLLM（以 Ryzen AI Max+ PRO 395 / gfx1151 为例）

**Ubuntu 24.04 (Linux) 部署支持 ROCm 7.12 的推理框架部署指南 — 环境准备部分**

本节以 **AMD Ryzen AI Max+ PRO 395（APU，gfx1151 架构）** 为参考，介绍在 Ubuntu 24.04 上完成以下步骤：

- 清理已有的 ROCm / AMD 相关环境
- 使用官方 `apt` 源安装 **ROCm 7.12.0**
- 基于 ROCm 7.12.0 安装 **PyTorch 2.9.1**
- 通过官方 **ROCm vLLM Docker 镜像**直接拉起推理服务

> 官方文档参考：
> - [Install AMD ROCm 7.12.0](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html?fam=ryzen&gpu=max-pro-395&os=ubuntu&os-version=24.04&i=pkgman)
> - [Install PyTorch on ROCm 7.12.0](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/pytorch.html?fam=ryzen&gpu=max-pro-395&os=linux&os-version=24.04&i=pkgman)
> - [vLLM inference](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/vllm.html?fam=ryzen&gpu=max-pro-395&i=docker&os=linux&os-version=24.04)

> 其他 GPU 架构（如 Instinct MI350X=gfx950、MI300X=gfx94X、RX 9070=gfx120X、RX 7900=gfx110X 等）可参考上述官方链接的硬件选择器，只需把 `gfx1151` 替换为对应架构名即可。

---

### 一、清理已有的 ROCm / AMD 相关软件

如果系统里已经装过旧版 ROCm，建议先清理，避免与 ROCm 7.12 冲突：

```bash
sudo apt remove 'rocm*' 'amdrocm*' 'amdgpu-dkms*' -y
sudo apt autoremove -y
```

---

### 二、准备系统环境（Ryzen APU 专用）

#### 2.1 安装 OEM 内核（6.14）

Ryzen AI APU 在 Ubuntu 24.04 上需要使用 OEM 内核 6.14，才能正确驱动 iGPU：

```bash
sudo apt update
sudo apt install -y linux-image-6.14.0-1018-oem
# 装完请重启
sudo reboot
```

#### 2.2 安装 ROCm 依赖与 Python

```bash
sudo apt update
sudo apt install -y libatomic1 libquadmath0
sudo apt install -y python3.12 python3.12-venv
```

#### 2.3 配置 GPU 访问权限

将当前用户加入 `render` 和 `video` 组（重启或重新登录后生效）：

```bash
sudo usermod -a -G render,video "$LOGNAME"
```

（可选）使用 udev 规则授予系统级 GPU 访问权限：

```bash
sudo tee /etc/udev/rules.d/70-amdgpu.rules <<'EOF'
KERNEL=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="render", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

### 三、安装 ROCm 7.12.0（apt / pkgman 方式）

> 说明：Ryzen AI APU 使用 Ubuntu 24.04 自带的 inbox kernel 驱动，无需单独安装 `amdgpu-dkms`；Instinct / Radeon 独立 GPU 请按官方文档额外安装 `amdgpu` 驱动。

#### 3.1 注册 ROCm apt 仓库

```bash
# 下载并写入 GPG key
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.amd.com/rocm/packages/gpg/rocm.gpg -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

# 注册 Ubuntu 24.04 的 ROCm 7.12 源
sudo tee /etc/apt/sources.list.d/rocm.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages/ubuntu2404 stable main
EOF

sudo apt update
```

#### 3.2 安装 ROCm 7.12 核心包（gfx1151）

```bash
sudo apt install -y amdrocm7.12-gfx1151
```

> 如果你使用其它架构，请替换为对应 meta 包，例如：
> - Instinct MI350X：`amdrocm7.12-gfx950`
> - Instinct MI300X / MI325X：`amdrocm7.12-gfx94x`
> - Radeon RX 9000 系列：`amdrocm7.12-gfx120x`
> - Radeon RX 7000 系列：`amdrocm7.12-gfx110x`
> - Ryzen AI 300 / Strix Halo（gfx1150）：`amdrocm7.12-gfx1150`

#### 3.3 配置环境变量

以"当前用户"的方式配置（推荐）：

```bash
tee --append ~/.bashrc <<'EOF'

# BEGIN ROCm environment configuration
export LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib
# END ROCm environment configuration
EOF

source ~/.bashrc
```

若需系统级配置：

```bash
sudo tee /etc/profile.d/set-rocm-env.sh <<'EOF'
export LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib
EOF
sudo chmod +x /etc/profile.d/set-rocm-env.sh
source /etc/profile.d/set-rocm-env.sh
```

#### 3.4 验证安装

依次执行以下命令，应能看到 GPU 信息与 ROCm 版本：

```bash
rocminfo
amd-smi version
amd-smi monitor
```

`amd-smi version` 输出示例：

```
AMDSMI Tool: 26.3.0+2bd1678d3d | AMDSMI Library version: 26.3.0 | ROCm version: 7.12.0 | amdgpu version: 6.16.13 | ...
```

`rocminfo` 中应能看到 `AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S` 这样的条目。

---

### 四、安装 PyTorch（ROCm 7.12 版）

官方推荐使用 Python 虚拟环境 + pip 安装针对 `gfx1151` 的 ROCm PyTorch。

#### 4.1 创建并激活虚拟环境

```bash
python3.12 -m venv ~/rocm-venv
source ~/rocm-venv/bin/activate
```

#### 4.2 安装 ROCm 版 PyTorch

```bash
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  torch torchvision torchaudio
```

> 如需与 vLLM 0.16 保持版本一致，可使用：
> ```bash
> python -m pip install \
>   --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
>   "torch==2.9.1+rocm7.12.0" \
>   "torchaudio==2.9.0+rocm7.12.0" \
>   "torchvision==0.24.0+rocm7.12.0"
> ```

#### 4.3 验证 PyTorch 是否能用 ROCm

```bash
python -c "import torch; print(torch.__version__); print('HIP available:', torch.cuda.is_available())"
```

预期输出 `True` 则代表 PyTorch + ROCm 安装成功。

---

### 五、安装 vLLM（Docker 方式，推荐）

vLLM 官方已提供针对 `gfx1151` 的 ROCm 7.12 Docker 镜像，开箱即用，避免手动编译 Triton / FlashAttention 的复杂度。

> 前置条件：系统已安装 Docker。
> 可参考：<https://docs.docker.com/engine/install/ubuntu/>

#### 5.1 拉取 ROCm vLLM 镜像（gfx1151）

```bash
docker pull rocm/vllm:rocm7.12.0_gfx1151_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0
```

> 其它架构的镜像命名规则一致，把 `gfx1151` 替换即可，例如：
> - `rocm/vllm:rocm7.12.0_gfx950-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0`（MI350X）
> - `rocm/vllm:rocm7.12.0_gfx94X-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0`（MI300X / MI325X）
> - `rocm/vllm:rocm7.12.0_gfx120X-all_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0`（RX 9000 系列）

#### 5.2 启动容器

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

> `HF_TOKEN` 需替换为你在 [Hugging Face](https://huggingface.co/settings/tokens) 生成的 token（Gemma / Llama 等受限模型必须先在模型页面点击 **Agree & Access**）。

#### 5.3 容器内已知问题的临时解决办法

ROCm 7.12 Docker 镜像中 vLLM 启动时可能因为路径解析问题失败，按官方 Release Notes 建议，在启动 vLLM 之前追加以下环境变量：

```bash
export LD_LIBRARY_PATH=/opt/python/lib/python3.12/site-packages/_rocm_sdk_core/lib:$LD_LIBRARY_PATH
```

#### 5.4 在容器内验证 vLLM

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'HIP:', torch.cuda.is_available())"
```

#### 5.5 启动一个 Gemma 4 E4B 推理服务（可选）

```bash
vllm serve google/gemma-4-E4B-it \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 32
```

服务就绪后可用 `curl http://127.0.0.1:8000/v1/models` 检查模型列表，接下来的性能测试脚本请参考 [vLLM 部署教程](./vllm-rocm7-deploy.md)。

---

### 六、常见问题

<details>
<summary>Q1: 安装完 ROCm 后 `rocminfo` 没有列出 GPU？</summary>

1. 确认已执行 OEM 内核 6.14 的安装并重启系统；
2. 确认当前用户在 `render` / `video` 组里，`groups` 命令里能看到；
3. 重新登录一次当前用户（或重启），让组权限生效。

</details>

<details>
<summary>Q2: Docker 里 <code>/dev/kfd</code> 不存在？</summary>

1. 宿主机先执行 `ls /dev/kfd /dev/dri` 确认设备存在；
2. 确认 Docker 服务已启动，并检查 `docker info` 输出中 `Runtimes` 能找到默认 runtime；
3. Ryzen APU 使用 inbox driver 时，`/dev/kfd` 只有在 OEM 内核下才会出现。

</details>

<details>
<summary>Q3: 拉取 ROCm vLLM 镜像很慢？</summary>

可以考虑为 Docker 配置国内镜像加速（`/etc/docker/daemon.json`），或者使用 `docker pull` 时挂代理；  
也可以先 `docker save` 在有网环境下缓存好再 `docker load` 到目标机器。

</details>

---

完成上述步骤后，即可继续按模型进入：

- [LM Studio 部署教程](./lm-studio-rocm7-deploy.md)
- [Ollama 部署教程](./ollama-rocm7-deploy.md)
- [llama.cpp 部署教程](./llamacpp-rocm7-deploy.md)
- [vLLM 部署教程](./vllm-rocm7-deploy.md)
