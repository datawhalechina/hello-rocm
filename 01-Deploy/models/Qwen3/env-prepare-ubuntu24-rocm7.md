## Ubuntu 24.04 环境准备：安装 ROCm 7.1.0

**Ubuntu24.04(Linux) 部署支持 ROCm 7+ 的推理框架部署指南 — 环境准备部分**

本节介绍如何在 Ubuntu 24.04 上：

- 清理已有的 ROCm / AMD 相关环境
- 使用官方脚本安装 ROCm 7.1.0
- 通过工具命令验证安装是否成功

---

### 一、删除目前环境中所有 ROCm 相关软件

```bash
sudo apt remove rocm*
sudo apt remove amd*
```

---

### 二、运行脚本安装 ROCm 7.1.0 到系统中

> 如果使用 Docker，需要安装 `amdgpu-dkms`：参考  
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html  
> 下方脚本已包含相关步骤；如果不执行脚本安装，需要自行手动安装。

下载安装脚本到本地：

```bash
https://github.com/amdjiahangpan/rocm-install-script/blob/ROCm_7.1.0_ubuntu_24.04/2-install.sh
```

更新系统并执行安装脚本：

```bash
sudo apt update
# 更新内核
sudo apt upgrade -y
sudo bash 2-install.sh
```

安装完成后，使用以下命令验证（均应有显卡相关输出）：

```bash
rocminfo
rocm-smi
amd-smi
```

更多安装细节可参考官方快速上手文档：

https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html


