## Ubuntu 24.04 Environment Setup: Installing ROCm 7.1.0

**Ubuntu 24.04 (Linux) Inference Framework Deployment Guide with ROCm 7+ Support — Environment Setup Section**

This section explains how to do the following on Ubuntu 24.04:

- Clean up existing ROCm / AMD related environments
- Install ROCm 7.1.0 using the official script
- Verify the installation using tool commands

---

### 1. Remove All ROCm Related Software from the Current Environment

```bash
sudo apt remove rocm*
sudo apt remove amd*
```

---

### 2. Run the Script to Install ROCm 7.1.0

> If using Docker, you need to install `amdgpu-dkms`: refer to  
> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html  
> The script below already includes the relevant steps; if you don't run the script, you need to install it manually.

Download the installation script to your local machine:

```bash
https://github.com/amdjiahangpan/rocm-install-script/blob/ROCm_7.1.0_ubuntu_24.04/2-install.sh
```

Update the system and run the installation script:

```bash
sudo apt update
# Update the kernel
sudo apt upgrade -y
sudo bash 2-install.sh
```

After installation, verify with the following commands (all should show GPU-related output):

```bash
rocminfo
rocm-smi
amd-smi
```

For more installation details, refer to the official quick start documentation:

https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html
