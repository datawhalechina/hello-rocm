<div align=center>
  <h1>04-References</h1>
  <strong>📚 Curated ROCm Resources</strong>
</div>

<div align="center">

*Selected official and community resources for AMD ROCm*

[Back to Home](/)

</div>

## Introduction

&emsp;&emsp;This section collects high-quality learning resources related to ROCm and AMD GPUs, including official documentation, community tutorials, technical blogs, and relevant news. Use it to quickly find the references you need.

## Official Resources

### AMD Official Documentation

| Resource | Description | Link |
|----------|-------------|------|
| ROCm Documentation | Official ROCm platform docs | [rocm.docs.amd.com](https://rocm.docs.amd.com/) |
| ROCm Release Notes | Release notes for each version | [Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) |
| HIP Programming Guide | HIP API and programming guide | [HIP Docs](https://rocm.docs.amd.com/projects/HIP/en/latest/) |
| AMD GitHub | AMD open-source repositories | [github.com/amd](https://github.com/amd) |
| ROCm GitHub | ROCm project repositories | [github.com/ROCm](https://github.com/ROCm) |

### Library Documentation

| Library | Purpose | Docs |
|---------|---------|------|
| rocBLAS | Basic linear algebra | [rocBLAS Docs](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/) |
| MIOpen | Deep learning primitives | [MIOpen Docs](https://rocm.docs.amd.com/projects/MIOpen/en/latest/) |
| RCCL | Collective communication | [RCCL Docs](https://rocm.docs.amd.com/projects/rccl/en/latest/) |
| rocFFT | Fast Fourier transforms | [rocFFT Docs](https://rocm.docs.amd.com/projects/rocFFT/en/latest/) |
| rocSPARSE | Sparse matrix operations | [rocSPARSE Docs](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/) |

## Community Resources

### Tutorials & Blogs

- [AMD ROCm Blog](https://rocm.blogs.amd.com/) - Official AMD technical blog
- [AMD Developer](https://developer.amd.com/) - AMD developer resource center
- [Datawhale](https://github.com/datawhalechina) - Open-source learning community

### Video Tutorials

> Coming soon...

### Forums & Communities

| Platform | Description | Link |
|----------|-------------|------|
| AMD Community | Official AMD community forum | [community.amd.com](https://community.amd.com/) |
| GitHub Discussions | ROCm project discussions | [ROCm Discussions](https://github.com/ROCm/ROCm/discussions) |
| Reddit r/Amd | AMD-related discussions | [r/Amd](https://www.reddit.com/r/Amd/) |

## News

### 2025

- **2025.12.11** - [ROCm 7.10.0 Released](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)
  - Windows platform support
  - pip install into Python virtual environments
  - TheRock project restructured underlying architecture

> More news coming soon...

## Hardware Support

### Supported GPU List

#### Consumer GPUs

| Series | Models | ROCm Support |
|--------|--------|--------------|
| RX 7000 | RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT | ✅ |
| RX 6000 | RX 6900 XT, RX 6800 XT, RX 6800, RX 6700 XT | ✅ |

#### Professional / Data Center GPUs

| Series | Models | ROCm Support |
|--------|--------|--------------|
| MI300 | MI300X, MI300A | ✅ |
| MI250 | MI250X, MI250 | ✅ |
| MI200 | MI210 | ✅ |
| MI100 | MI100 | ✅ |

> For the full support list, see [ROCm System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

## Common Tools

### Development Tools

| Tool | Purpose | Install Command |
|------|---------|-----------------|
| hipcc | HIP compiler | `sudo apt install hip-dev` |
| rocprof | Performance profiler | `sudo apt install rocprofiler` |
| rocgdb | GPU debugger | `sudo apt install rocgdb` |
| hipify-clang | CUDA-to-HIP converter | `sudo apt install hipify-clang` |

### AI Frameworks

| Framework | ROCm Support | Installation |
|-----------|--------------|--------------|
| PyTorch | ✅ | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| TensorFlow | ✅ | See official docs |
| JAX | ✅ | See official docs |
| ONNX Runtime | ✅ | See official docs |

## Recommended Books

> Coming soon...

## Contributing Resources

&emsp;&emsp;If you have quality ROCm-related resources to share, feel free to submit a PR or Issue!

### Submission Requirements

- Links must be valid and content must be high-quality
- Provide a short description of the resource
- Organize according to existing categories

---

<div align="center">

**Contributions welcome!** 🎉

[Open an Issue](https://github.com/datawhalechina/hello-rocm/issues) | [Submit a PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>