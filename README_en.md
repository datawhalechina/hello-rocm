<div align=center>
  <h1>hello-rocm</h1>
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*Open Source · Community Driven · Making AMD AI Ecosystem More Accessible*

[中文](./README.md) | English

</div>


&emsp;&emsp;Since **ROCm 7.10.0** (released on December 11, 2025), ROCm now supports seamless installation in Python virtual environments just like CUDA, and officially supports both **Linux and Windows** platforms. This marks a major breakthrough for AMD in the AI field — learners and LLM enthusiasts are no longer limited to NVIDIA for hardware choices, and AMD GPUs are becoming a strong competitive alternative.

&emsp;&emsp;Dr. Lisa Su announced at the launch event that ROCm will maintain a **new release every 6 weeks** iteration pace, fully pivoting towards AI. The future looks exciting!

&emsp;&emsp;However, there is currently a global lack of systematic learning tutorials for ROCm LLM inference, deployment, training, fine-tuning, and infrastructure. **hello-rocm** was created to fill this gap.

&emsp;&emsp;**The main content of this project is tutorials, helping more students and future practitioners understand and become familiar with AMD ROCm! Anyone can raise issues or submit PRs to help build and maintain this project together.**

> &emsp;&emsp;***Learning Suggestion: We recommend starting with environment configuration and deployment, then moving on to model fine-tuning, and finally exploring Infra operator optimization. Beginners can start with LM Studio or vLLM deployment.***

## Project Significance

&emsp;&emsp;What is ROCm?

> ROCm (Radeon Open Compute) is an open-source GPU computing platform launched by AMD, designed to provide an open software stack for high-performance computing and machine learning. It supports parallel computing on AMD GPUs and serves as an alternative to CUDA on the AMD platform.

&emsp;&emsp;The battle of LLMs is in full swing, with open-source LLMs emerging one after another. However, most current LLM tutorials and development tools are based on the NVIDIA CUDA ecosystem. For developers who want to use AMD GPUs, the lack of systematic learning resources is a pain point.

&emsp;&emsp;Starting from ROCm 7.10.0 (December 11, 2025), AMD restructured the underlying ROCm architecture through TheRock project, decoupling the compute runtime from the operating system. This allows the same ROCm upper-level interfaces to run on both Linux and Windows, and supports direct installation into Python virtual environments just like CUDA. This means ROCm is no longer just an "engineering tool" for Linux, but has evolved into a truly cross-platform GPU computing platform for AI learners and developers — whether using Windows or Linux, users can now use AMD GPUs for training and inference with a lower barrier to entry. LLM and AI enthusiasts are no longer bound to the single NVIDIA ecosystem for hardware choices. AMD GPUs are gradually becoming an AI computing platform that can be genuinely used by ordinary users.

&emsp;&emsp;This project aims to provide complete tutorials for LLM deployment, fine-tuning, and training on the AMD ROCm platform based on the experience of core contributors. We hope to gather co-creators to enrich the AMD AI ecosystem together.

&emsp;&emsp;***We hope to become a bridge between AMD GPUs and the general public, embracing a broader AI world with the spirit of freedom and equality in open source.***

## Target Audience

&emsp;&emsp;This project is suitable for the following learners:

* Want to use AMD GPUs for LLM development but can't find systematic tutorials;
* Hope to deploy and run LLMs at low cost with high value;
* Interested in the ROCm ecosystem and want to get hands-on experience;
* AI learners who want to expand knowledge beyond NVIDIA GPU platforms;
* Want to build domain-specific private LLMs on the AMD platform;
* And the broadest, most ordinary student community.

## Project Roadmap and Progress

&emsp;&emsp;This project is organized around the full workflow of ROCm LLM applications, including environment configuration, deployment, fine-tuning, and operator optimization:

### Latest News

- *2025.12.11:* [*ROCm 7.10.0 Release Notes*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### Project Structure

```
hello-rocm/
├── 01-Deploy/              # ROCm LLM Deployment Practice
├── 02-Fine-tune/           # ROCm LLM Fine-tuning Practice
├── 03-Infra/               # ROCm Operator Optimization Practice
├── 04-References/          # ROCm Quality Reference Materials
└── 05-AMD-YES/             # AMD Project Case Collection
```

### 01. Deploy - ROCm LLM Deployment

<p align="center">
  <strong>🚀 ROCm LLM Deployment Practice</strong><br>
  <em>Quick start guide for LLM deployment on AMD GPUs from scratch</em><br>
  📖 <strong><a href="./01-Deploy/README.md">Getting Started with ROCm Deploy</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • LM Studio LLM Deployment from Scratch<br>
      • vLLM LLM Deployment from Scratch<br>
    </td>
    <td valign="top" width="50%">
      • SGLang LLM Deployment from Scratch<br>
      • ATOM LLM Deployment from Scratch
    </td>
  </tr>
</table>

### 02. Fine-tune - ROCm LLM Fine-tuning

<p align="center">
  <strong>🔧 ROCm LLM Fine-tuning Practice</strong><br>
  <em>Efficient model fine-tuning on AMD GPUs</em><br>
  📖 <strong><a href="./02-Fine-tune/README.md">Getting Started with ROCm Fine-tune</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • LLM Fine-tuning Tutorial from Scratch<br>
      • Single-machine LLM Fine-tuning Scripts<br>
    </td>
    <td valign="top" width="50%">
      • Multi-node Multi-GPU Fine-tuning Tutorial
    </td>
  </tr>
</table>

### 03. Infra - ROCm Operator Optimization

<p align="center">
  <strong>⚙️ ROCm Operator Optimization Practice</strong><br>
  <em>Migration and optimization guide from CUDA to ROCm</em><br>
  📖 <strong><a href="./03-Infra/README.md">Getting Started with ROCm Infra</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • HIPify Automated Migration in Practice<br>
      • Seamless Switching of BLAS and DNN<br>
    </td>
    <td valign="top" width="50%">
      • Migration from NCCL to RCCL<br>
      • Mapping from Nsight to Rocprof
    </td>
  </tr>
</table>

### 04. References - ROCm Quality Reference Materials

<p align="center">
  <strong>📚 ROCm Quality Reference Materials</strong><br>
  <em>Curated AMD official and community resources</em><br>
  📖 <strong><a href="./04-References/README.md">ROCm References</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • <a href="https://rocm.docs.amd.com/">ROCm Official Documentation</a><br>
      • <a href="https://github.com/amd">AMD GitHub</a><br>
    </td>
    <td valign="top" width="50%">
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCm Release Notes</a><br>
      • Related News
    </td>
  </tr>
</table>

### 05. AMD-YES - AMD Project Case Collection

<p align="center">
  <strong>✨ AMD Project Case Collection</strong><br>
  <em>Community-driven AMD GPU project practices</em><br>
  📖 <strong><a href="./05-AMD-YES/README.md">Getting Started with ROCm AMD-YES</a></strong>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • AMchat - Advanced Mathematics<br>
      • Chat-Huanhuan<br>
      • Tianji<br>
    </td>
    <td valign="top" width="50%">
      • Digital Life<br>
      • happy-llm
    </td>
  </tr>
</table>

## Contributing

&emsp;&emsp;We welcome all forms of contributions! Whether it's:

* Improving or adding new tutorials
* Fixing errors and bugs
* Sharing your AMD projects
* Providing suggestions and ideas

&emsp;&emsp;Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

&emsp;&emsp;If you want to participate deeply, please contact us, and we will add you to the project maintainers.

## Acknowledgments

### Core Contributors

- [Zhixue Song (No Green Onion Ginger Garlic) - Project Lead](https://github.com/KMnO4-zx) (Datawhale Member)
- [Yu Chen - Project Lead](https://github.com/lucachen) (Content Creator - Google Developer Expert in Machine Learning)

> Note: More contributors are welcome to join!

### Others

- If you have any ideas, please contact us. Issues are also very welcome!
- Special thanks to the following contributors who have contributed to the tutorials!
- Thanks to the AMD University Program for supporting this project!!

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>

## License

[MIT License](./LICENSE)

---

<div align="center">

**Let's build the future of AMD AI together!** 💪

Made with ❤️ by the hello-rocm community

</div>
