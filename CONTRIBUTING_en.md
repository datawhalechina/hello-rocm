# Contributing Guide

Thank you for contributing to **hello-rocm**. This guide explains how to submit issues and pull requests efficiently while keeping new content consistent with existing tutorials such as Qwen3 and Gemma4.

For the Chinese version, see **[CONTRIBUTING.md](./CONTRIBUTING.md)**. For detailed content, naming, image, and link conventions, see **[CONTENT_GUIDE_en.md](./CONTENT_GUIDE_en.md)**.

## Project Structure

hello-rocm organizes documentation, source examples, and multilingual README files together. Before contributing, first decide where the content should live:

```text
hello-rocm/
├── docs/                   # VitePress documentation source
│   ├── zh/                 # Chinese docs
│   │   ├── 00-environment/ # ROCm baseline environment setup
│   │   ├── 01-deploy/      # LLM deployment on ROCm
│   │   ├── 02-fine-tune/   # LLM fine-tuning on ROCm
│   │   ├── 03-infra/       # ROCm operators and infrastructure practice
│   │   ├── 04-references/  # Curated ROCm references
│   │   └── 05-amd-yes/     # AMD practice cases and community projects
│   └── en/                 # English docs, mirroring the Chinese structure
├── src/                    # Example code, notebooks, skills, and project sources
├── docs-readme/            # Multilingual README files
├── scripts/                # Project scripts
├── CONTENT_GUIDE.md        # Chinese content guide
├── CONTENT_GUIDE_en.md     # English content guide
├── CONTRIBUTING.md         # Chinese contribution guide
└── CONTRIBUTING_en.md      # English contribution guide
```

## What Contributions Are Welcome

- **Tutorials**: deployment, fine-tuning, environment setup, troubleshooting, and version updates for ROCm, frameworks, and models.
- **Fixes**: typos, broken links, outdated commands, missing images, or unclear descriptions.
- **Structure**: directory or cross-reference improvements that make the project easier to navigate.
- **Discussion**: issues that report environment details, hardware models, and reproducible steps.

For detailed content and formatting requirements, see **[CONTENT_GUIDE_en.md](./CONTENT_GUIDE_en.md)**.

## Before You Start

1. Read **[CONTENT_GUIDE_en.md](./CONTENT_GUIDE_en.md)** to avoid inconsistent paths, naming, images, and references.
2. For large changes or new model families, open an **Issue** first to align directory structure and README presentation with maintainers.

## Where to Put New Content

| Contribution type | Recommended location |
|:---|:---|
| ROCm / PyTorch / baseline environment notes | `docs/zh/00-environment/`, with English sync under `docs/en/00-environment/` |
| Model deployment tutorials | `docs/zh/01-deploy/<model>/`, with English sync under `docs/en/01-deploy/<model>/` |
| Model fine-tuning tutorials | `docs/zh/02-fine-tune/<model>/`, with English sync under `docs/en/02-fine-tune/<model>/` |
| Fine-tuning notebooks / training scripts | `src/fine-tune/models/<model>/` |
| ROCm operators / HIP / PyTorch extensions | Docs under `docs/zh/03-infra/` or `docs/en/03-infra/`; code under `src/infra/` |
| Complete applications / community projects | Docs under `docs/zh/05-amd-yes/<project>/` or `docs/en/05-amd-yes/<project>/`; source code under `src/amd-yes/<project>/` |
| References / external links | `docs/zh/04-references/` and `docs/en/04-references/` |

## Submitting Issues

- **Bug or documentation error**: include the page path, expected behavior, actual behavior, and full error output if commands are involved. Include OS, ROCm version, and GPU model when relevant.
- **Content suggestion**: describe the model, framework, or chapter you want to add. Include upstream references when available.

## Submitting Pull Requests

1. **Fork** the repository and create a feature branch from the latest default branch, for example `fix/qwen3-image-path` or `feat/model-xxx-deploy`.
2. Keep each PR focused on one type of change, such as fixing links or adding one tutorial.
3. In the PR description, include:
   - **Motivation**: what problem it solves or what capability it adds;
   - **Summary**: which files changed;
   - **Validation**: whether you ran the documented steps locally.
4. If you add or adjust **model-specific tutorials**, check whether the supported-model section in **[README.md](./README.md)** and **[README_en.md](./README_en.md)** needs updates.

## Recommended Structure for Model-specific Tutorials

Follow **[`docs/zh/01-deploy/qwen3/`](./docs/zh/01-deploy/qwen3/)** and **[`docs/zh/01-deploy/gemma4/`](./docs/zh/01-deploy/gemma4/)** for deployment tutorial organization:

```text
docs/zh/01-deploy/<model>/
├── env-prepare-ubuntu24-rocm7.md   # Environment preparation and ROCm verification
├── <model>_model.md                # Model introduction, if applicable
├── lm-studio-rocm7-deploy.md
├── vllm-rocm7-deploy.md
├── ollama-rocm7-deploy.md
├── llamacpp-rocm7-deploy.md
└── images/
    ├── image1.png
    ├── image2.png
    └── ...
```

- **Fine-tuning documents** usually go under `docs/zh/02-fine-tune/<model>/`.
- **Fine-tuning code or notebooks** usually go under `src/fine-tune/models/<model>/`.
- When a tutorial references environment setup, prefer a relative link to `env-prepare-ubuntu24-rocm7.md` in the same model directory.

## Code of Conduct

Please keep communication friendly, respectful, and constructive. We value reproducible and verifiable technical writing.

## License

Content contributed to this repository follows the root **[LICENSE](./LICENSE)** file (MIT) by default. If you introduce third-party resources, describe their source and license in the PR.

---

Thank you again for your time and expertise.
