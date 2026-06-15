# Content Guide

This document defines directory, naming, formatting, and pre-submit conventions for tutorial Markdown files, example code, and related assets in **hello-rocm**. The current reference implementations are `docs/zh/01-deploy/qwen3/`, `docs/zh/01-deploy/gemma4/`, and their matching `docs/en/` directories. New content should stay close to these patterns so readers can move between models, languages, and projects with minimal friction.

For the Chinese version, see **[CONTENT_GUIDE.md](./CONTENT_GUIDE.md)**. For contribution workflow, see **[CONTRIBUTING_en.md](./CONTRIBUTING_en.md)**.

## 1. Directory and Naming Conventions

| Type | Convention | Example |
|:---|:---|:---|
| Chinese deployment tutorials | `docs/zh/01-deploy/<model>/`, using lowercase model directory names | `docs/zh/01-deploy/qwen3/`, `docs/zh/01-deploy/gemma4/` |
| English deployment tutorials | `docs/en/01-deploy/<model>/`, mirroring the Chinese directory name | `docs/en/01-deploy/qwen3/` |
| Chinese fine-tuning tutorials | `docs/zh/02-fine-tune/<model>/` | `docs/zh/02-fine-tune/qwen3/` |
| English fine-tuning tutorials | `docs/en/02-fine-tune/<model>/` | `docs/en/02-fine-tune/gemma4/` |
| Fine-tuning code / notebooks | `src/fine-tune/models/<model>/` | `src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb` |
| ROCm operator / infra examples | Docs under `docs/zh/03-infra/` or `docs/en/03-infra/`; code under `src/infra/` | `docs/zh/03-infra/custom-pytorch-operator.md` |
| AMD practice case docs | `docs/zh/05-amd-yes/<project>/` or single-page `docs/zh/05-amd-yes/<project>.md` | `docs/zh/05-amd-yes/hello-agents/`, `docs/zh/05-amd-yes/openclaw.md` |
| AMD practice case source | `src/amd-yes/<project>/` | `src/amd-yes/hello-agents/` |
| Images | Use existing chapter categories under `docs/public/images/`; reference them with paths relative to the current Markdown file | `docs/public/images/01-deploy/qwen3/image11.png`, referenced as `../../../public/images/01-deploy/qwen3/image11.png` |

Common deployment tutorial filenames should match existing Qwen3 / Gemma4 tutorials:

- `env-prepare-ubuntu24-rocm7.md`: environment preparation and verification.
- `lm-studio-rocm7-deploy.md`: LM Studio deployment.
- `vllm-rocm7-deploy.md`: vLLM deployment.
- `ollama-rocm7-deploy.md`: Ollama deployment.
- `llamacpp-rocm7-deploy.md`: llama.cpp deployment.

## 2. Document Structure

1. **Title and opening**
   - Follow the heading style already used in nearby files. Do not introduce extra structure for a single page.
   - The opening paragraph should state the **system** (for example Ubuntu 24.04), **ROCm major version** (for example ROCm 7+), and **topic** (framework + model example).
   - Reference example: `docs/zh/01-deploy/qwen3/lm-studio-rocm7-deploy.md`.

2. **Prerequisites**
   - State the environment preparation document or existing chapter the tutorial depends on, such as `env-prepare-ubuntu24-rocm7.md` in the same directory.
   - If the tutorial depends on external model weights, datasets, account permissions, or specific GPU models, mention that near the beginning.

3. **Sections**
   - Numbered Chinese sections, `### 1.`, or `#### 1.1` are all acceptable; keep one style within a single document.
   - Use `---` between large steps when it improves readability. Do not reorder unrelated sections just for style consistency.

4. **Commands and code**
   - Use ` ```bash ` fences for shell commands. Use plain text or comments for output-only blocks.
   - Use full `https://` links for official documentation and scripts.
   - When commands depend on versions, mention the applicable ROCm / PyTorch / framework version.

5. **Images**
   - Image paths must point to real files in the repository.
   - Tutorial images in this project are mostly centralized under `docs/public/images/`. Follow the existing chapter categories, for example `docs/public/images/01-deploy/qwen3/` or `docs/public/images/05-amd-yes/toy-cli/`.
   - Image references must use paths **relative to the current Markdown file** and point to real files under `docs/public/images/`; do not use repository-root paths. This keeps images rendering correctly in both GitHub Markdown preview and VitePress pages.
   - Example: if the document is `docs/zh/01-deploy/qwen3/vllm-rocm7-deploy.md` and the image is `docs/public/images/01-deploy/qwen3/image11.png`, use:

     ```markdown
     ![vLLM ROCm deployment screenshot](../../../public/images/01-deploy/qwen3/image11.png)
     ```

     If width control is needed, HTML is acceptable, but keep the path relative:

     ```html
     <img src="../../../public/images/01-deploy/qwen3/image11.png" alt="vLLM ROCm deployment screenshot" width="90%" />
     ```

   - Another common example: from `docs/zh/05-amd-yes/toy-cli.md` to `docs/public/images/05-amd-yes/toy-cli/toy_cli_rocm_agent_flow_zh.png`, write `../../public/images/05-amd-yes/toy-cli/toy_cli_rocm_agent_flow_zh.png`.
   - Do not use nonexistent middle directories such as `./images/media/`.

6. **Cross-references**
   - Same-directory files: use `./xxx.md`.
   - Cross-chapter links within the same language: use relative paths or the existing VitePress route style.
   - Root files: use relative links that work when clicked on GitHub.

## 3. README and Supported Models

- When adding a full model line or changing model presentation, check whether **[README.md](./README.md)** and **[README_en.md](./README_en.md)** need updates.
- When adding only one sub-tutorial, add it to the home page or chapter index only if it should be exposed there. Otherwise, ensure it is discoverable from the relevant chapter or directory.
- Multilingual README files live under `docs-readme/`. They normally link to the English contribution guide and English content guide so each translation does not maintain separate contribution rules.

## 4. Chinese / English Synchronization

- Chinese contribution guide: `CONTRIBUTING.md`
- English contribution guide: `CONTRIBUTING_en.md`
- Chinese content guide: `CONTENT_GUIDE.md`
- English content guide: `CONTENT_GUIDE_en.md`

The Chinese and English documents do not need to be word-for-word translations, but directory structure, path conventions, submission requirements, and checklists should stay aligned. If a PR updates only one language, explain whether the other language needs follow-up synchronization.

## 5. Language and Style

- Chinese pages should use Simplified Chinese by default. English pages should use clear, direct technical writing.
- Keep common technical terms as-is: ROCm, vLLM, Ollama, PyTorch, CUDA, LoRA, and similar names.
- Avoid ambiguity: state boundaries such as host/container, system install/virtual environment, and Windows/Ubuntu.
- Versions, commands, and official documentation change quickly. When editing, state the target ROCm or framework version when possible.

## 6. Pre-submit Checklist

- [ ] All relative links point to real files in the repository.
- [ ] All image paths point to real files or existing public image directories.
- [ ] New model lines or important entry points have been checked against README / README_en.
- [ ] Chinese and English versions have matching structure and key requirements, or the PR explains why they are not synchronized.
- [ ] No local-only absolute paths, account information, API keys, or personal data are included.

---

Maintainers may update this page as the repository evolves. For major convention changes, also update **[CONTRIBUTING_en.md](./CONTRIBUTING_en.md)**.
