# Chapter 6: Practical LLM Training Workflow (AMD ROCm Edition)

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

</div>

This chapter is based on the Transformers + DeepSpeed framework and implements a full LLM pipeline on AMD ROCm, including pretraining and SFT fine-tuning. For a detailed framework introduction and training principles, see [chapter6-llm-training-workflow-practice.md](./chapter6-llm-training-workflow-practice.md).

For the Chinese version, see [第六章 大模型训练流程实践.md](./第六章%20大模型训练流程实践.md).

Section 6.4 (English): [chapter6-4-wip-preference-alignment.md](./chapter6-4-wip-preference-alignment.md)

---

## File Structure

```
chapter6/
├── 第六章 大模型训练流程实践.md    # Detailed chapter tutorial
├── chapter6-llm-training-workflow-practice.md  # English chapter tutorial
├── 6.4[WIP] 偏好对齐.md           # Section 6.4 (work in progress)
├── chapter6-4-wip-preference-alignment.md      # English section 6.4
├── README.md                      # Chinese README
├── README_EN.md                   # This file
└── code/
    ├── 00_download_model.py        # Step 0: Download base model (Qwen2.5-1.5B)
    ├── 01_download_dataset.py      # Step 1a: Download dataset
    ├── 01_process_dataset.ipynb    # Step 1b: Dataset processing (Notebook)
    ├── 02_pretrain.py              # Step 2: Pretraining script
    ├── 02_pretrain.sh              # Step 2: Pretraining launch script (DeepSpeed)
    ├── 02_pretrain.ipynb           # Step 2: Pretraining (Notebook)
    ├── 03_finetune.py              # Step 3: SFT fine-tuning script
    ├── 03_finetune.sh              # Step 3: SFT fine-tuning launch script (DeepSpeed)
    ├── ds_config_zero2.json        # DeepSpeed ZeRO-2 configuration
    ├── whole.ipynb                 # End-to-end pipeline Notebook
    └── requirements.txt
```

---

## Execution Workflow

### Step 0: Install Dependencies & Download Base Model

```bash
# Install ROCm and ROCm-specific torch torchvision torchaudio
# This test uses 4*AMD Radeon™ AI PRO R9700 with gfx1201 architecture.
# If your GPU uses another architecture, install the matching packages yourself.
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ "rocm[libraries,devel]"
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ torch torchvision torchaudio

# Install dependencies
pip install -r code/requirements.txt

# Download Qwen2.5-1.5B as the base model (accelerated via hf-mirror)
cd code
python 00_download_model.py
```

> This project was tested with 4*AMD Radeon™ AI PRO R9700. For compatibility across other Instinct / Radeon PRO / Radeon / Ryzen series devices, see https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html

The model is downloaded to `autodl-tmp/qwen-1.5b/`. To download other models, modify the `huggingface-cli download` target repository and the `--local-dir` parameter in the script.

---

### Step 1: Download & Process Dataset

```bash
# Download dataset
python 01_download_dataset.py
```

| Dataset | Purpose | Download Path |
|--------|------|----------|
| seq-monkey (ModelScope) | Pretraining corpus | `autodl-tmp/dataset/pretrain_data/` |
| BelleGroup/train_3.5M_CN | SFT instruction data | `autodl-tmp/dataset/sft_data/BelleGroup/` |

> Use `01_process_dataset.ipynb` for interactive data processing and analysis.

---

### Step 2: Pretraining (DeepSpeed ZeRO-2)

**Recommended: launch directly with the shell script**

```bash
bash 02_pretrain.sh
```

**Key parameter notes in `02_pretrain.sh`:**

| Parameter | Description | Default |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | Visible GPUs; equivalent to `HIP_VISIBLE_DEVICES` under ROCm | `0,1` |
| `--config_name` | Model architecture config directory (used for training from scratch) | `autodl-tmp/qwen-1.5b` |
| `--tokenizer_name` | Tokenizer directory | `autodl-tmp/qwen-1.5b` |
| `--train_files` | Pretraining data file path | `autodl-tmp/dataset/pretrain_data/...jsonl` |
| `--output_dir` | Training output directory (checkpoints) | `autodl-tmp/output/pretrain` |
| `--per_device_train_batch_size` | Batch size per GPU | `16` |
| `--gradient_accumulation_steps` | Gradient accumulation steps | `4` |
| `--num_train_epochs` | Number of training epochs | `1` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--block_size` | Maximum text length (number of tokens) | `2048` |
| `--bf16` | Enable bfloat16 precision (recommended on ROCm) | Enabled |
| `--deepspeed` | DeepSpeed configuration file | `./ds_config_zero2.json` |

> **ROCm note**: `--bf16` is natively supported on both AMD MI and RDNA architectures. DeepSpeed ZeRO-2 can significantly reduce per-GPU VRAM usage and enable larger batch sizes.

---

### Step 3: SFT Fine-Tuning (DeepSpeed ZeRO-2)

**Recommended: launch directly with the shell script**

```bash
bash 03_finetune.sh
```

**Key parameter notes in `03_finetune.sh`:**

| Parameter | Description | Default |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | Visible GPUs | `0,1` |
| `--model_name_or_path` | Pretrained model or checkpoint path | `autodl-tmp/qwen-1.5b` |
| `--train_files` | SFT data file path (JSON format) | `autodl-tmp/dataset/sft_data/BelleGroup/train_3.5M_CN.json` |
| `--output_dir` | Fine-tuning output directory | `autodl-tmp/output/sft` |
| `--per_device_train_batch_size` | Batch size per GPU | `16` |
| `--gradient_accumulation_steps` | Gradient accumulation steps | `4` |
| `--num_train_epochs` | Number of training epochs | `3` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--block_size` | Maximum text length | `2048` |
| `--bf16` | Enable bfloat16 precision | Enabled |
| `--deepspeed` | DeepSpeed configuration file | `./ds_config_zero2.json` |

**Resume from checkpoint (supported by both scripts):**

```bash
# Uncomment the last line in the script and replace with your checkpoint path
--resume_from_checkpoint autodl-tmp/output/pretrain/checkpoint-XXXXX
```

---

## DeepSpeed ZeRO-2 Configuration

`ds_config_zero2.json` is preconfigured for AMD ROCm. Core options:

| Option | Description |
|------|------|
| `zero_optimization.stage: 2` | Optimizer state + gradient sharding, balanced multi-GPU memory usage |
| `bf16.enabled: true` | ROCm bfloat16 acceleration |
| `gradient_clipping: 1.0` | Gradient clipping threshold |

To further reduce VRAM usage, you can set `stage` to `3` (full parameter sharding).

---

## Training Monitoring

Both training scripts integrate SwanLab for visualization:

```bash
# --report_to swanlab is already included in the .sh files
# Log in before first use
swanlab login
```
