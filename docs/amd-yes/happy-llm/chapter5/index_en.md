# Chapter 5: Hands-On LLM Building (AMD ROCm Version)

<div align='center'>

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)

</div>

In this chapter, we implement an 80M-parameter LLaMA2-architecture language model from scratch, and complete pretraining plus SFT fine-tuning on the AMD ROCm platform. For detailed principles and code walkthroughs, see [chapter5-hands-on-llm-building.md](./chapter5-hands-on-llm-building.md).

For the Chinese version, see [第五章 动手搭建大模型.md](./第五章%20动手搭建大模型.md).

---

## File Structure

```
chapter5/
├── 第五章 动手搭建大模型.md      # Detailed chapter tutorial
├── chapter5-hands-on-llm-building.md  # English chapter tutorial
├── README.md                    # Chinese README
├── README_EN.md                 # This file
└── code/
    ├── 00_download_dataset.sh          # Step 0: Download dataset (Linux)
    ├── 00_windows_download_dataset.sh  # Step 0: Download dataset (Windows)
    ├── 01_deal_dataset.py              # Step 1: Preprocess dataset
    ├── 02_train_tokenizer.py           # Step 2: Train BPE Tokenizer
    ├── 03_ddp_pretrain.py              # Step 3: DDP multi-GPU pretraining
    ├── 04_ddp_sft_full.py              # Step 4: DDP multi-GPU SFT fine-tuning
    ├── 05_model_sample.py              # Step 5: Inference test
    ├── 06_export_model.py              # Step 6: Export to HuggingFace format
    ├── k_model.py                      # Model definition (library file)
    ├── dataset.py                      # Dataset class (library file)
    ├── tokenizer_k/                    # Pretrained Tokenizer
    └── requirements.txt
```

---

## Execution Workflow

### Step 0: Install Dependencies & Download Dataset

```bash
# Install ROCm and ROCm versions of torch, torchvision, and torchaudio
# This test uses 4*AMD Radeon™ AI PRO R9700 with gfx1201 architecture.
# If your architecture differs, download the appropriate packages yourself.
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ "rocm[libraries,devel]"
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ torch torchvision torchaudio

# Install dependencies
pip install -r code/requirements.txt

# Download dataset (requires modelscope preinstalled)
cd code
bash 00_download_dataset.sh
```

> This project was tested on 4*AMD Radeon™ AI PRO R9700. For compatibility of other Instinct/Radeon PRO/Radeon/Ryzen series devices, see https://rocm.docs.amd.com/en/7.12.0-preview/compatibility/compatibility-matrix.html

The script downloads data into the `./datasets/` directory, including:
- SeqMonkey pretraining corpus (about 10B tokens)
- BelleGroup 3.5 million Chinese SFT samples

> Windows users should use `00_windows_download_dataset.sh` (supports both PowerShell and CMD)

---

### Step 1: Preprocess Dataset

```bash
python 01_deal_dataset.py
```

**Input paths (edit at the top of the script):**

| Variable | Description | Default |
|------|------|--------|
| `pretrain_data` | Path to the raw pretraining jsonl file | `./datasets/mobvoi_seq_monkey_general_open_corpus.jsonl` |
| `sft_data` | Path to the raw SFT json file | `./datasets/BelleGroup/train_3.5M_CN.json` |
| `output_pretrain_data` | Output path for processed pretraining data | `./tmp/seq_monkey_datawhale.jsonl` |
| `output_sft_data` | Output path for processed SFT data | `./tmp/BelleGroup_sft.jsonl` |

---

### Step 2: Train BPE Tokenizer

```bash
python 02_train_tokenizer.py
```

**Key parameters (edit in the `main()` function):**

| Parameter | Description | Default |
|------|------|--------|
| `data_path` | Path to jsonl data used to train the Tokenizer | `./tmp/seq_monkey_datawhale.jsonl` |
| `save_dir` | Tokenizer save directory | `./tokenizer_k/` |
| `vocab_size` | Vocabulary size | `6144` |

> The project already provides a pretrained Tokenizer (`tokenizer_k/`), so you can skip this step and start training directly.

---

### Step 3: Multi-GPU Pretraining (DDP)

```bash
# Use ROCm multi-GPU DDP training
python 03_ddp_pretrain.py --gpus 0,1,2,3 --data_path ./tmp/seq_monkey_datawhale.jsonl
```

**Required parameters:**

| Parameter | Description | Default |
|------|------|--------|
| `--gpus` | GPU IDs to use, separated by commas | `0,1,2,3` |
| `--data_path` | Pretraining data path (output from Step 1) | `./datasets/mobvoi_seq_monkey_general_open_corpus.jsonl` |

**Common optional parameters:**

| Parameter | Description | Default |
|------|------|--------|
| `--out_dir` | Output directory for model weights | `base_model_215M` |
| `--epochs` | Number of training epochs | `1` |
| `--batch_size` | Per-GPU batch size (reduce if VRAM is insufficient) | `64` |
| `--learning_rate` | Initial learning rate | `2e-4` |
| `--dtype` | Training precision, ROCm recommends `bfloat16` | `bfloat16` |
| `--accumulation_steps` | Gradient accumulation steps | `8` |
| `--use_swanlab` | Enable SwanLab training visualization | Disabled |

> **ROCm tip**: On the ROCm platform, the `cuda` device is automatically mapped to AMD GPUs, so no extra device-parameter changes are needed. If VRAM is insufficient, lowering `--batch_size` to 4-8 can reduce VRAM requirements to around 7GB.

---

### Step 4: Multi-GPU SFT Fine-Tuning (DDP)

```bash
python 04_ddp_sft_full.py --gpus 0,1,2,3 --data_path ./tmp/BelleGroup_sft.jsonl --out_dir sft_model_215M
```

**Required parameters:**

| Parameter | Description | Default |
|------|------|--------|
| `--gpus` | GPU IDs to use, separated by commas | `0,1,2,3` |
| `--data_path` | SFT data path (output from Step 1) | `./BelleGroup_sft.jsonl` |
| `--out_dir` | Output directory for fine-tuned model weights | `sft_model_215M` |

**Common optional parameters:**

| Parameter | Description | Default |
|------|------|--------|
| `--epochs` | Number of training epochs | `1` |
| `--batch_size` | Per-GPU batch size | `64` |
| `--learning_rate` | Initial learning rate | `2e-4` |
| `--dtype` | Training precision | `bfloat16` |
| `--accumulation_steps` | Gradient accumulation steps | `8` |
| `--use_swanlab` | Enable SwanLab training visualization | Disabled |

---

### Step 5: Inference Test

```bash
python 05_model_sample.py
```

By default, it loads `./base_model_215M/pretrain_1024_18_6144.pth` (pretrained) and `./sft_model_215M/` (SFT). You can switch by modifying the `TextGenerator` initialization parameters at the top of the script:

| Parameter | Description | Default |
|------|------|--------|
| `checkpoint` | Path to model `.pth` weights | `./base_model_215M/pretrain_1024_18_6144.pth` |
| `tokenizer_model_path` | Tokenizer path | `./tokenizer_k/` |
| `device` | Inference device | Auto-detect |
| `dtype` | Inference precision | `bfloat16` |

---

### Step 6: Export to HuggingFace Format (Optional)

```bash
python 06_export_model.py
```

Modify the parameters in the `export_model()` call at the bottom of the script:

| Parameter | Description |
|------|------|
| `tokenizer_path` | Tokenizer directory path |
| `model_ckpt_path` | Path to trained `.pth` weights |
| `save_directory` | Save directory for HuggingFace-format model |

After export, it can be loaded directly with `AutoModelForCausalLM.from_pretrained()`.

---

## Training Reference Data

Test environment: 4 x AMD Radeon™ AI PRO R9700 (ROCm 7.12)

| Stage | Data Scale | Hardware | Time Reference |
|------|--------|------|----------|
| Pretraining | Full SeqMonkey dataset | 8 x GPU | ~46 hours |
| SFT | BelleGroup 3.5 million samples | 8 x GPU | ~24 hours |
| Pretraining (batch=4, single GPU) | - | 1 x GPU (7GB VRAM) | ~533 hours |

---

## Pretrained Model Download

If your hardware is not sufficient for full training, you can directly download the model already trained by the author:

- 🤖 [ModelScope Model Download](https://www.modelscope.cn/collections/Happy-LLM-e98b91b10b684a)
- 🎮 [ModelScope Creative Space Demo](https://www.modelscope.cn/studios/kmno4zx/happy_llm_215M_sft)
