# Fine-tune

[中文](./README.md) | [Back to Home](../README_EN.md)

This chapter collects fine-tuning materials, a general tutorial, and model-specific experiment notes for running LLM fine-tuning on AMD GPU / ROCm environments. It walks through environment setup, dataset construction, LoRA fine-tuning, and LoRA weight merging.

## Model Examples

| Example | Description | Document |
|:---|:---|:---|
| Qwen3-0.6B-LoRA | LoRA fine-tuning with Qwen3-0.6B, including SwanLab training visualization | [View note](./models/Qwen3/01-Qwen3-0.6B-LoRA及SwanLab可视化记录.md) |
| Gemma4-E4B-LoRA | Fine-tuning Gemma 4 E4B-it on the `dair-ai/emotion` dataset | [View note](./models/Gemma4/01-Gemma4-E4B-LoRA及SwanLab可视化记录.md) |

Contributions are welcome. Feel free to add more model examples for edge deployment and AMD GPU fine-tuning experiments.

## General Fine-Tuning Tutorial

Prompt engineering is often effective for LLM applications, but fine-tuning becomes important when tasks require highly customized behavior, vertical-domain knowledge, or smaller models that can handle difficult downstream tasks. This tutorial introduces a full supervised fine-tuning (SFT) workflow with `transformers` and `peft`.

The example environment is based on AMD Ryzen AI Max+ 395. Although this machine is often used for LLM inference, it is also capable of small-scale fine-tuning experiments under resource constraints.

> Note: Apart from environment setup, this chapter focuses on the core concepts and workflow of fine-tuning. For runnable model-specific practice, refer to the examples under the `models` directory.

### Environment Setup

This section focuses on the Python virtual environment and training dependencies. For ROCm base installation, Windows/Ubuntu prerequisites, driver versions, Visual Studio requirements, and GPU architecture mapping, refer to [00-Environment](../00-Environment/README_EN.md).

For Windows + Ryzen AI Max+ 395, see [Windows 11 Installation](../00-Environment/README_EN.md#1-windows-11-installation) and [Install ROCm + PyTorch](../00-Environment/README_EN.md#15-install-rocm--pytorch). ROCm 7.12.0 and later support installing ROCm-related Python packages into a virtual environment via pip / uv pip.

![ROCm screenshot](./images/rocm.png)

The following commands use AMD Ryzen AI Max+ 395 (`gfx1151`) on Windows as the example. In this tutorial, `uv` is used as a replacement for conda/venv and as the pip installation entrypoint. Do not use `uv add` / `uv sync` to manage ROCm PyTorch.

#### Step 1: Create and Activate a Python Virtual Environment

Git Bash:

```bash
uv venv --python=3.13
source .venv/Scripts/activate
```

PowerShell:

```powershell
uv venv --python=3.13
.venv\Scripts\Activate.ps1
```

#### Step 2: Install ROCm Runtime and Libraries

```bash
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]"
```

After installation, use `hipinfo` to verify ROCm. If GPU information is displayed, ROCm is working.

```bash
--------------------------------------------------------------------------------
device#                           0
Name:                             AMD Radeon(TM) 8060S Graphics
pciBusID:                         244
pciDeviceID:                      0
pciDomainID:                      0
multiProcessorCount:              20
maxThreadsPerMultiProcessor:      2048
isMultiGpuBoard:                  0
clockRate:                        2900 Mhz
memoryClockRate:                  800 Mhz
memoryBusWidth:                   512
totalGlobalMem:                   107.87 GB
totalConstMem:                    2147483647

...

arch.has3dGrid:                   1
arch.hasDynamicParallelism:       0
gcnArchName:                      gfx1151
maxAvailableVgprsPerThread:       256 DWORDs
peers:
non-peers:                        device#0

memInfo.total:                    107.87 GB
memInfo.free:                     107.72 GB (100%)
```

#### Step 3: Install PyTorch

Install ROCm-enabled PyTorch from the AMD wheel index. See the [ROCm PyTorch installation guide](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/pytorch.html) for details.

```bash
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
```

If `torch.__version__` includes ROCm build information, `torch.version.hip` is not `None`, and `torch.cuda.is_available()` prints `True`, PyTorch is correctly configured for ROCm.

#### Step 4: Install Fine-Tuning Dependencies

```bash
uv pip install -r requirements.txt
```

If you need a mirror for regular PyPI packages:

```bash
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

#### Step 5: Download a Model

Replace `please-change-me` with the local path where you want to store the model files.

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    'Qwen/Qwen3-0.6B',
    cache_dir='please-change-me',
    revision='master',
)

print(f"Model downloaded to: {model_dir}")
```

### SFT Dataset Construction

Supervised fine-tuning (SFT) teaches a pretrained model how to follow instructions by training it on input-output pairs. A typical instruction dataset contains:

```json
{
  "instruction": "The user instruction",
  "input": "Optional extra input; empty if not needed",
  "output": "The expected model response"
}
```

For example:

```json
{
  "instruction": "Translate the following text into English:",
  "input": "今天天气真好",
  "output": "Today is a nice day!"
}
```

Different LLMs use different prompt formats. For Qwen3, inspect the chat template with:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('your model path', trust_remote=True)

messages = [
    {"role": "system", "content": "===system_message_test==="},
    {"role": "user", "content": "===user_message_test==="},
    {"role": "assistant", "content": "===assistant_message_test==="},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

print(text)
```

Example dataset loading:

```python
from datasets import Dataset
import pandas as pd

df = pd.read_json('./huanhuan.json')
ds = Dataset.from_pandas(df)
```

Example preprocessing function:

```python
def process_func(example):
    MAX_LENGTH = 1024
    instruction = tokenizer(
        f"<s><|im_start|>system\nNow you should role-play as Zhen Huan.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

### Parameter-Efficient Fine-Tuning (PEFT)

LoRA is one of the most common PEFT methods. It inserts low-rank trainable matrices into selected modules, so only a small number of parameters are updated during fine-tuning.

```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
```

Example training arguments:

```python
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()
```

### Merge LoRA Weights

LoRA fine-tuning saves adapter weights. For inference, load the base model and LoRA adapter together:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = 'base model path'
lora_path = 'LoRA checkpoint path'

tokenizer = AutoTokenizer.from_pretrained(mode_path)
model = AutoModelForCausalLM.from_pretrained(
    mode_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "Who are you?"
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=False,
)

inputs = inputs.to("cuda")

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
