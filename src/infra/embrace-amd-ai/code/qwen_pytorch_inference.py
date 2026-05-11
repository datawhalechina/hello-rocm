# file: src/infra/embrace-amd-ai/code/qwen_pytorch_inference.py
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径
MODEL_PATH = "./Qwen/Qwen2___5-7B-Instruct"
DEVICE = "cuda:0"


def run_inference():
    print(f"=== AMD ROCm PyTorch 推理测试 ===")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"使用设备: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("[警告] 未检测到 ROCm/CUDA 设备，将使用 CPU 运行（极慢）")

    print("\n[1/3] 正在加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        print(f"[错误] Tokenizer 加载失败: {e}")
        return

    print("\n[2/3] 正在加载模型权重 (BFloat16)...")
    st = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[致命错误] 模型加载失败: {e}")
        print("如果是显存不足，请尝试使用量化模型。")
        return

    print(f"模型加载耗时: {time.time() - st:.2f} 秒")

    prompt = "你好，请用这台高性能显卡为我写一首关于 AMD 显卡逆袭的七言绝句。"
    messages = [
        {"role": "system", "content": "你是一个才华横溢的诗人。"},
        {"role": "user", "content": prompt}
    ]

    print("\n[3/3] 开始推理...")

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    st = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    et = time.time()

    input_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[:, input_len:]
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    tokens_gen = output_ids.shape[1]
    speed = tokens_gen / (et - st)

    print("\n" + "="*20 + " 生成结果 " + "="*20)
    print(response)
    print("="*50)
    print(f"生成速度: {speed:.2f} tokens/s")
    print(f"显存占用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    import os
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    run_inference()
