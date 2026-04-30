import torch
import torchvision
import time

# 设置设备
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"=== GPU 信息 ===")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"架构: {props.name}")  # 应该显示 gfx1151
    print(f"显存: {props.total_memory / 1024**3:.1f} GB")
    print(f"计算能力: {props.major}.{props.minor}")
else:
    print("未检测到 GPU")

# 加载 ResNet-50 模型
print("\n=== 加载 ResNet-50 模型 ===")
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# 准备测试数据
batch_size = 32
dummy_input = torch.randn(batch_size, 3, 224, 224)

# ========== CPU 推理测试 ==========
print("\n=== CPU 推理测试 ===")
model_cpu = model.to(device_cpu)

# 预热
for _ in range(3):
    with torch.no_grad():
        _ = model_cpu(dummy_input)

# 正式测试
torch.cpu.synchronize() if hasattr(torch, 'cpu') else None
start = time.time()
num_iterations = 10
with torch.no_grad():
    for _ in range(num_iterations):
        _ = model_cpu(dummy_input)
end = time.time()

cpu_time_ms = (end - start) / num_iterations * 1000 / batch_size
cpu_throughput = batch_size * num_iterations / (end - start)

print(f"CPU 平均延迟: {cpu_time_ms:.2f} ms/image")
print(f"CPU 吞吐量: {cpu_throughput:.2f} img/s")

# ========== GPU 推理测试 ==========
if torch.cuda.is_available():
    print("\n=== GPU 推理测试 ===")
    model_gpu = model.to(device_gpu)
    dummy_input_gpu = dummy_input.to(device_gpu)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model_gpu(dummy_input_gpu)
        torch.cuda.synchronize()

    # 正式测试
    torch.cuda.synchronize()
    start = time.time()
    num_iterations = 100
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model_gpu(dummy_input_gpu)
    torch.cuda.synchronize()
    end = time.time()

    gpu_time_ms = (end - start) / num_iterations * 1000 / batch_size
    gpu_throughput = batch_size * num_iterations / (end - start)

    print(f"GPU 平均延迟: {gpu_time_ms:.2f} ms/image")
    print(f"GPU 吞吐量: {gpu_throughput:.2f} img/s")

    # 计算加速比
    speedup = cpu_time_ms / gpu_time_ms
    print(f"\n🚀 GPU 加速比: {speedup:.1f}x")

    # 显存使用
    print(f"\n显存使用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")