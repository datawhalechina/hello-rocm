import time
import torch
import my_custom_swish_backend

class FusedSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = my_custom_swish_backend.forward(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = my_custom_swish_backend.backward(grad_output.contiguous(), x)
        return grad_x

def fused_swish(x):
    return FusedSwishFunction.apply(x)

# 准备 5000 万个元素的大张量 (约 200MB 显存)
size = 50000000
# native 作为参照组
x_native = torch.randn(size, device='cuda', requires_grad=True)
# custom 作为自定义算子测试组，克隆一份独立的数据
x_custom = x_native.clone().detach().requires_grad_(True)

print(f"\n开始 Benchmark，数据大小: {size} 元素...")

# 预热 GPU (防止第一次初始化和 JIT 编译开销影响计时)
for _ in range(10):
    (x_native * torch.sigmoid(x_native)).sum().backward()
    fused_swish(x_custom).sum().backward()

# --- 测试 1: 原生 PyTorch 性能 ---
torch.cuda.synchronize() # 确保 GPU 空闲
start = time.time()
for _ in range(50):
    out = x_native * torch.sigmoid(x_native) # Forward 启动至少 2 个 Kernel
    out.sum().backward()                     # Backward 启动数个 Kernel
torch.cuda.synchronize() # 等待所有任务完成
torch_time = (time.time() - start) / 50 * 1000 # 计算平均耗时 (ms)

# --- 测试 2: 自定义 Fused C++ 算子 ---
torch.cuda.synchronize()
start = time.time()
for _ in range(50):
    out = fused_swish(x_custom)  # Forward 仅启动 1 个 Kernel
    out.sum().backward()         # Backward 仅启动 1 个 Kernel
torch.cuda.synchronize()
custom_time = (time.time() - start) / 50 * 1000

print(f"\n--- 极限性能 Benchmark (5000万元素, 50轮平均) ---")
print(f"原生 PyTorch (Forward + Backward) 耗时: {torch_time:.2f} ms")
print(f"自定义 Fused 算子 (纯 C++) 耗时: {custom_time:.2f} ms")
print(f"综合性能提升: {torch_time / custom_time:.2f} 倍！")
