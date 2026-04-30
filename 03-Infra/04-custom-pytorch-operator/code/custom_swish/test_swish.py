import torch
# 导入我们刚才编译安装好的底层 C++ 库
import my_custom_swish_backend

class FusedSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        前向传播逻辑
        ctx: 上下文对象，用于存储反向传播需要的信息
        x: 输入 Tensor
        """
        # 1. 调用底层 C++ 前向函数
        result = my_custom_swish_backend.forward(x)
        # 2. 将输入 x 存入上下文(Context)，留给反向求导时使用
        # 因为 Swish 的导数计算需要用到原始输入 x
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播逻辑
        ctx: 上下文对象，取回前向存储的信息
        grad_output: 上游传来的梯度
        """
        # 1. 取出前向时保存的 x
        x, = ctx.saved_tensors

        # 2. 检查链式法则上游是否需要梯度 (工业级优化)
        grad_x = None
        if ctx.needs_input_grad[0]:
            # 3. 调用底层 C++ 反向函数
            # 注意: 反向传播传进来的 grad_output 可能因为经过各种切片操作导致在内存中不再连续
            # 所以调用 .contiguous() 是必不可少的防御手段！
            grad_x = my_custom_swish_backend.backward(grad_output.contiguous(), x)

        # 返回输入 x 的梯度
        return grad_x

# 封装成一个优雅的 Python 函数供深度学习模型使用
def fused_swish(x):
    return FusedSwishFunction.apply(x)

# ======== 验证求导链路是否畅通 ========
print("--- 功能与精度验证 ---")
# 创建一个需要梯度的 Tensor
x = torch.randn(5, device='cuda', dtype=torch.float32, requires_grad=True)

print("输入 x:", x)

# 前向传播
out = fused_swish(x)
print("Swish 输出:", out)

# 模拟算出一个标量 Loss
loss = out.sum()

# 一键反向传播！PyTorch 会自动调用我们定义的 backward 方法
loss.backward()

print("自动求导后的梯度 (x.grad):", x.grad)

# 验证：Swish 在 x=0 处的导数应该是 0.5
x_zero = torch.tensor([0.0], device='cuda', requires_grad=True)
fused_swish(x_zero).backward()
print("x=0 处的导数 (预期 0.5):", x_zero.grad.item())

print("Autograd 反向传播打通！自定义算子现在具有学习能力了！")
