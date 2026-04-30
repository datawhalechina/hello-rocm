#include <torch/extension.h>

// 声明外部 HIP 文件中定义的模板 Launch 函数
template <typename scalar_t>
void launch_fused_swish_forward(const scalar_t* input, scalar_t* output, int size);
template <typename scalar_t>
void launch_fused_swish_backward(const scalar_t* grad_output, const scalar_t* x, scalar_t* grad_x, int size);

// --- 防御性检查宏 ---
// 检查 Tensor 是否在 GPU 上
#define CHECK_HIP(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a HIP/CUDA tensor")
// 检查 Tensor 内存是否连续
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// 组合检查
#define CHECK_INPUT(x) CHECK_HIP(x); CHECK_CONTIGUOUS(x)

// --- 前向传播 C++ 接口 ---
torch::Tensor fused_swish_forward(torch::Tensor input) {
    CHECK_INPUT(input);
    // 预先分配好显存存放结果，形状、类型和设备与 input 保持一致
    auto output = torch::empty_like(input);

    // 动态分发宏：根据 input 的实际 scalar_type()，自动实例化并调用对应的 C++ 模板函数
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_swish_forward", ([&] {
        launch_fused_swish_forward<scalar_t>(
            input.data_ptr<scalar_t>(), // 获取底层显存指针
            output.data_ptr<scalar_t>(),
            input.numel() // 获取元素总数
        );
    }));
    return output;
}

// --- 反向传播 C++ 接口 ---
torch::Tensor fused_swish_backward(torch::Tensor grad_output, torch::Tensor x) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    // 分配用于存储 x 梯度的显存
    auto grad_x = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_swish_backward", ([&] {
        launch_fused_swish_backward<scalar_t>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            x.numel()
        );
    }));
    return grad_x;
}

// 使用 Pybind11 将 C++ 函数暴露给 Python
// TORCH_EXTENSION_NAME 是编译时自动生成的模块名
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_swish_forward, "Fused Swish Forward (HIP)");
    m.def("backward", &fused_swish_backward, "Fused Swish Backward (HIP)");
}
