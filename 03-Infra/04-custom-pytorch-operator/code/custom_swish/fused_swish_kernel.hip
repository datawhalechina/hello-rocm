#include <hip/hip_runtime.h>
#include <math.h>

// 1. 前向传播 Kernel：支持 Grid-Stride Loop 和 模板泛型
template <typename scalar_t>
__global__ void fused_swish_forward_kernel(const scalar_t* input, scalar_t* output, int size) {
    // 计算当前线程的全局索引
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // 计算跨步大小：整个网格的线程总数
    int stride = hipBlockDim_x * hipGridDim_x;

    // 网格跨步循环，处理大于线程总数的数据
    for (int i = idx; i < size; i += stride) {
        // 强制转换为 float 进行中间计算，保证精度
        float x = static_cast<float>(input[i]);
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        // 计算 Swish: x * sigmoid(x)，并转回原类型写回
        output[i] = static_cast<scalar_t>(x * sigmoid_x);
    }
}

// 2. 反向传播 Kernel
// Swish 的导数推导: f'(x) = f(x) + sigmoid(x) * (1 - f(x))
template <typename scalar_t>
__global__ void fused_swish_backward_kernel(const scalar_t* grad_output, const scalar_t* x, scalar_t* grad_x, int size) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int stride = hipBlockDim_x * hipGridDim_x;

    for (int i = idx; i < size; i += stride) {
        float val_x = static_cast<float>(x[i]);
        float go = static_cast<float>(grad_output[i]);

        float sigmoid_x = 1.0f / (1.0f + expf(-val_x));
        float swish_x = val_x * sigmoid_x;

        // 根据链式法则计算当前元素的梯度：grad_output * f'(x)
        float grad_val = go * (swish_x + sigmoid_x * (1.0f - swish_x));
        grad_x[i] = static_cast<scalar_t>(grad_val);
    }
}

// 3. 供 C++ Wrapper 调用的 Host 端启动函数
template <typename scalar_t>
void launch_fused_swish_forward(const scalar_t* input, scalar_t* output, int size) {
    int threads = 256;
    // 限制最多启动 256 个 Block，利用跨步循环处理超大数据，避免调度过载
    int blocks = min((size + threads - 1) / threads, 256);
    hipLaunchKernelGGL(fused_swish_forward_kernel<scalar_t>, dim3(blocks), dim3(threads), 0, 0, input, output, size);
}

template <typename scalar_t>
void launch_fused_swish_backward(const scalar_t* grad_output, const scalar_t* x, scalar_t* grad_x, int size) {
    int threads = 256;
    int blocks = min((size + threads - 1) / threads, 256);
    hipLaunchKernelGGL(fused_swish_backward_kernel<scalar_t>, dim3(blocks), dim3(threads), 0, 0, grad_output, x, grad_x, size);
}

// 4. 显式实例化模板（告诉编译器我们需要编译哪些数据类型的版本）
template void launch_fused_swish_forward<float>(const float*, float*, int);
template void launch_fused_swish_backward<float>(const float*, const float*, float*, int);
