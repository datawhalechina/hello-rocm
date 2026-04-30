#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// 宏定义：捕捉底层 API 错误（工业界标配）
#define HIP_CHECK(command) {               \
    hipError_t status = command;           \
    if (status != hipSuccess) {            \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " at line " << __LINE__ << std::endl;      \
        exit(1);                           \
    }                                      \
}

// 核函数：向量加法
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    // 使用 1D 寻址公式
    int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    // 边界保护：防止最后一个 Block 里的多余线程越界访问
    if (id < n) {
        c[id] = a[id] + b[id]; // 每个线程只负责一个元素的加法！
    }
}

int main() {
    int n = 10000000; // 1000万个元素
    size_t bytes = n * sizeof(float);

    // 1. Host 端内存分配与初始化
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    // 2. Device 端显存 (VRAM) 分配
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // 创建事件计时器
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);

    // 3. 数据搬运：CPU -> GPU (记录耗时)
    hipEventRecord(start);
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms_memcpy_h2d;
    hipEventElapsedTime(&ms_memcpy_h2d, start, stop);

    // 4. 执行 Kernel 计算
    int threadsPerBlock = 256;
    // 向上取整计算需要的 Block 数量
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    hipEventRecord(start);
    // 启动核函数：<<<Grid, Block>>>
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, n);
    hipEventRecord(stop);
    hipEventSynchronize(stop); // 等待 GPU 计时结束
    float ms_kernel;
    hipEventElapsedTime(&ms_kernel, start, stop);

    // 5. 数据搬运：GPU -> CPU
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

    // 打印性能数据
    std::cout << "验证: c[0] = " << h_c[0] << " (预期: 3.0)" << std::endl;
    std::cout << "[耗时] H2D 搬运 (PCIe): " << ms_memcpy_h2d << " ms" << std::endl;
    std::cout << "[耗时] Kernel 计算 (VRAM): " << ms_kernel << " ms" << std::endl;

    // 6. 释放显存
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return 0;
}
