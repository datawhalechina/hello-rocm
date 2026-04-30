#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

// 定义一个宏来检查 HIP API 的返回值
#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP Error: " << hipGetErrorString(err)                \
                      << " at line " << __LINE__ << std::endl;                  \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// 这是一个"核函数"(Kernel)，它将在 AMD GPU 上运行
// __global__ 是告诉编译器：这个函数在 GPU 上跑，但由 CPU 调用
__global__ void vector_add(float *a, float *b, float *c, int n) {
    // 获取当前线程的 ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i]; // 每个线程只算一个数的加法
    }
}

int main() {
    int n = 1024; // 向量长度
    size_t bytes = n * sizeof(float);

    // 1. 在 CPU (Host) 上分配内存
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // 初始化数据
    for(int i=0; i<n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 2. 在 GPU (Device) 上分配显存
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // 3. 把数据从 CPU 搬运到 GPU (H2D)
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    // 4. 启动核函数！让显卡干活
    // 语法: <<<GridDim, BlockDim>>>
    // 这里开启 1 个 Block，里面有 1024 个线程并行计算
    hipLaunchKernelGGL(vector_add, dim3(1), dim3(n), 0, 0, d_a, d_b, d_c, n);

    // 等待 GPU 干完活
    HIP_CHECK(hipDeviceSynchronize());

    // 5. 把结果搬回 CPU (D2H)
    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));

    // 验证结果
    std::cout << "Element [0]: " << h_a[0] << " + " << h_b[0] << " = " << h_c[0] << std::endl;
    std::cout << "Element [1023]: " << h_a[1023] << " + " << h_b[1023] << " = " << h_c[1023] << std::endl;
    std::cout << ">>> ROCm HIP Kernel executed successfully on AMD GPU!" << std::endl;

    // 清理内存
    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_b)); HIP_CHECK(hipFree(d_c));
    free(h_a); free(h_b); free(h_c);

    return 0;
}