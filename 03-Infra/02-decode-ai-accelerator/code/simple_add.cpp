// file: simple_add.cpp
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);

    float *a, *b, *c;
    hipMalloc(&a, bytes);
    hipMalloc(&b, bytes);
    hipMalloc(&c, bytes);

    // 初始化数据
    float *h_a = new float[n];
    float *h_b = new float[n];
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    hipMemcpy(a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(b, h_b, bytes, hipMemcpyHostToDevice);

    // 启动核函数
    hipLaunchKernelGGL(add, dim3(1), dim3(n), 0, 0, a, b, c, n);
    hipDeviceSynchronize();

    // 验证结果
    float *h_c = new float[n];
    hipMemcpy(h_c, c, bytes, hipMemcpyDeviceToHost);

    std::cout << "Result: " << h_c[0] << ", " << h_c[n-1] << std::endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(a);
    hipFree(b);
    hipFree(c);

    return 0;
}