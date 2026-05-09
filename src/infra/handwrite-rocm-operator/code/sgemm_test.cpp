#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <vector>

int main() {
    rocblas_int m = 1024, n = 1024, k = 1024;
    float alpha = 1.0f, beta = 0.0f;
    size_t size = m * n * sizeof(float);

    // 1. Host 端初始化
    std::vector<float> h_A(m * k, 1.0f);
    std::vector<float> h_B(k * n, 2.0f);
    std::vector<float> h_C(m * n, 0.0f);

    // 2. Device 端显存分配与拷贝
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size); hipMalloc(&d_B, size); hipMalloc(&d_C, size);
    hipMemcpy(d_A, h_A.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size, hipMemcpyHostToDevice);

    // 3. 初始化 rocBLAS 句柄（Handle 管理上下文和资源）
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // 4. 调用高度优化的矩阵乘法 (C = alpha*A*B + beta*C)
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  m, n, k, &alpha,
                  d_A, m, d_B, k, &beta, d_C, m);

    // 5. 拷回结果并打印左上角 4x4
    hipMemcpy(h_C.data(), d_C, size, hipMemcpyDeviceToHost);

    std::cout << "=== 结果矩阵 C 的左上角 4x4 局部 ===\n";
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            // 列主序寻址：index = i + j * m
            std::cout << h_C[i + j * m] << "\t";
        }
        std::cout << "\n";
    }

    rocblas_destroy_handle(handle);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}
