------

# 第 3 章：迈入 ROCm 编程世界——手写一个“PyTorch 算子”

> **🖥️ 实验环境**
>
> - **设备**: AMD AI+ MAX395
> - **GPU**: Radeon 8060S
> - **架构**: gfx1151 (RDNA 3)
> - **ROCm 版本**: 7.x
> - **系统**: Ubuntu 24.04 / 22.04

------

## 🎯 本章学习目标

通过本章，你将掌握以下核心技能：

1. ✅ **HIP 语言基础**：理解 Host 与 Device 的分工，掌握 GPU 编程的“普通话”。
2. ✅ **手写 Kernel**：从底层 C++ 复现 Python 中的 Tensor 加法逻辑，并学会使用 `hipEvent` 测量底层性能。
3. ✅ **调用高性能库**：理解 rocBLAS 与 MIOpen 的作用，实战编写一个完整的矩阵乘法（GEMM）程序。

------

在上一章中，我们了解了 GPU 的硬件架构与高并发的执行哲学。但在实际的 AI 开发中，当我们写下一句简单的 `c = a + b` 时，底层到底发生了什么？为了真正理解 PyTorch 这样的深度学习框架是如何驱动 GPU 狂奔的，本章我们将脱掉 Python 的高级外衣，深入底层，用 HIP 语言亲手写一个算子。

------

## 🗣️ 3.1 HIP 语言：GPU 编程的“普通话”

## ❓ HIP 是什么？它与 CUDA 的渊源

HIP（Heterogeneous-Compute Interface for Portability）是由 AMD 推出的一种基于 C++ 的异构计算编程语言。

在 AI 圈子里，NVIDIA 的 CUDA 具有极高的统治力。而 HIP 的语法与 CUDA **高达 99% 的相似**。AMD 甚至提供了一个名为 `hipify` 的工具，能够将写好的 CUDA 代码（`.cu`）一键转换为 HIP 代码。

**掌握了 HIP，你实际上也就掌握了 CUDA，反之亦然。**

## 🏢 Host (主机) 与 Device (设备) 的代码结构

在异构计算的世界里，代码有明确的分工。我们在同一个 `.cpp` 文件中，通过特定的**修饰符**来区分它们：

C++

```
#include <hip/hip_runtime.h>
#include <math.h>

// 🎮 __device__: 只能在 GPU 上执行，且只能被 GPU 上的其他函数调用
// 相当于 GPU 内部的辅助工具函数
__device__ float activate_relu(float x) {
    return fmaxf(0.0f, x);
}

// 🚀 __global__: 核函数 (Kernel)。在 CPU 上调用，在 GPU 上执行。
// 它是连接 Host 和 Device 的大门！
__global__ void relu_activation_kernel(float* data, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
        data[idx] = activate_relu(data[idx]); // 调用 __device__ 函数
    }
}

// 🖥️ __host__: 在 CPU 上执行的普通 C++ 函数（默认，可省略）
int main() {
    // 这里是跑在 CPU 上的主逻辑
    return 0;
}
```

------

## 🛠️ 3.2 揭秘 Tensor 加法：编写你的第一个 Kernel

## 🎬 场景引入：`c = a + b` 底层长什么样？

当你在 PyTorch 里对一个拥有一千万个元素的 Tensor 执行加法时，GPU 会启动一个包含一千万个线程的网格（Grid）。每个线程（Thread）只负责计算其中一个元素的加法。这就是 GPU “人海战术”的精髓。

## ✍️ 完整实战：带有计时器与错误检查的 `vector_add.cpp`

下面提供了一个**完整的、工业级标准**的向量加法代码。我们不仅实现了计算，还加入了规范的错误检查宏（`HIP_CHECK`）以及使用 `hipEvent` 进行纳秒级性能剖析（Profiling）的代码，以此揭示 Python 变慢的真正原因。

C++

```
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// 💡 宏定义：用于捕捉底层 HIP API 的错误
#define HIP_CHECK(command) {               \
    hipError_t status = command;           \
    if (status != hipSuccess) {            \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " at line " << __LINE__ << std::endl;      \
        exit(1);                           \
    }                                      \
}

// 🚀 核函数：向量加法
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    int n = 10000000; // 1000万个元素
    size_t bytes = n * sizeof(float);

    // 1️⃣ Host 端内存分配与初始化
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    // 2️⃣ Device 端显存分配
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // ⏱️ 创建事件，用于计时
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // 3️⃣ 数据搬运：CPU -> GPU (记录耗时)
    hipEventRecord(start);
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms_memcpy_h2d;
    hipEventElapsedTime(&ms_memcpy_h2d, start, stop);

    // 4️⃣ 执行 Kernel 计算 (记录耗时)
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    hipEventRecord(start);
    // 启动核函数语法：<<<Grid, Block>>>
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, n);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms_kernel;
    hipEventElapsedTime(&ms_kernel, start, stop);

    // 5️⃣ 数据搬运：GPU -> CPU
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

    // 📊 打印性能数据
    std::cout << "验证结果: c[0] = " << h_c[0] << " (预期: 3.0)" << std::endl;
    std::cout << "[性能剖析] 数据搬运 (H2D) 耗时: " << ms_memcpy_h2d << " ms" << std::endl;
    std::cout << "[性能剖析] Kernel 计算耗时: " << ms_kernel << " ms" << std::endl;

    // 6️⃣ 释放显存
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    hipEventDestroy(start); hipEventDestroy(stop);

    return 0;
}
```

## ⚙️ 编译与运行分析

使用 `hipcc` 编译上述代码：

Bash

```
hipcc vector_add.cpp -o vector_add -O3
./vector_add
```

⚠️ **运行结果揭秘**：你会惊讶地发现，**数据搬运的耗时（数十毫秒）可能比 Kernel 纯计算的耗时（不到一毫秒）高出几十倍！** 这就是为什么在 PyTorch 中频繁调用 `.cpu()` 会导致模型训练卡顿的根本原因——PCIe 带宽成为了真正的瓶颈。

------

## 📚 3.3 学会使用巨人的肩膀：ROCm 数学库

## 🆚 库 vs 手写：为什么不手写矩阵乘法？

上面的向量加法很简单，但如果是矩阵乘法（GEMM）呢？

要在 GPU 上写出极致性能的 GEMM，你需要手动利用共享内存（LDS）进行分块（Tiling）、管理寄存器溢出（Register Spilling）、并利用 Tensor Core (或 Matrix Core) 指令。这往往需要上千行极其晦涩的汇编级优化代码。

因此，现代深度学习框架（如 PyTorch 的 `nn.Linear`）在底层是**直接调用 ROCm 提供的数学库**的。

## 🧮 rocBLAS 实战：完整的 SGEMM 程序

`rocBLAS` 是 AMD 针对 BLAS (Basic Linear Algebra Subprograms) 的极致优化库。

下面是一个完整使用 rocBLAS 执行单精度矩阵乘法 $C = \alpha A \times B + \beta C$ 的精简示例：

C++

```
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>

int main() {
    // 矩阵维度: M=1024, N=1024, K=1024
    rocblas_int m = 1024, n = 1024, k = 1024;
    float alpha = 1.0f, beta = 0.0f;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    // 1. 分配显存
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    // (此处省略 CPU 初始化并通过 hipMemcpy 传入 d_A, d_B 的代码)

    // 2. 初始化 rocBLAS 句柄
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // 3. 🚀 调用高度优化的矩阵乘法 API
    // 注意：BLAS 库默认采用列主序(Column-Major)存储，参数顺序有严格要求
    rocblas_status status = rocblas_sgemm(
        handle,
        rocblas_operation_none, rocblas_operation_none,
        m, n, k,
        &alpha,
        d_A, m, // lda (Leading dimension)
        d_B, k, // ldb
        &beta,
        d_C, m  // ldc
    );

    if (status == rocblas_status_success) {
        std::cout << "rocBLAS SGEMM 执行成功！" << std::endl;
    }

    // 4. 清理资源
    rocblas_destroy_handle(handle);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);

    return 0;
}
```

**编译命令**（注意需要链接 `-lrocblas`）：

Bash

```
hipcc sgemm_example.cpp -o sgemm_example -lrocblas
```

## 🧠 MIOpen 简介：深度学习的“核动力引擎”

如果说 rocBLAS 搞定了全连接层，那么 **MIOpen** 就是搞定整个深度学习生态的引擎。它对标 NVIDIA 的 cuDNN。

当你调用 PyTorch 的 `nn.Conv2d` 时，底层并不是简单的矩阵相乘，MIOpen 会在后台执行极其复杂的决策：

1. **自动调优 (Auto-Tuning)**：分析你的卷积核大小、步长和批量大小。
2. **算法选择**：在 Direct 卷积、Winograd 算法、FFT（快速傅里叶变换）或 Implicit GEMM 中，动态挑出当前显卡上跑得最快的那一个算法。
3. **融合执行 (Kernel Fusion)**：试图把卷积和后面的 ReLU 激活合并成一个 Kernel 执行，进一步减少访存！

正是依靠 HIP、rocBLAS 和 MIOpen 这套稳固的底层软件栈，AMD 加速器才能完美支撑起上层 PyTorch 等框架的庞大算力需求。