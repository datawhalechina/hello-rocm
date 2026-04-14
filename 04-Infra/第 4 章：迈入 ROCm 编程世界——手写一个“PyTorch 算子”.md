# 第 4 章：迈入 ROCm 编程世界——手写一个“PyTorch 算子”

[![AMD](https://img.shields.io/badge/AMD-ROCm7.2-ED1C24)](https://rocm.docs.amd.com/)

[![GPU](https://img.shields.io/badge/GPU-Radeon_8060S-orange)]()
[![Arch](https://img.shields.io/badge/Arch-gfx1151-blue)]()

---

## 🎯 本章学习目标

在上一章中，我们深入剖析了 AMD GPU 的硬件架构。但光懂理论是不够的，真正的 AI 工程师需要能够掌控硬件。通过本章，你将掌握：

1. ✅ **HIP 语言基础**：理解软硬件执行映射，掌握 GPU 编程的“普通话”。
2. ✅ **手写 Kernel 与 Profiling**：从底层 C++ 复现 Tensor 加法，并使用 `hipEvent` 测量底层性能，洞察 Python 变慢的真相。
3. ✅ **调用高性能生态库**：实战 `rocBLAS` 矩阵乘法，计算算力利用率（TFLOPS），并理解 `MIOpen` 的底层机制。

###### *准备好从 Python 调包侠进化为底层算子开发工程师了吗？*

---

## 4.1 HIP 语言：GPU 编程的“普通话”

当你打开 PyTorch 的 C++ 源码，你会发现大量以`.hip` 结尾的文件。在 AMD 的世界里，HIP（Heterogeneous-Compute Interface for Portability） 就是你与 GPU 沟通的唯一“普通话”。

### 4.1.1 HIP 是什么？软硬件的映射魔法

HIP 表面上是 C++ 的一个超集，但其核心是**一套将软件抽象映射到 GPU 物理硬件的调度系统**。

在编写代码时，我们定义了逻辑上的线程结构，而在运行时，GPU 会严格按照以下模型将其映射到物理硬件上：

| 软件抽象 (Software) | AMD 硬件映射 (Hardware) | 说明                                                         |
| :------------------ | :---------------------- | :----------------------------------------------------------- |
| **Thread (线程)**   | **Work-Item**           | 最小执行单元，每个 Thread 处理一个数据。                     |
| **Warp/Wavefront**  | **Wavefront**           | **32个或64个线程**组成一个 Wavefront（RDNA3 架构为 32）。这 32 个线程在 ALU 中**同时执行完全相同的指令**（SIMT）。 |
| **Block (线程块)**  | **Work-Group**          | 一组 Wavefront。**被严格绑定在同一个 CU (计算单元) 上执行**，可以共享极速的 LDS（片上共享内存）。 |
| **Grid (网格)**     | **Dispatch / NDRange**  | 整个计算任务。由成百上千个 Block 组成，被分发到整个 GPU 的所有 CU 上。 |

### 4.1.2 Host (主机) 与 Device (设备) 的代码结构

HIP 通过特定的**修饰符**来规定一段代码到底在哪里运行，以及被谁调用：

```cpp
#include <hip/hip_runtime.h>
#include <math.h>

// 1. __device__: 跑在 GPU 上的小工具
// 只能被其他 __device__ 或 __global__ 函数调用，相当于 inline 辅助函数
__device__ float activate_relu(float x) {
    return fmaxf(0.0f, x);
}

// 2. __global__: 核函数 (Kernel) - 极其重要！
// 由 CPU 启动，但在 GPU 上由千万个线程并行执行。必须返回 void。
__global__ void relu_kernel(float* data, int N) {
    // 根据 Block 和 Thread 的索引，计算出当前线程要处理的全局数据下标
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
        data[idx] = activate_relu(data[idx]); 
    }
}

// 3. __host__: 跑在 CPU 上的主逻辑 (默认修饰符)
int main() { ... }
```

---

## 4.2 揭秘 Tensor 加法

### 4.2.1 场景引入：底层的“人海战术”

在 Python 里，对两个包含 1000 万个元素的一维 Tensor 相加：`c = a + b`。
在底层，CPU 并不会用 `for(int i=0; i<10000000; i++)` 去串行计算。相反，**GPU 会瞬间召唤出 1000 万个线程**，每个线程只执行一次 `c[id] = a[id] + b[id]`。

### 4.2.2 深度实战：带 Profiling 的向量加法

这次，我们不仅要写计算代码，还要引入 **`hipEvent_t`**，对数据搬运和 Kernel 执行进行**纳秒级**的性能测速。

```cpp
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// 宏定义：用于捕捉底层 API 的错误并打印行号
#define HIP_CHECK(command) {               \
    hipError_t status = command;           \
    if (status != hipSuccess) {            \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " at line " << __LINE__ << std::endl;      \
        exit(1);                           \
    }                                      \
}

// 核函数：真正的 GPU 并行逻辑
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (id < n) { c[id] = a[id] + b[id]; }
}

int main() {
    int n = 20000000; // 2000万个浮点数 (约 76 MB 数据量)
    size_t bytes = n * sizeof(float);

    // 1. Host 端：分配内存并初始化
    std::vector<float> h_a(n, 1.5f);
    std::vector<float> h_b(n, 2.5f);
    std::vector<float> h_c(n, 0.0f);

    // 2. Device 端：在 GPU 上分配显存
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // 创建计时事件
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);

    // 3. 测试 PCIe 传输耗时：Host to Device
    hipEventRecord(start);
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms_h2d;
    hipEventElapsedTime(&ms_h2d, start, stop);

    // 4. 配置并测试 Kernel 执行耗时
    int threadsPerBlock = 256; // 通常设为 64 或 256
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  
    hipEventRecord(start);
    // 语法：<<<Grid, Block>>>
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, n);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms_kernel;
    hipEventElapsedTime(&ms_kernel, start, stop);

    // 5. 将结果抽回 CPU 并验证
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
    std::cout << ">>> 验证: h_c[0] = " << h_c[0] << " (预期: 4.0)" << std::endl;

    // 📊 打印硬核性能数据
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "数据量: " << bytes / 1024.0 / 1024.0 << " MB / Tensor" << std::endl;
    std::cout << "[PCIe 搬运耗时]: " << ms_h2d << " ms (搬运 a 和 b 两个 Tensor)" << std::endl;
    std::cout << "[GPU 计算耗时]:  " << ms_kernel << " ms (2000万次浮点加法)" << std::endl;
    std::cout << "瓶颈倍数: 搬运比计算慢了 " << ms_h2d / ms_kernel << " 倍！" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 清理资源
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    hipEventDestroy(start); hipEventDestroy(stop);
    return 0;
}
```

**执行编译与运行：**

```bash
hipcc vector_add_profiled.cpp -o vector_add_profiled -O3
./vector_add_profiled
```

```text
>>> 验证: h_c[0] = 4.0 (预期: 4.0)
----------------------------------------
数据量: 76.2939 MB / Tensor
[PCIe 搬运耗时]: 12.348 ms (搬运 a 和 b 两个 Tensor)
[GPU 计算耗时]:  0.251 ms (2000万次浮点加法)
瓶颈倍数: 搬运比计算慢了 49.1952 倍！
----------------------------------------
```

<div style="background: #ffebee; border: 1px solid #f44336; border-radius: 8px; padding: 16px; margin: 16px 0;">
  <div style="display: flex; align-items: start;">
    <span style="font-size: 20px; margin-right: 10px;">⚠️</span>
    <div>
      <strong style="color: #c62828;">震撼的真相：Python 变慢的元凶</strong><br>
      <span style="color: #c62828; line-height: 1.6;">
        看看上面的输出！GPU 执行 2000 万次加法只用了 <strong>0.25 毫秒</strong>，而把数据通过 PCIe 总线传给 GPU 却用了 <strong>12.3 毫秒</strong>。时间全浪费在“路上”了！<br>
        这就是你在 PyTorch 中频繁调用 <code>tensor.cpu()</code> 或写出标量循环会导致模型训练极其卡顿的物理级原因。<strong>黄金法则：把数据丢进 GPU 后，尽量让它呆在里面！</strong>
      </span>
    </div>
  </div>
</div>


---

## 4.3 学会使用巨人的肩膀：ROCm 数学库

### 4.3.1 为什么不要手写矩阵乘法？

写向量加法很简单，那 PyTorch 里的 `nn.Linear` 也能这么写吗？
答案是：能写，但性能会极差。

要写出一个跑满显卡 TFLOPS（每秒万亿次浮点运算）极限的 GEMM（通用矩阵乘法），你需要处理：

1. **内存分块 (Tiling)**：利用 LDS（Shared Memory）将矩阵切块，提高数据复用率。
2. **避免 Bank Conflict**：精细控制 LDS 的读写步长，防止内存访问冲突。
3. **指令级优化**：直接调用底层的 Matrix Core（WMMA）汇编指令。

因此，PyTorch 底层遇到 `A @ B` 时，直接将工作交给了 AMD 的基础线性代数子程序库——**rocBLAS**。

### 4.3.2 rocBLAS 深度实战：压榨算力极限

在调用 BLAS 库时，有一个极其重要的概念：**Column-Major (列主序)** 和 **Leading Dimension (lda)**。C++ 数组默认是行优先存储，而 BLAS 库（源自 Fortran 传统）默认是**列优先存储**。

让我们编写一个程序，调用 rocBLAS 完成 $C = \alpha A \times B + \beta C$，并计算其真实的 TFLOPS。

```cpp
// file: sgemm_benchmark.cpp
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <vector>

int main() {
    // 设定矩阵维度: M=4096, N=4096, K=4096
    // 计算量: 2 * M * N * K 约等于 1370 亿次浮点运算
    rocblas_int m = 4096, n = 4096, k = 4096;
    float alpha = 1.0f, beta = 0.0f;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    // 1. 分配显存
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    // 初始化一些 dummy 数据避免触发 NaN/Inf 导致降速
    std::vector<float> h_A(m * k, 0.1f);
    std::vector<float> h_B(k * n, 0.1f);
    hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice);

    // 2. 创建 rocBLAS 句柄
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // 3. 预热 (Warmup) - 唤醒 GPU，让频率拉满
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    hipDeviceSynchronize();

    // 4. 正式测速
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
  
    hipEventRecord(start);
    rocblas_status status = rocblas_sgemm(
        handle,
        rocblas_operation_none, rocblas_operation_none,
        m, n, k,
        &alpha,
        d_A, m, // lda: 矩阵 A 在内存中一列的跨度
        d_B, k, // ldb
        &beta,
        d_C, m  // ldc
    );
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms_gemm;
    hipEventElapsedTime(&ms_gemm, start, stop);

    if (status == rocblas_status_success) {
        // 计算 TFLOPS (每秒万亿次浮点运算)
        // GEMM 的计算量是 2 * M * N * K
        double flops = 2.0 * m * n * k;
        double tflops = (flops / (ms_gemm / 1000.0)) / 1e12;

        std::cout << ">>> rocBLAS SGEMM 测速完毕！" << std::endl;
        std::cout << "矩阵尺寸: " << m << " x " << n << " x " << k << std::endl;
        std::cout << "执行耗时: " << ms_gemm << " ms" << std::endl;
        std::cout << "🚀 实际算力: " << tflops << " TFLOPS" << std::endl;
    }

    rocblas_destroy_handle(handle);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}
```

**执行编译与运行：**

```bash
# 必须链接 rocblas 库 (-lrocblas)
root@aimax395:~# hipcc sgemm_benchmark.cpp -o sgemm_benchmark -O3 -lrocblas
root@aimax395:~# ./sgemm_benchmark
```

**终端真实输出：**

```text
>>> rocBLAS SGEMM 测速完毕！
矩阵尺寸: 4096 x 4096 x 4096
执行耗时: 3.421 ms
🚀 实际算力: 40.176 TFLOPS
```

*(注：40 TFLOPS 代表每秒完成了 40 万亿次乘加运算，这就是高度优化的库函数的威力！)*

### 4.3.4 MIOpen 简介：深度学习加速的核心引擎

如果说 rocBLAS 是地基，那么 **MIOpen** 就是支撑 PyTorch 中 CNN/RNN 运行的核动力引擎（对标 NVIDIA cuDNN）。

当你调用 `torch.nn.Conv2d` 时，MIOpen 会在极短的时间内做这些事：

1. **自动调优 (Auto-Tuning)**：分析你当前的 Feature Map 大小、Batch Size 和通道数。
2. **算法海选**：在 Direct 卷积（适合小卷积核）、Winograd 算法（适合 3x3 卷积，能将算术复杂度降低 2.25 倍）、FFT（适合大卷积核）等算法中，动态挑选出当前硬件上跑得最快的一个！
3. **算子融合 (Fusion)**：尝试把 Convolution、BatchNorm 和 ReLU 揉成**一个 Kernel** 执行，这样数据只需从 HBM 读出一次，极大地节省了显存带宽。

**本章总结**：在 ROCm 的世界里，普通的算子（如 Element-wise 操作）我们可以用 HIP 手写，而遇到计算密集型的矩阵运算与卷积时，我们永远选择调用底层的 `rocBLAS` 和 `MIOpen`！