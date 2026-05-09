#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// AMD wavefront 通常是 64；在 ROCm 上 host 侧不要用 warpSize（它是 device 内建）
constexpr int WARP_SIZE = 32;

// =============== 版本1：分支+同步的传统 reduction ===============
__global__ void reduce_branchy(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int global = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0f;
    if (global < n) sum += in[global];
    if (global + blockDim.x < n) sum += in[global + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// =============== wavefront shuffle sum ===============
__device__ __forceinline__ float wf_reduce_sum(float v) {
    // 用固定 WARP_SIZE，避免 warpSize 在 host/device 混用的坑
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, WARP_SIZE);
    }
    return v;
}

__global__ void reduce_shuffle(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n) {
    // 每个 wavefront 一个 partial sum
    extern __shared__ float wf_sums[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;

    float v = 0.0f;
    if (idx < n) v += in[idx];
    if (idx + blockDim.x < n) v += in[idx + blockDim.x];

    float wf_sum = wf_reduce_sum(v);

    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;

    if (lane == 0) wf_sums[wid] = wf_sum;
    __syncthreads();

    // wave0 汇总所有 wave partials
    if (wid == 0) {
        int num_waves = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float x = (lane < num_waves) ? wf_sums[lane] : 0.0f;
        float block_sum = wf_reduce_sum(x);
        if (lane == 0) out[blockIdx.x] = block_sum;
    }
}

// =============== Host：多轮 reduction 直到剩一个数 ===============
float run_reduce(const float* d_in, int n,
                 bool use_shuffle,
                 int threads, int iterations) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    int max_blocks = (n + (threads * 2 - 1)) / (threads * 2);
    float* d_buf1 = nullptr;
    float* d_buf2 = nullptr;
    HIP_CHECK(hipMalloc(&d_buf1, max_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_buf2, max_blocks * sizeof(float)));

    auto launch_once = [&](int cur_n, const float* cur_in, float* cur_out) {
        int blocks = (cur_n + (threads * 2 - 1)) / (threads * 2);

        size_t smem = 0;
        if (use_shuffle) {
            int num_waves = (threads + WARP_SIZE - 1) / WARP_SIZE;
            smem = num_waves * sizeof(float);
            hipLaunchKernelGGL(reduce_shuffle, dim3(blocks), dim3(threads),
                               smem, 0, cur_in, cur_out, cur_n);
        } else {
            smem = threads * sizeof(float);
            hipLaunchKernelGGL(reduce_branchy, dim3(blocks), dim3(threads),
                               smem, 0, cur_in, cur_out, cur_n);
        }
        return blocks;
    };

    // warmup
    {
        int cur_n = n;
        const float* cur_in = d_in;
        float* cur_out = d_buf1;
        while (cur_n > 1) {
            int next_n = launch_once(cur_n, cur_in, cur_out);
            cur_n = next_n;
            cur_in = cur_out;
            cur_out = (cur_out == d_buf1) ? d_buf2 : d_buf1;
        }
        HIP_CHECK(hipDeviceSynchronize());
    }

    HIP_CHECK(hipEventRecord(start));
    for (int it = 0; it < iterations; ++it) {
        int cur_n = n;
        const float* cur_in = d_in;
        float* cur_out = d_buf1;

        while (cur_n > 1) {
            int next_n = launch_once(cur_n, cur_in, cur_out);
            cur_n = next_n;
            cur_in = cur_out;
            cur_out = (cur_out == d_buf1) ? d_buf2 : d_buf1;
        }
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    HIP_CHECK(hipFree(d_buf1));
    HIP_CHECK(hipFree(d_buf2));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
}

int main() {
    int n = 1024 * 1024 * 1000; // 10M
    size_t bytes = n * sizeof(float);

    std::cout << "=== 向量求和（reduction）发散对比 ===\n";
    std::cout << "数据量: " << n << " (" << bytes / 1024.0 / 1024.0 << " MB)\n";
    std::cout << "WARP_SIZE(AMD wavefront): " << WARP_SIZE << "\n";

    std::vector<float> h(n, 1.0f);
    float* d_in = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMemcpy(d_in, h.data(), bytes, hipMemcpyHostToDevice));

    int threads = 256;
    int iterations = 50;

    std::cout << "\n=== reduce_branchy（分支+同步）===\n";
    float t1 = run_reduce(d_in, n, false, threads, iterations);
    std::cout << "总时间: " << t1 << " ms\n";
    std::cout << "平均每次: " << t1 / iterations << " ms\n";

    std::cout << "\n=== reduce_shuffle（shuffle 更少分支）===\n";
    float t2 = run_reduce(d_in, n, true, threads, iterations);
    std::cout << "总时间: " << t2 << " ms\n";
    std::cout << "平均每次: " << t2 / iterations << " ms\n";

    std::cout << "\n=== 对比 ===\n";
    float speedup = t1 / t2;
    std::cout << "shuffle 版本加速比: " << speedup << "x\n";
    std::cout << "性能提升: " << (speedup - 1.f) * 100.f << "%\n";

    HIP_CHECK(hipFree(d_in));
    return 0;
}