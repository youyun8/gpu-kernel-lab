// ex_b_block_reduce.cpp
//
// Reference solution for exercise B9: block sum reduction with two finalizers,
// (a) a pure shared-memory tree and (b) a tree that narrows to one warp and
// finishes with warp shuffle. Both are validated against a CPU sum.
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

// (a) Pure shared-memory tree reduction, grid-stride input.
__global__ void reduceTree(const float* in, float* out, int n) {
  __shared__ float smem[kBlockSize];
  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  smem[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = smem[0];
}

// (b) Tree down to one warp, then warp-shuffle finalize.
__global__ void reduceTreeWarp(const float* in, float* out, int n) {
  __shared__ float smem[kBlockSize];
  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  smem[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > warpSize; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  if (tid < warpSize) {
    float v = smem[tid];
    if (blockDim.x >= 2 * warpSize) v += smem[tid + warpSize];
    for (int o = warpSize / 2; o > 0; o >>= 1) {
#if defined(USE_CUDA)
      v += __shfl_down_sync(0xffffffffu, v, o);
#else
      v += __shfl_down(v, o);
#endif
    }
    if (tid == 0) out[blockIdx.x] = v;
  }
}

}  // namespace

int main() {
  constexpr int kN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  constexpr int kGrid = 1024;

  std::vector<float> host(kN);
  double refAcc = 0.0;
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>((i % 19) - 9) * 0.01f;
    refAcc += host[i];
  }
  const float reference = static_cast<float>(refAcc);

  float* devIn = nullptr;
  float* devPartial = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devPartial), kGrid * sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, host.data(), bytes));

  std::vector<float> partial(kGrid);
  constexpr double kPeakGbPerSec = 1555.0;

  auto check = [&](const char* name, const std::function<void()>& launch) -> bool {
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(partial.data(), devPartial, kGrid * sizeof(float)));
    double total = 0.0;
    for (float v : partial) total += v;
    const float rel = std::fabs(static_cast<float>(total) - reference) /
                      std::max(1.0e-6f, std::fabs(reference));
    if (rel > 1.0e-2f) {
      std::fprintf(stderr, "%s FAILED: %f vs %f\n", name, total, reference);
      return false;
    }
    std::printf("%s correctness OK (%.4f vs %.4f)\n", name, total, reference);
    gklab::report(name, gklab::benchmarkKernel(launch, bytes, kN), kPeakGbPerSec, 0.0);
    return true;
  };

  if (!check("reduce_tree", [&]() { GPU_LAUNCH(reduceTree, kGrid, kBlockSize, 0, devIn, devPartial, kN); }))
    return EXIT_FAILURE;
  if (!check("reduce_tree_warp", [&]() { GPU_LAUNCH(reduceTreeWarp, kGrid, kBlockSize, 0, devIn, devPartial, kN); }))
    return EXIT_FAILURE;

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devPartial));
  return EXIT_SUCCESS;
}
