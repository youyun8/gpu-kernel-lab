// reduction.cpp
//
// Sum reduction over a large array, comparing a shared-memory tree reduction
// with a warp-shuffle-finalized version (chapter 10). Both use a grid-stride
// loop and are validated against a CPU sum.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

__device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
#if defined(USE_CUDA)
    val += __shfl_down_sync(0xffffffffu, val, offset);
#else
    val += __shfl_down(val, offset);
#endif
  }
  return val;
}

// Shared-memory tree reduction down to one warp, then warp shuffle.
__global__ void reduceShared(const float* in, float* out, int n) {
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

// Grid-stride accumulate, then a single warp-shuffle reduction per warp and a
// small shared-memory combine across warps.
__global__ void reduceWarpShuffle(const float* in, float* out, int n) {
  __shared__ float warpSums[kBlockSize / 32 + 1];
  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = warpReduceSum(sum);
  int lane = tid % warpSize;
  int warpId = tid / warpSize;
  if (lane == 0) warpSums[warpId] = sum;
  __syncthreads();
  if (warpId == 0) {
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    float v = (lane < numWarps) ? warpSums[lane] : 0.0f;
    v = warpReduceSum(v);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

float finalizeHost(const std::vector<float>& partial) {
  double acc = 0.0;
  for (float v : partial) acc += v;
  return static_cast<float>(acc);
}

}  // namespace

int main() {
  constexpr int kN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  constexpr int kGrid = 1024;

  std::vector<float> host(kN);
  double refAcc = 0.0;
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>((i % 17) - 8) * 0.01f;
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
    const float total = finalizeHost(partial);
    const float rel = std::fabs(total - reference) / std::max(1.0e-6f, std::fabs(reference));
    if (rel > 1.0e-2f) {
      std::fprintf(stderr, "%s correctness FAILED: %f vs %f\n", name, total, reference);
      return false;
    }
    std::printf("%s correctness OK (%.4f vs %.4f)\n", name, total, reference);
    gklab::report(name, gklab::benchmarkKernel(launch, bytes, kN), kPeakGbPerSec, 0.0);
    return true;
  };

  if (!check("reduce_shared", [&]() { GPU_LAUNCH(reduceShared, kGrid, kBlockSize, 0, devIn, devPartial, kN); }))
    return EXIT_FAILURE;
  if (!check("reduce_warp_shuffle", [&]() { GPU_LAUNCH(reduceWarpShuffle, kGrid, kBlockSize, 0, devIn, devPartial, kN); }))
    return EXIT_FAILURE;

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devPartial));
  return EXIT_SUCCESS;
}
