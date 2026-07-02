// warp_reduce.cpp
//
// Demonstrates a portable warp-level sum reduction using shuffle (chapter 9).
// Uses warpSize (32 on CUDA, 64 on CDNA) rather than a hardcoded constant so
// the same source is correct on both platforms. One block per segment.
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

// Each block reduces blockDim.x elements and writes one partial sum.
__global__ void blockReduce(const float* in, float* out, int n) {
  __shared__ float warpSums[kBlockSize / 32 + 1];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (gid < n) ? in[gid] : 0.0f;
  v = warpReduceSum(v);
  int lane = threadIdx.x % warpSize;
  int warpId = threadIdx.x / warpSize;
  if (lane == 0) warpSums[warpId] = v;
  __syncthreads();
  if (warpId == 0) {
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    float s = (lane < numWarps) ? warpSums[lane] : 0.0f;
    s = warpReduceSum(s);
    if (lane == 0) out[blockIdx.x] = s;
  }
}

}  // namespace

int main() {
  constexpr int kN = 1 << 22;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  const int grid = (kN + kBlockSize - 1) / kBlockSize;

  std::vector<float> host(kN);
  double refAcc = 0.0;
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>((i % 11) - 5) * 0.02f;
    refAcc += host[i];
  }

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), grid * sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, host.data(), bytes));

  auto launch = [&]() { GPU_LAUNCH(blockReduce, grid, kBlockSize, 0, devIn, devOut, kN); };
  launch();
  GPU_CHECK(gpuDeviceSynchronize());

  std::vector<float> partial(grid);
  GPU_CHECK(gpuMemcpyDeviceToHost(partial.data(), devOut, grid * sizeof(float)));
  double total = 0.0;
  for (float v : partial) total += v;
  const float rel = std::fabs(static_cast<float>(total - refAcc)) / std::max(1.0e-6f, std::fabs(static_cast<float>(refAcc)));
  if (rel > 1.0e-2f) {
    std::fprintf(stderr, "warp_reduce correctness FAILED: %f vs %f\n", total, refAcc);
    return EXIT_FAILURE;
  }
  std::printf("warp_reduce correctness OK (%.4f vs %.4f)\n", total, refAcc);

  constexpr double kPeakGbPerSec = 1555.0;
  gklab::report("warp_reduce", gklab::benchmarkKernel(launch, bytes, kN), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
