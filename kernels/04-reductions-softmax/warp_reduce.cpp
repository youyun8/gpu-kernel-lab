// warp_reduce.cpp
//
// Demonstrates a portable warp-level sum reduction using shuffle (chapter 9).
// Uses warp_size (32 on CUDA, 64 on CDNA) rather than a hardcoded constant so
// the same source is correct on both platforms. One block per segment.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

__device__ float warpReduceSum(float val) {
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
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
  __shared__ float warp_sums[kBlockSize / 32 + 1];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (gid < n) ? in[gid] : 0.0f;
  v = warpReduceSum(v);
  int lane = threadIdx.x % warp_size;
  int warp_id = threadIdx.x / warp_size;
  if (lane == 0) warp_sums[warp_id] = v;
  __syncthreads();
  if (warp_id == 0) {
    int num_warps = (blockDim.x + warp_size - 1) / warp_size;
    float s = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    s = warpReduceSum(s);
    if (lane == 0) out[blockIdx.x] = s;
  }
}

}  // namespace

int main() {
  constexpr int kSizeN = 1 << 22;
  const size_t bytes = static_cast<size_t>(kSizeN) * sizeof(float);
  const int grid = (kSizeN + kBlockSize - 1) / kBlockSize;

  std::vector<float> host(kSizeN);
  double ref_acc = 0.0;
  for (int i = 0; i < kSizeN; ++i) {
    host[i] = static_cast<float>((i % 11) - 5) * 0.02f;
    ref_acc += host[i];
  }

  float* dev_in = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), grid * sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), bytes));

  auto launch = [&]() { GPU_LAUNCH(blockReduce, grid, kBlockSize, 0, dev_in, dev_out, kSizeN); };
  launch();
  GPU_CHECK(gpuDeviceSynchronize());

  std::vector<float> partial(grid);
  GPU_CHECK(gpuMemcpyDeviceToHost(partial.data(), dev_out, grid * sizeof(float)));
  double total = 0.0;
  for (float v : partial) total += v;
  const float rel = std::fabs(static_cast<float>(total - ref_acc)) / std::max(1.0e-6f, std::fabs(static_cast<float>(ref_acc)));
  if (rel > 1.0e-2f) {
    std::fprintf(stderr, "warp_reduce correctness FAILED: %f vs %f\n", total, ref_acc);
    return EXIT_FAILURE;
  }
  std::printf("warp_reduce correctness OK (%.4f vs %.4f)\n", total, ref_acc);

  constexpr double kPeakGbPerSec = 1555.0;
  gklab::report("warp_reduce", gklab::benchmarkKernel(launch, bytes, kSizeN), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(dev_in));
  GPU_CHECK(gpuFree(dev_out));
  return EXIT_SUCCESS;
}
