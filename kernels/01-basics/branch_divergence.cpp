// branch_divergence.cpp
//
// Measures the SIMT divergence penalty. Both kernels do identical total work:
// half the threads run heavy math, half run light math. In the uniform
// version the split follows warp boundaries (whole warps take one path); in
// the divergent version even/odd lanes of the SAME warp take different paths,
// so the hardware serializes both paths for every warp.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kN = 1 << 24;
constexpr int kHeavyIters = 200;
constexpr int kLightIters = 20;

__device__ float heavyPath(float v) {
  for (int t = 0; t < kHeavyIters; ++t) v = v * 1.0000001f + 0.5f;
  return v;
}

__device__ float lightPath(float v) {
  for (int t = 0; t < kLightIters; ++t) v = v * 0.9999999f - 0.25f;
  return v;
}

// Uniform: threads 0..31 of a warp all take the same branch, because the
// condition depends only on the warp index. No serialization.
__global__ void branchUniform(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int warp = i / kWarpSize;
  out[i] = (warp % 2 == 0) ? heavyPath(in[i]) : lightPath(in[i]);
}

// DIVERGENT on purpose: even and odd lanes of the same warp branch apart,
// so every warp executes heavyPath AND lightPath back to back.
__global__ void branchDivergent(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = (i % 2 == 0) ? heavyPath(in[i]) : lightPath(in[i]);
}

}  // namespace

int main() {
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  std::vector<float> host(kN);
  for (int i = 0; i < kN; ++i) host[i] = static_cast<float>(i % 7) * 0.1f;

  float* dev_in = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const int grid = (kN + kBlockSize - 1) / kBlockSize;
  const size_t moved = 2 * bytes;

  auto launch_uniform = [&]() { GPU_LAUNCH(branchUniform, grid, kBlockSize, 0, dev_in, dev_out, kN); };
  auto launch_divergent = [&]() { GPU_LAUNCH(branchDivergent, grid, kBlockSize, 0, dev_in, dev_out, kN); };

  std::printf("branch_divergence: same total work, warp-aligned vs lane-interleaved branch\n");
  gklab::report("uniform_by_warp", gklab::benchmarkKernel(launch_uniform, moved, 0.0), kPeakGbPerSec, 0.0);
  gklab::report("divergent_by_lane", gklab::benchmarkKernel(launch_divergent, moved, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(dev_in));
  GPU_CHECK(gpuFree(dev_out));
  return EXIT_SUCCESS;
}
