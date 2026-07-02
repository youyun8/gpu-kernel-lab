// vectorized_copy.cpp
//
// Compares scalar float copy against float4 (128-bit) vectorized copy to show
// the instruction-count / bandwidth benefit of vectorized access (chapter 11).
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

__global__ void copyScalar(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

// Each thread moves 128 bits per load/store. Requires n divisible by 4 and
// 16-byte aligned buffers (device allocations satisfy the alignment).
__global__ void copyVec4(const float4* in, float4* out, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) out[i] = in[i];
}

}  // namespace

int main() {
  constexpr int kN = 1 << 24;  // divisible by 4
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);

  std::vector<float> host(kN);
  for (int i = 0; i < kN; ++i) host[i] = static_cast<float>(i % 97);
  std::vector<float> result(kN);

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, host.data(), bytes));

  constexpr double kPeakGbPerSec = 1555.0;

  const int gridScalar = (kN + kBlockSize - 1) / kBlockSize;
  auto launchScalar = [&]() { GPU_LAUNCH(copyScalar, gridScalar, kBlockSize, 0, devIn, devOut, kN); };
  launchScalar();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, bytes));
  if (!gklab::verifyClose(result, host)) return EXIT_FAILURE;
  gklab::report("copy_scalar", gklab::benchmarkKernel(launchScalar, 2 * bytes, 0.0), kPeakGbPerSec, 0.0);

  const int n4 = kN / 4;
  const int gridVec = (n4 + kBlockSize - 1) / kBlockSize;
  auto launchVec = [&]() {
    GPU_LAUNCH(copyVec4, gridVec, kBlockSize, 0, reinterpret_cast<const float4*>(devIn),
               reinterpret_cast<float4*>(devOut), n4);
  };
  launchVec();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, bytes));
  if (!gklab::verifyClose(result, host)) return EXIT_FAILURE;
  gklab::report("copy_float4", gklab::benchmarkKernel(launchVec, 2 * bytes, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
