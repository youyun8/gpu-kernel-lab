// saxpy_unroll.cpp
//
// SAXPY (y = a*x + y) with different grid-stride unroll factors to explore the
// effect of loop unrolling / ILP on a memory-bound kernel (chapter 11).
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

template <int Unroll>
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int stride = blockDim.x * gridDim.x;
  int start = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int u = 0; u < Unroll; ++u) {
    int i = start + u * stride;
    if (i < n) y[i] = a * x[i] + y[i];
  }
}

}  // namespace

int main() {
  constexpr int kN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  constexpr float kA = 2.5f;

  std::vector<float> hostX(kN);
  std::vector<float> hostY(kN);
  std::vector<float> reference(kN);
  for (int i = 0; i < kN; ++i) {
    hostX[i] = static_cast<float>(i % 91) * 0.3f;
    hostY[i] = static_cast<float>(i % 47) * 0.7f;
    reference[i] = kA * hostX[i] + hostY[i];
  }
  std::vector<float> result(kN);

  float* devX = nullptr;
  float* devY = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devX), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devY), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const size_t bytesMoved = 3 * bytes;  // read x, read y, write y
  const double flops = static_cast<double>(kN) * 2.0;

  auto resetY = [&]() { GPU_CHECK(gpuMemcpyHostToDevice(devY, hostY.data(), bytes)); };
  GPU_CHECK(gpuMemcpyHostToDevice(devX, hostX.data(), bytes));

  // Unroll factor 1.
  {
    const int grid = (kN + kBlockSize - 1) / kBlockSize;
    resetY();
    auto launch = [&]() { GPU_LAUNCH(saxpy<1>, grid, kBlockSize, 0, devX, devY, kA, kN); };
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devY, bytes));
    if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
    resetY();
    gklab::report("saxpy_unroll1", gklab::benchmarkKernel(launch, bytesMoved, flops), kPeakGbPerSec, 0.0);
  }

  // Unroll factor 4: grid covers a quarter of the elements per launch pass.
  {
    const int grid = (kN / 4 + kBlockSize - 1) / kBlockSize;
    resetY();
    auto launch = [&]() { GPU_LAUNCH(saxpy<4>, grid, kBlockSize, 0, devX, devY, kA, kN); };
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devY, bytes));
    if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
    resetY();
    gklab::report("saxpy_unroll4", gklab::benchmarkKernel(launch, bytesMoved, flops), kPeakGbPerSec, 0.0);
  }

  GPU_CHECK(gpuFree(devX));
  GPU_CHECK(gpuFree(devY));
  return EXIT_SUCCESS;
}
