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
  constexpr int kSizeN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kSizeN) * sizeof(float);
  constexpr float kScaleA = 2.5f;

  std::vector<float> host_x(kSizeN);
  std::vector<float> host_y(kSizeN);
  std::vector<float> reference(kSizeN);
  for (int i = 0; i < kSizeN; ++i) {
    host_x[i] = static_cast<float>(i % 91) * 0.3f;
    host_y[i] = static_cast<float>(i % 47) * 0.7f;
    reference[i] = kScaleA * host_x[i] + host_y[i];
  }
  std::vector<float> result(kSizeN);

  float* dev_x = nullptr;
  float* dev_y = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_x), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_y), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const size_t bytes_moved = 3 * bytes;  // read x, read y, write y
  const double flops = static_cast<double>(kSizeN) * 2.0;

  auto reset_y = [&]() { GPU_CHECK(gpuMemcpyHostToDevice(dev_y, host_y.data(), bytes)); };
  GPU_CHECK(gpuMemcpyHostToDevice(dev_x, host_x.data(), bytes));

  // Unroll factor 1.
  {
    const int grid = (kSizeN + kBlockSize - 1) / kBlockSize;
    reset_y();
    auto launch = [&]() { GPU_LAUNCH(saxpy<1>, grid, kBlockSize, 0, dev_x, dev_y, kScaleA, kSizeN); };
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_y, bytes));
    if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
    reset_y();
    gklab::report("saxpy_unroll1", gklab::benchmarkKernel(launch, bytes_moved, flops), kPeakGbPerSec, 0.0);
  }

  // Unroll factor 4: grid covers a quarter of the elements per launch pass.
  {
    const int grid = (kSizeN / 4 + kBlockSize - 1) / kBlockSize;
    reset_y();
    auto launch = [&]() { GPU_LAUNCH(saxpy<4>, grid, kBlockSize, 0, dev_x, dev_y, kScaleA, kSizeN); };
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_y, bytes));
    if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
    reset_y();
    gklab::report("saxpy_unroll4", gklab::benchmarkKernel(launch, bytes_moved, flops), kPeakGbPerSec, 0.0);
  }

  GPU_CHECK(gpuFree(dev_x));
  GPU_CHECK(gpuFree(dev_y));
  return EXIT_SUCCESS;
}
