// ex_a_saxpy.cpp
//
// Reference solution for exercise A7: implement SAXPY (y = a*x + y) using the
// vector_add.cpp template, with a CPU correctness check and bandwidth report.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

// One thread per element; one FMA per element.
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
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
    host_x[i] = static_cast<float>(i % 89) * 0.3f;
    host_y[i] = static_cast<float>(i % 41) * 0.7f;
    reference[i] = kScaleA * host_x[i] + host_y[i];
  }
  std::vector<float> result(kSizeN);

  float* dev_x = nullptr;
  float* dev_y = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_x), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_y), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_x, host_x.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_y, host_y.data(), bytes));

  const int grid = (kSizeN + kBlockSize - 1) / kBlockSize;
  auto launch = [&]() { GPU_LAUNCH(saxpy, grid, kBlockSize, 0, dev_x, dev_y, kScaleA, kSizeN); };

  launch();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_y, bytes));
  if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;

  // SAXPY reads x, reads y, writes y -> 3 arrays; 2 FLOP per element.
  const size_t bytes_moved = 3 * bytes;
  const double flops = static_cast<double>(kSizeN) * 2.0;
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  // Note: benchmarkKernel re-runs the kernel in place; y accumulates, but the
  // reported bandwidth/throughput is unaffected. Correctness was checked above.
  gklab::report("ex_a_saxpy", gklab::benchmarkKernel(launch, bytes_moved, flops), kPeakGbPerSec,
                kPeakGflopPerSec);

  GPU_CHECK(gpuFree(dev_x));
  GPU_CHECK(gpuFree(dev_y));
  return EXIT_SUCCESS;
}
