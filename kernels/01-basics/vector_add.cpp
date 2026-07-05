// vector_add.cpp
//
// The canonical first kernel: C = A + B. Demonstrates the thread/block/grid
// launch pattern, a CPU correctness check, and the shared benchmark harness
// reporting achieved bandwidth. Vector add is purely memory-bound.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

// One thread per element, with a boundary guard for the tail block.
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

int main() {
  constexpr int kSizeN = 1 << 24;  // ~16M elements
  const size_t bytes = static_cast<size_t>(kSizeN) * sizeof(float);

  std::vector<float> host_a(kSizeN);
  std::vector<float> host_b(kSizeN);
  std::vector<float> host_c(kSizeN);
  std::vector<float> reference(kSizeN);
  for (int i = 0; i < kSizeN; ++i) {
    host_a[i] = static_cast<float>(i % 97) * 0.5f;
    host_b[i] = static_cast<float>(i % 13) * 1.5f;
    reference[i] = host_a[i] + host_b[i];
  }

  float* dev_a = nullptr;
  float* dev_b = nullptr;
  float* dev_c = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_a), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_b), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_c), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_a, host_a.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_b, host_b.data(), bytes));

  const int grid = (kSizeN + kBlockSize - 1) / kBlockSize;
  auto launch = [&]() { GPU_LAUNCH(vectorAdd, grid, kBlockSize, 0, dev_a, dev_b, dev_c, kSizeN); };

  launch();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(host_c.data(), dev_c, bytes));
  if (!gklab::verifyClose(host_c, reference)) {
    return EXIT_FAILURE;
  }

  // Vector add moves 3 arrays (2 read + 1 write) and does 1 FLOP per element.
  const size_t bytes_moved = 3 * bytes;
  const double flops = static_cast<double>(kSizeN);
  const gklab::BenchResult result = gklab::benchmarkKernel(launch, bytes_moved, flops);

  // Replace peaks with your device's numbers for accurate % of peak.
  constexpr double kIllustrativePeakGbPerSec = 1555.0;
  constexpr double kIllustrativePeakGflopPerSec = 19500.0;
  gklab::report("vector_add", result, kIllustrativePeakGbPerSec, kIllustrativePeakGflopPerSec);

  GPU_CHECK(gpuFree(dev_a));
  GPU_CHECK(gpuFree(dev_b));
  GPU_CHECK(gpuFree(dev_c));
  return EXIT_SUCCESS;
}
