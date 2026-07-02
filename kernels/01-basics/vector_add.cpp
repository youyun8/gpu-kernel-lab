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
  constexpr int kN = 1 << 24;  // ~16M elements
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);

  std::vector<float> hostA(kN);
  std::vector<float> hostB(kN);
  std::vector<float> hostC(kN);
  std::vector<float> reference(kN);
  for (int i = 0; i < kN; ++i) {
    hostA[i] = static_cast<float>(i % 97) * 0.5f;
    hostB[i] = static_cast<float>(i % 13) * 1.5f;
    reference[i] = hostA[i] + hostB[i];
  }

  float* devA = nullptr;
  float* devB = nullptr;
  float* devC = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devA), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devB), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devC), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devA, hostA.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devB, hostB.data(), bytes));

  const int grid = (kN + kBlockSize - 1) / kBlockSize;
  auto launch = [&]() { GPU_LAUNCH(vectorAdd, grid, kBlockSize, 0, devA, devB, devC, kN); };

  launch();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(hostC.data(), devC, bytes));
  if (!gklab::verifyClose(hostC, reference)) {
    return EXIT_FAILURE;
  }

  // Vector add moves 3 arrays (2 read + 1 write) and does 1 FLOP per element.
  const size_t bytesMoved = 3 * bytes;
  const double flops = static_cast<double>(kN);
  const gklab::BenchResult result = gklab::benchmarkKernel(launch, bytesMoved, flops);

  // Replace peaks with your device's numbers for accurate % of peak.
  constexpr double kIllustrativePeakGbPerSec = 1555.0;
  constexpr double kIllustrativePeakGflopPerSec = 19500.0;
  gklab::report("vector_add", result, kIllustrativePeakGbPerSec, kIllustrativePeakGflopPerSec);

  GPU_CHECK(gpuFree(devA));
  GPU_CHECK(gpuFree(devB));
  GPU_CHECK(gpuFree(devC));
  return EXIT_SUCCESS;
}
