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
  constexpr int kN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  constexpr float kA = 2.5f;

  std::vector<float> hostX(kN);
  std::vector<float> hostY(kN);
  std::vector<float> reference(kN);
  for (int i = 0; i < kN; ++i) {
    hostX[i] = static_cast<float>(i % 89) * 0.3f;
    hostY[i] = static_cast<float>(i % 41) * 0.7f;
    reference[i] = kA * hostX[i] + hostY[i];
  }
  std::vector<float> result(kN);

  float* devX = nullptr;
  float* devY = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devX), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devY), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devX, hostX.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devY, hostY.data(), bytes));

  const int grid = (kN + kBlockSize - 1) / kBlockSize;
  auto launch = [&]() { GPU_LAUNCH(saxpy, grid, kBlockSize, 0, devX, devY, kA, kN); };

  launch();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devY, bytes));
  if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;

  // SAXPY reads x, reads y, writes y -> 3 arrays; 2 FLOP per element.
  const size_t bytesMoved = 3 * bytes;
  const double flops = static_cast<double>(kN) * 2.0;
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  // Note: benchmarkKernel re-runs the kernel in place; y accumulates, but the
  // reported bandwidth/throughput is unaffected. Correctness was checked above.
  gklab::report("ex_a_saxpy", gklab::benchmarkKernel(launch, bytesMoved, flops), kPeakGbPerSec,
                kPeakGflopPerSec);

  GPU_CHECK(gpuFree(devX));
  GPU_CHECK(gpuFree(devY));
  return EXIT_SUCCESS;
}
