// ex_c_tail_correctness.cpp
//
// Reference solution for a tail-shape correctness exercise: implement SAXPY as
// a grid-stride kernel and validate edge sizes around warp and block
// boundaries.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr float kA = 1.75f;

__global__ void saxpyGridStride(const float* x, float* y, float a, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

bool runCase(int n) {
  std::vector<float> x(std::max(1, n));
  std::vector<float> y(std::max(1, n));
  std::vector<float> reference(std::max(1, n));
  std::vector<float> result(std::max(1, n));
  for (int i = 0; i < std::max(1, n); ++i) {
    x[i] = static_cast<float>((i % 17) - 8) * 0.5f;
    y[i] = static_cast<float>((i % 13) - 6) * 0.25f;
    reference[i] = (i < n) ? (kA * x[i] + y[i]) : y[i];
  }

  float* devX = nullptr;
  float* devY = nullptr;
  const size_t bytes = static_cast<size_t>(std::max(1, n)) * sizeof(float);
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devX), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devY), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devX, x.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devY, y.data(), bytes));

  int grid = std::max(1, (n + kBlockSize - 1) / kBlockSize);
  GPU_LAUNCH(saxpyGridStride, grid, kBlockSize, 0, devX, devY, kA, n);
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devY, bytes));

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    if (result[i] != reference[i]) {
      std::fprintf(stderr, "n=%d mismatch at %d: got %.6f want %.6f\n", n, i, result[i],
                   reference[i]);
      ok = false;
      break;
    }
  }

  GPU_CHECK(gpuFree(devX));
  GPU_CHECK(gpuFree(devY));
  return ok;
}

}  // namespace

int main() {
  const int sizes[] = {0, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025};
  for (int n : sizes) {
    if (!runCase(n)) return EXIT_FAILURE;
  }
  std::printf("tail-shape correctness OK (%zu cases)\n", sizeof(sizes) / sizeof(sizes[0]));

  constexpr int kN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  std::vector<float> x(kN);
  std::vector<float> y(kN);
  for (int i = 0; i < kN; ++i) {
    x[i] = static_cast<float>(i % 97) * 0.125f;
    y[i] = static_cast<float>(i % 53) * 0.25f;
  }

  float* devX = nullptr;
  float* devY = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devX), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devY), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devX, x.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devY, y.data(), bytes));

  int grid = (kN + kBlockSize - 1) / kBlockSize;
  auto launch = [&]() { GPU_LAUNCH(saxpyGridStride, grid, kBlockSize, 0, devX, devY, kA, kN); };
  const size_t bytesMoved = 3 * bytes;
  const double flops = 2.0 * kN;
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  gklab::report("ex_c_tail_saxpy", gklab::benchmarkKernel(launch, bytesMoved, flops), kPeakGbPerSec,
                kPeakGflopPerSec);

  GPU_CHECK(gpuFree(devX));
  GPU_CHECK(gpuFree(devY));
  return EXIT_SUCCESS;
}
