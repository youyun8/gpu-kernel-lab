// roofline_probe.cpp
//
// Sweeps arithmetic intensity by doing a variable number of FMAs per loaded
// element. Low-AI kernels sit on the bandwidth roof; high-AI kernels approach
// the compute roof (chapter 5).
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

// flopsPerElem controls arithmetic intensity: 2 FLOP per FMA iteration.
__global__ void variableIntensity(const float* in, float* out, int n, int fmaIters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = in[i];
  for (int t = 0; t < fmaIters; ++t) {
    acc = acc * 1.0000001f + 0.5f;
  }
  out[i] = acc;
}

}  // namespace

int main() {
  constexpr int kN = 1 << 22;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  const int fmaCounts[] = {1, 4, 16, 64, 256};
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;

  std::vector<float> host(kN);
  for (int i = 0; i < kN; ++i) host[i] = static_cast<float>(i % 5) * 0.1f;

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, host.data(), bytes));

  const int grid = (kN + kBlockSize - 1) / kBlockSize;
  std::printf("throughput vs arithmetic intensity\n");
  for (int iters : fmaCounts) {
    auto launch = [&]() { GPU_LAUNCH(variableIntensity, grid, kBlockSize, 0, devIn, devOut, kN, iters); };
    const size_t bytesMoved = 2 * bytes;  // one read + one write
    const double flops = static_cast<double>(kN) * iters * 2.0;
    const double ai = flops / static_cast<double>(bytesMoved);
    char label[48];
    std::snprintf(label, sizeof(label), "AI=%.2f FLOP/byte", ai);
    gklab::report(label, gklab::benchmarkKernel(launch, bytesMoved, flops), kPeakGbPerSec, kPeakGflopPerSec);
  }

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
