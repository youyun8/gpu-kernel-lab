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

// flops_per_elem controls arithmetic intensity: 2 FLOP per FMA iteration.
__global__ void variableIntensity(const float* in, float* out, int n, int fma_iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = in[i];
  for (int t = 0; t < fma_iters; ++t) {
    acc = acc * 1.0000001f + 0.5f;
  }
  out[i] = acc;
}

}  // namespace

int main() {
  constexpr int kSizeN = 1 << 22;
  const size_t bytes = static_cast<size_t>(kSizeN) * sizeof(float);
  const int fma_counts[] = {1, 4, 16, 64, 256};
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;

  std::vector<float> host(kSizeN);
  for (int i = 0; i < kSizeN; ++i) host[i] = static_cast<float>(i % 5) * 0.1f;

  float* dev_in = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), bytes));

  const int grid = (kSizeN + kBlockSize - 1) / kBlockSize;
  std::printf("throughput vs arithmetic intensity\n");
  for (int iters : fma_counts) {
    auto launch = [&]() { GPU_LAUNCH(variableIntensity, grid, kBlockSize, 0, dev_in, dev_out, kSizeN, iters); };
    const size_t bytes_moved = 2 * bytes;  // one read + one write
    const double flops = static_cast<double>(kSizeN) * iters * 2.0;
    const double ai = flops / static_cast<double>(bytes_moved);
    char label[48];
    std::snprintf(label, sizeof(label), "AI=%.2f FLOP/byte", ai);
    gklab::report(label, gklab::benchmarkKernel(launch, bytes_moved, flops), kPeakGbPerSec, kPeakGflopPerSec);
  }

  GPU_CHECK(gpuFree(dev_in));
  GPU_CHECK(gpuFree(dev_out));
  return EXIT_SUCCESS;
}
