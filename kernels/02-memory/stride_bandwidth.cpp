// stride_bandwidth.cpp
//
// Measures achieved bandwidth of a strided read for stride = 1, 2, 4, ... to
// show how memory coalescing efficiency collapses as stride grows (chapter 6).
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

// Each thread reads one element at position (i * stride) modulo n and writes it
// out contiguously. Larger stride spreads reads across more cache segments.
__global__ void stridedCopy(const float* in, float* out, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[(static_cast<long long>(i) * stride) % n];
  }
}

}  // namespace

int main() {
  constexpr int kN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  const int strides[] = {1, 2, 4, 8, 16, 32};
  constexpr double kPeakGbPerSec = 1555.0;

  std::vector<float> host(kN);
  for (int i = 0; i < kN; ++i) host[i] = static_cast<float>(i % 101);
  std::vector<float> result(kN);

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, host.data(), bytes));

  const int grid = (kN + kBlockSize - 1) / kBlockSize;
  std::printf("achieved bandwidth vs access stride\n");
  for (int stride : strides) {
    auto launch = [&]() { GPU_LAUNCH(stridedCopy, grid, kBlockSize, 0, devIn, devOut, kN, stride); };

    // Verify stride = 1 against a straightforward reference.
    if (stride == 1) {
      launch();
      GPU_CHECK(gpuDeviceSynchronize());
      GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, bytes));
      if (!gklab::verifyClose(result, host)) return EXIT_FAILURE;
    }

    char label[48];
    std::snprintf(label, sizeof(label), "stride=%d", stride);
    gklab::report(label, gklab::benchmarkKernel(launch, 2 * bytes, 0.0), kPeakGbPerSec, 0.0);
  }

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
