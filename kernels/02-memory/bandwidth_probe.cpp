// bandwidth_probe.cpp
//
// Streaming copy over a range of buffer sizes. Small buffers hit L1/L2 and show
// apparent bandwidth above HBM peak; large buffers reveal true HBM bandwidth.
// This makes the cache-vs-global-memory boundary visible (chapter 3).
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

__global__ void copyKernel(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

}  // namespace

int main() {
  const int sizes[] = {1 << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 26};
  constexpr double kPeakGbPerSec = 1555.0;

  std::printf("streaming copy bandwidth vs buffer size\n");
  for (int n : sizes) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> host(n, 1.0f);
    std::vector<float> result(n);

    float* dev_in = nullptr;
    float* dev_out = nullptr;
    GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), bytes));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), bytes));
    GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), bytes));

    const int grid = (n + kBlockSize - 1) / kBlockSize;
    auto launch = [&]() { GPU_LAUNCH(copyKernel, grid, kBlockSize, 0, dev_in, dev_out, n); };

    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, bytes));
    if (!gklab::verifyClose(result, host)) return EXIT_FAILURE;

    char label[64];
    std::snprintf(label, sizeof(label), "copy n=2^%d", static_cast<int>(std::log2(static_cast<double>(n))));
    gklab::report(label, gklab::benchmarkKernel(launch, 2 * bytes, 0.0), kPeakGbPerSec, 0.0);

    GPU_CHECK(gpuFree(dev_in));
    GPU_CHECK(gpuFree(dev_out));
  }
  return EXIT_SUCCESS;
}
