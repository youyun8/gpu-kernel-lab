// device_query.cpp
//
// Prints the hardware numbers chapter 1 builds intuition around: how many
// SMs/CUs the GPU has and how many threads it takes to fill them. Run this
// first on any new machine to calibrate "how much parallelism do I need".
#include <cstdio>

#include "gpu_portability.h"

int main() {
  int device_count = 0;
  GPU_CHECK(gpuGetDeviceCount(&device_count));
  std::printf("devices: %d\n", device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    GPU_CHECK(gpuSetDevice(dev));
    int sm_count = 0;
    GPU_CHECK(gpuGetMultiprocessorCount(&sm_count));
    // 2048 resident threads per SM/CU is a common ceiling; the exact number
    // comes from the occupancy chapter (8).
    std::printf("device %d: %d SM/CU, ~%d threads to fill at 2048 resident/SM\n",
                dev, sm_count, sm_count * 2048);
  }
  return 0;
}
