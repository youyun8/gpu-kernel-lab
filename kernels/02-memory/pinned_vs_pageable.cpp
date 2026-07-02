// pinned_vs_pageable.cpp
//
// Times host-to-device copies from pageable (new/malloc) memory vs pinned
// (gpuMallocHost) memory. Pageable copies force the driver to stage through
// an internal pinned buffer, so they are slower and can never overlap with
// compute; pinned copies DMA directly and enable async transfer (chapter 28).
#include <chrono>
#include <cstdio>
#include <vector>

#include "gpu_portability.h"

namespace {

constexpr size_t kBytes = size_t{256} << 20;  // 256 MiB per copy
constexpr int kTrials = 5;

double copyGbPerSec(void* dst, const void* src, size_t bytes) {
  // Warmup, then time kTrials synchronous H2D copies.
  GPU_CHECK(gpuMemcpyHostToDevice(dst, src, bytes));
  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < kTrials; ++t) {
    GPU_CHECK(gpuMemcpyHostToDevice(dst, src, bytes));
  }
  auto stop = std::chrono::steady_clock::now();
  double seconds = std::chrono::duration<double>(stop - start).count() / kTrials;
  return static_cast<double>(bytes) / seconds / 1e9;
}

}  // namespace

int main() {
  float* dev = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev), kBytes));

  // Pageable host memory: an ordinary heap allocation.
  std::vector<float> pageable(kBytes / sizeof(float), 1.0f);

  // Pinned host memory: page-locked, DMA-able.
  float* pinned = nullptr;
  GPU_CHECK(gpuMallocHost(reinterpret_cast<void**>(&pinned), kBytes));
  for (size_t i = 0; i < kBytes / sizeof(float); ++i) pinned[i] = 1.0f;

  std::printf("H2D copy of %zu MiB, average of %d trials\n", kBytes >> 20, kTrials);
  std::printf("  pageable: %6.1f GB/s\n", copyGbPerSec(dev, pageable.data(), kBytes));
  std::printf("  pinned:   %6.1f GB/s\n", copyGbPerSec(dev, pinned, kBytes));

  GPU_CHECK(gpuFreeHost(pinned));
  GPU_CHECK(gpuFree(dev));
  return 0;
}
