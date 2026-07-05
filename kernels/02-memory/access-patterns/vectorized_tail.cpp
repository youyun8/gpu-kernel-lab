// vectorized_tail.cpp
//
// float4 vectorized SAXPY over a length that is NOT divisible by 4, showing
// the standard main-body + scalar-tail-cleanup decomposition. Compares against
// the plain scalar kernel and verifies both give identical results, including
// the last (n % 4) elements.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

__global__ void saxpyScalar(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

// Main body: each thread handles one float4 (16 bytes). Device allocations are
// at least 256-byte aligned, so reinterpreting the base pointer is safe.
__global__ void saxpyVec4Body(const float4* x, float4* y, float a, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 xv = x[i];
    float4 yv = y[i];
    yv.x = a * xv.x + yv.x;
    yv.y = a * xv.y + yv.y;
    yv.z = a * xv.z + yv.z;
    yv.w = a * xv.w + yv.w;
    y[i] = yv;
  }
}

// Tail cleanup: the last n % 4 elements are processed as scalars. Launched
// with a single tiny block; its cost is negligible next to the main body.
__global__ void saxpyTail(const float* x, float* y, float a, int tail_start, int n) {
  int i = tail_start + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

}  // namespace

int main() {
  constexpr int kSizeN = (1 << 24) + 3;  // deliberately not divisible by 4
  constexpr float kScaleA = 2.0f;
  const size_t bytes = static_cast<size_t>(kSizeN) * sizeof(float);

  std::vector<float> host_x(kSizeN), host_y(kSizeN), expected(kSizeN), result(kSizeN);
  for (int i = 0; i < kSizeN; ++i) {
    host_x[i] = static_cast<float>(i % 101);
    host_y[i] = static_cast<float>(i % 53);
    expected[i] = kScaleA * host_x[i] + host_y[i];
  }

  float* dev_x = nullptr;
  float* dev_y = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_x), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_y), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_x, host_x.data(), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const size_t moved = 3 * bytes;  // read x, read y, write y

  // Scalar baseline.
  GPU_CHECK(gpuMemcpyHostToDevice(dev_y, host_y.data(), bytes));
  const int grid_scalar = (kSizeN + kBlockSize - 1) / kBlockSize;
  auto launch_scalar = [&]() { GPU_LAUNCH(saxpyScalar, grid_scalar, kBlockSize, 0, dev_x, dev_y, kScaleA, kSizeN); };
  launch_scalar();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_y, bytes));
  // The benchmark loop re-runs the kernel, so verify against a single application only.
  if (!gklab::verifyClose(result, expected)) return EXIT_FAILURE;

  GPU_CHECK(gpuMemcpyHostToDevice(dev_y, host_y.data(), bytes));
  gklab::report("saxpy_scalar", gklab::benchmarkKernel(launch_scalar, moved, 0.0), kPeakGbPerSec, 0.0);

  // Vectorized body + scalar tail.
  const int n4 = kSizeN / 4;           // 4,194,304 float4 elements
  const int tail_start = n4 * 4;   // 3 leftover floats
  const int grid_vec = (n4 + kBlockSize - 1) / kBlockSize;
  auto launch_vec = [&]() {
    GPU_LAUNCH(saxpyVec4Body, grid_vec, kBlockSize, 0, reinterpret_cast<const float4*>(dev_x),
               reinterpret_cast<float4*>(dev_y), kScaleA, n4);
    GPU_LAUNCH(saxpyTail, 1, kBlockSize, 0, dev_x, dev_y, kScaleA, tail_start, kSizeN);
  };
  GPU_CHECK(gpuMemcpyHostToDevice(dev_y, host_y.data(), bytes));
  launch_vec();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_y, bytes));
  if (!gklab::verifyClose(result, expected)) return EXIT_FAILURE;

  GPU_CHECK(gpuMemcpyHostToDevice(dev_y, host_y.data(), bytes));
  gklab::report("saxpy_float4_tail", gklab::benchmarkKernel(launch_vec, moved, 0.0), kPeakGbPerSec, 0.0);

  std::printf("n = %d (tail of %d scalars handled by cleanup kernel)\n", kSizeN, kSizeN - tail_start);

  GPU_CHECK(gpuFree(dev_x));
  GPU_CHECK(gpuFree(dev_y));
  return EXIT_SUCCESS;
}
