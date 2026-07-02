// occupancy_experiment.cpp
//
// Demonstrates the "higher occupancy is not always faster" idea from chapter 8.
// The same compute-heavy workload is run two ways:
//   1. High occupancy: small per-thread work, many threads.
//   2. Low occupancy + high ILP: each thread computes many independent outputs
//      held in registers, using __launch_bounds__ to cap register pressure.
// Both are timed; on many GPUs the high-ILP variant wins despite lower occupancy.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kIlpFactor = 8;      // independent accumulators per thread
constexpr int kInnerIters = 512;   // arithmetic to make the kernel compute-heavy

// High-occupancy variant: one output per thread.
__global__ void __launch_bounds__(kBlockSize)
    computeHighOccupancy(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = in[i];
  for (int t = 0; t < kInnerIters; ++t) {
    acc = acc * 1.0000001f + 0.5f;  // dependent FMA chain
  }
  out[i] = acc;
}

// Low-occupancy + high-ILP variant: each thread owns kIlpFactor independent
// accumulators so the scheduler has independent work to hide latency.
__global__ void __launch_bounds__(kBlockSize, 1)
    computeHighIlp(const float* in, float* out, int n) {
  int base = (blockIdx.x * blockDim.x + threadIdx.x) * kIlpFactor;
  float acc[kIlpFactor];
#pragma unroll
  for (int j = 0; j < kIlpFactor; ++j) {
    acc[j] = (base + j < n) ? in[base + j] : 0.0f;
  }
  for (int t = 0; t < kInnerIters; ++t) {
#pragma unroll
    for (int j = 0; j < kIlpFactor; ++j) {
      acc[j] = acc[j] * 1.0000001f + 0.5f;  // independent chains -> ILP
    }
  }
#pragma unroll
  for (int j = 0; j < kIlpFactor; ++j) {
    if (base + j < n) out[base + j] = acc[j];
  }
}

float cpuReference(float x) {
  float acc = x;
  for (int t = 0; t < kInnerIters; ++t) acc = acc * 1.0000001f + 0.5f;
  return acc;
}

}  // namespace

int main() {
  constexpr int kN = 1 << 22;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);

  std::vector<float> hostIn(kN);
  std::vector<float> hostOut(kN);
  std::vector<float> reference(kN);
  for (int i = 0; i < kN; ++i) {
    hostIn[i] = static_cast<float>(i % 7) * 0.25f;
    reference[i] = cpuReference(hostIn[i]);
  }

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, hostIn.data(), bytes));

  const int gridHigh = (kN + kBlockSize - 1) / kBlockSize;
  const int gridIlp = (kN / kIlpFactor + kBlockSize - 1) / kBlockSize;

  auto launchHigh = [&]() { GPU_LAUNCH(computeHighOccupancy, gridHigh, kBlockSize, 0, devIn, devOut, kN); };
  auto launchIlp = [&]() { GPU_LAUNCH(computeHighIlp, gridIlp, kBlockSize, 0, devIn, devOut, kN); };

  // Each element runs kInnerIters iterations of 2 FLOP (one FMA).
  const double flops = static_cast<double>(kN) * kInnerIters * 2.0;
  const size_t bytesMoved = 2 * bytes;

  launchHigh();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(hostOut.data(), devOut, bytes));
  if (!gklab::verifyClose(hostOut, reference, 1.0e-2f)) return EXIT_FAILURE;

  launchIlp();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(hostOut.data(), devOut, bytes));
  if (!gklab::verifyClose(hostOut, reference, 1.0e-2f)) return EXIT_FAILURE;

  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  gklab::report("high_occupancy", gklab::benchmarkKernel(launchHigh, bytesMoved, flops), kPeakGbPerSec, kPeakGflopPerSec);
  gklab::report("low_occ_high_ilp", gklab::benchmarkKernel(launchIlp, bytesMoved, flops), kPeakGbPerSec, kPeakGflopPerSec);

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
