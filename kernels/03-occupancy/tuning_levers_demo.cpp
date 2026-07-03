// tuning_levers_demo.cpp
//
// Companion to chapter b8 ("Occupancy 與 Latency Hiding"). It turns the tuning
// levers from that chapter into one runnable experiment so the reader can see
// the numbers behind each knob on their own GPU instead of trusting the prose:
//
//   1. Block size   - query achieved occupancy for 128 / 256 / 512 and time each.
//   2. Register cap - the same kernel with and without __launch_bounds__ so the
//                     resident-block count (and thus occupancy) changes.
//   3. Tile size    - 1 output per thread (high occupancy) vs kTile outputs per
//                     thread (fewer warps, but more reuse / ILP).
//   4. Accumulators - a single dependent chain vs kTile independent chains.
//
// Read it alongside the compiler's resource report:
//   HIP:  hipcc -Rpass-analysis=kernel-resource-usage tuning_levers_demo.cpp
//   CUDA: nvcc  -Xptxas=-v                              tuning_levers_demo.cpp
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kWarpSize =
#if defined(USE_HIP)
    64;  // AMD wavefront
#else
    32;  // NVIDIA warp
#endif

constexpr int kInnerIters = 1024;  // arithmetic so the kernel is compute-bound
constexpr int kTile = 8;           // outputs-per-thread for the low-occupancy variant

// One output per thread, single dependent FMA chain. Latency hiding here relies
// entirely on having many resident warps -> this kernel wants high occupancy.
__global__ void computeScalar(const float* in, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = in[i];
  for (int t = 0; t < kInnerIters; ++t) acc = acc * 1.0000001f + 0.5f;
  out[i] = acc;
}

// Same math, but capped to fit at least 3 blocks per SM. The compiler now trades
// per-thread registers for residency; watch for spills in the resource report.
__global__ void __launch_bounds__(256, 3) computeScalarCapped(const float* in, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = in[i];
  for (int t = 0; t < kInnerIters; ++t) acc = acc * 1.0000001f + 0.5f;
  out[i] = acc;
}

// kTile outputs per thread held in registers = kTile independent chains. A single
// warp now issues kTile FMAs back to back, so this variant hides latency with ILP
// and needs far fewer warps -> it deliberately runs at lower occupancy.
__global__ void __launch_bounds__(256, 1) computeTiled(const float* in, float* out, int n) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kTile;
  float acc[kTile];
#pragma unroll
  for (int j = 0; j < kTile; ++j) acc[j] = (base + j < n) ? in[base + j] : 0.0f;
  for (int t = 0; t < kInnerIters; ++t) {
#pragma unroll
    for (int j = 0; j < kTile; ++j) acc[j] = acc[j] * 1.0000001f + 0.5f;
  }
#pragma unroll
  for (int j = 0; j < kTile; ++j) {
    if (base + j < n) out[base + j] = acc[j];
  }
}

float cpuReference(float x) {
  float acc = x;
  for (int t = 0; t < kInnerIters; ++t) acc = acc * 1.0000001f + 0.5f;
  return acc;
}

// Print theoretical occupancy for a kernel at a given block size, using the
// runtime's resource-aware estimate (same math the chapter table walks by hand).
template <typename KernelFn>
void reportOccupancy(const char* label, KernelFn kernel, int block_size, int max_threads_per_sm) {
  int max_blocks = 0;
  GPU_CHECK(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, 0));
  const int active_warps = max_blocks * (block_size / kWarpSize);
  const int max_warps = max_threads_per_sm / kWarpSize;
  const double occ = max_warps > 0 ? 100.0 * active_warps / max_warps : 0.0;
  std::printf("  %-26s block=%4d -> %2d blocks/SM, %2d/%2d warps (%.0f%% occupancy)\n", label,
              block_size, max_blocks, active_warps, max_warps, occ);
}

}  // namespace

int main() {
  constexpr int kN = 1 << 22;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);

  int max_threads_per_sm = 0;
  GPU_CHECK(gpuGetMaxThreadsPerMultiprocessor(&max_threads_per_sm));

  std::vector<float> hostIn(kN), hostOut(kN), reference(kN);
  for (int i = 0; i < kN; ++i) {
    hostIn[i] = static_cast<float>(i % 7) * 0.25f;
    reference[i] = cpuReference(hostIn[i]);
  }

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, hostIn.data(), bytes));

  const double flops = static_cast<double>(kN) * kInnerIters * 2.0;
  const size_t bytesMoved = 2 * bytes;

  std::printf("tuning_levers_demo: %d threads/SM max, warp/wave = %d\n\n", max_threads_per_sm,
              kWarpSize);

  // --- Lever 1: block size. Occupancy from the runtime for 128/256/512. -------
  std::printf("[lever 1] block size vs occupancy (scalar kernel):\n");
  for (int bs : {128, 256, 512}) reportOccupancy("computeScalar", computeScalar, bs, max_threads_per_sm);

  // --- Lever 2: register cap via __launch_bounds__. ---------------------------
  std::printf("\n[lever 2] register cap (__launch_bounds__) at block=256:\n");
  reportOccupancy("computeScalar (uncapped)", computeScalar, 256, max_threads_per_sm);
  reportOccupancy("computeScalarCapped(,3)", computeScalarCapped, 256, max_threads_per_sm);

  // --- Lever 3/4: tile size + independent accumulators. -----------------------
  std::printf("\n[lever 3+4] tile size / independent accumulators at block=256:\n");
  reportOccupancy("computeScalar (1/thread)", computeScalar, 256, max_threads_per_sm);
  reportOccupancy("computeTiled  (kTile/thr)", computeTiled, 256, max_threads_per_sm);

  // --- Timed comparison: high occupancy vs low occupancy + high ILP. ----------
  constexpr int kBlock = 256;
  const int gridScalar = (kN + kBlock - 1) / kBlock;
  const int gridTiled = (kN / kTile + kBlock - 1) / kBlock;
  auto launchScalar = [&]() { GPU_LAUNCH(computeScalar, gridScalar, kBlock, 0, devIn, devOut, kN); };
  auto launchTiled = [&]() { GPU_LAUNCH(computeTiled, gridTiled, kBlock, 0, devIn, devOut, kN); };

  launchScalar();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(hostOut.data(), devOut, bytes));
  if (!gklab::verifyClose(hostOut, reference, 1.0e-2f)) return EXIT_FAILURE;

  launchTiled();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(hostOut.data(), devOut, bytes));
  if (!gklab::verifyClose(hostOut, reference, 1.0e-2f)) return EXIT_FAILURE;

  // Illustrative peaks; replace with your GPU's numbers for real % of peak.
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  std::printf("\n[timing] does lower occupancy + higher ILP win here?\n");
  gklab::report("high_occupancy_scalar", gklab::benchmarkKernel(launchScalar, bytesMoved, flops),
                kPeakGbPerSec, kPeakGflopPerSec);
  gklab::report("low_occ_high_ilp_tiled", gklab::benchmarkKernel(launchTiled, bytesMoved, flops),
                kPeakGbPerSec, kPeakGflopPerSec);

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
