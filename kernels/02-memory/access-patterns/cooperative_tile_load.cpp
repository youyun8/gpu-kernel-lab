// cooperative_tile_load.cpp
//
// 1D box-filter stencil (radius 8): each output needs 17 neighboring inputs.
// The naive kernel lets every thread issue its own 17 global loads; the
// cooperative kernel has the block stage tile + halo into shared memory once
// (coalesced), then everyone reads neighbors from shared memory. Note the two
// __syncthreads() boundaries: one after the load phase, one is NOT needed at
// the end because each thread only reads what the whole block wrote before
// the first barrier.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kRadius = 8;
constexpr int kN = 1 << 24;

// BASELINE: every thread reads its full 17-element window from global memory.
// Neighboring threads re-read mostly the same elements; L1/L2 absorb a lot of
// it, but the load instruction count is 17x the cooperative version.
__global__ void boxFilterNaive(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sum = 0.0f;
  for (int r = -kRadius; r <= kRadius; ++r) {
    int j = i + r;
    if (j >= 0 && j < n) sum += in[j];
  }
  out[i] = sum / (2 * kRadius + 1);
}

// COOPERATIVE: the block loads kBlockSize + 2*kRadius elements into shared
// memory with two coalesced phases, then computes purely from shared memory.
__global__ void boxFilterShared(const float* in, float* out, int n) {
  __shared__ float tile[kBlockSize + 2 * kRadius];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int block_start = blockIdx.x * blockDim.x;

  // Phase 1: main tile, one coalesced load per thread.
  tile[kRadius + threadIdx.x] = (i < n) ? in[i] : 0.0f;

  // Phase 2: the first 2*kRadius threads fetch the left and right halos.
  if (threadIdx.x < kRadius) {
    int left = block_start - kRadius + static_cast<int>(threadIdx.x);
    tile[threadIdx.x] = (left >= 0) ? in[left] : 0.0f;
    int right = block_start + kBlockSize + static_cast<int>(threadIdx.x);
    tile[kRadius + kBlockSize + threadIdx.x] = (right < n) ? in[right] : 0.0f;
  }

  // Barrier: no thread may read tile[] until every thread finished writing it.
  __syncthreads();

  if (i >= n) return;
  float sum = 0.0f;
  for (int r = -kRadius; r <= kRadius; ++r) {
    sum += tile[kRadius + threadIdx.x + r];
  }
  out[i] = sum / (2 * kRadius + 1);
}

}  // namespace

int main() {
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);

  std::vector<float> host(kN), expected(kN), result(kN);
  for (int i = 0; i < kN; ++i) host[i] = static_cast<float>((i * 37) % 1009);
  for (int i = 0; i < kN; ++i) {
    float sum = 0.0f;
    for (int r = -kRadius; r <= kRadius; ++r) {
      int j = i + r;
      if (j >= 0 && j < kN) sum += host[j];
    }
    expected[i] = sum / (2 * kRadius + 1);
  }

  float* dev_in = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const int grid = (kN + kBlockSize - 1) / kBlockSize;
  // Useful traffic: read each input once + write each output once.
  const size_t moved = 2 * bytes;

  auto launch_naive = [&]() { GPU_LAUNCH(boxFilterNaive, grid, kBlockSize, 0, dev_in, dev_out, kN); };
  launch_naive();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, bytes));
  if (!gklab::verifyClose(result, expected)) return EXIT_FAILURE;
  gklab::report("box_naive_17loads", gklab::benchmarkKernel(launch_naive, moved, 0.0), kPeakGbPerSec, 0.0);

  auto launch_shared = [&]() { GPU_LAUNCH(boxFilterShared, grid, kBlockSize, 0, dev_in, dev_out, kN); };
  launch_shared();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, bytes));
  if (!gklab::verifyClose(result, expected)) return EXIT_FAILURE;
  gklab::report("box_shared_tile", gklab::benchmarkKernel(launch_shared, moved, 0.0), kPeakGbPerSec, 0.0);

  return EXIT_SUCCESS;
}
