// gather_scatter.cpp
//
// Contrasts the two directions of indexed access on a histogram-style
// workload (4M items into 256 bins):
//   gather      - out[i] = table[idx[i]]: random reads, coalesced writes.
//   scatter     - atomicAdd(&bins[idx[i]], 1): random atomic writes; global
//                 contention serializes colliding lanes.
//   scatter_agg - block aggregation: accumulate into a shared-memory histogram
//                 first, then flush 256 atomics per block to global memory.
// The aggregated version issues orders of magnitude fewer global atomics.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kNumBins = 256;
constexpr int kNumItems = 1 << 22;

// GATHER: reads are data-dependent and scattered, but every lane's write goes
// to out[i] -> stores are perfectly coalesced.
__global__ void gatherValues(const float* table, const int* idx, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = table[idx[i]];
}

// SCATTER, naive: every item is one global atomicAdd. With only 256 bins the
// collision rate is huge and the atomics serialize.
__global__ void scatterAtomicNaive(const int* idx, unsigned int* bins, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bins[idx[i]], 1u);
}

// SCATTER, block-aggregated: shared-memory histogram absorbs the collisions
// inside the block (shared atomics are much cheaper), then one flush pass
// issues kNumBins global atomics per block instead of kBlockSize.
__global__ void scatterAtomicAggregated(const int* idx, unsigned int* bins, int n) {
  __shared__ unsigned int local_bins[kNumBins];
  for (int b = threadIdx.x; b < kNumBins; b += blockDim.x) local_bins[b] = 0u;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&local_bins[idx[i]], 1u);
  __syncthreads();

  for (int b = threadIdx.x; b < kNumBins; b += blockDim.x) {
    if (local_bins[b] != 0u) atomicAdd(&bins[b], local_bins[b]);
  }
}

}  // namespace

int main() {
  std::vector<int> idx(kNumItems);
  std::vector<float> table(kNumBins);
  std::vector<unsigned int> reference(kNumBins, 0u);
  for (int b = 0; b < kNumBins; ++b) table[b] = static_cast<float>(b) * 0.5f;
  for (int i = 0; i < kNumItems; ++i) {
    idx[i] = static_cast<int>((i * 2654435761u) % kNumBins);
    ++reference[idx[i]];
  }

  int* dev_idx = nullptr;
  float* dev_table = nullptr;
  float* dev_out = nullptr;
  unsigned int* dev_bins = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_idx), kNumItems * sizeof(int)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_table), kNumBins * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), kNumItems * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_bins), kNumBins * sizeof(unsigned int)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_idx, idx.data(), kNumItems * sizeof(int)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_table, table.data(), kNumBins * sizeof(float)));

  constexpr double kPeakGbPerSec = 1555.0;
  const int grid = (kNumItems + kBlockSize - 1) / kBlockSize;
  // Useful traffic: idx read + value written (gather) or bin updated (scatter).
  const size_t moved = static_cast<size_t>(kNumItems) * (sizeof(int) + sizeof(float));

  auto check_bins = [&](const char* name) {
    std::vector<unsigned int> bins(kNumBins);
    GPU_CHECK(gpuMemcpyDeviceToHost(bins.data(), dev_bins, kNumBins * sizeof(unsigned int)));
    for (int b = 0; b < kNumBins; ++b) {
      if (bins[b] != reference[b]) {
        std::fprintf(stderr, "%s: bin %d = %u, want %u\n", name, b, bins[b], reference[b]);
        std::exit(EXIT_FAILURE);
      }
    }
  };

  auto launch_gather = [&]() { GPU_LAUNCH(gatherValues, grid, kBlockSize, 0, dev_table, dev_idx, dev_out, kNumItems); };
  gklab::report("gather", gklab::benchmarkKernel(launch_gather, moved, 0.0), kPeakGbPerSec, 0.0);

  auto launch_naive = [&]() {
    GPU_CHECK(gpuMemset(dev_bins, 0, kNumBins * sizeof(unsigned int)));
    GPU_LAUNCH(scatterAtomicNaive, grid, kBlockSize, 0, dev_idx, dev_bins, kNumItems);
  };
  launch_naive();
  GPU_CHECK(gpuDeviceSynchronize());
  check_bins("scatter_naive");
  gklab::report("scatter_naive", gklab::benchmarkKernel(launch_naive, moved, 0.0), kPeakGbPerSec, 0.0);

  auto launch_agg = [&]() {
    GPU_CHECK(gpuMemset(dev_bins, 0, kNumBins * sizeof(unsigned int)));
    GPU_LAUNCH(scatterAtomicAggregated, grid, kBlockSize, 0, dev_idx, dev_bins, kNumItems);
  };
  launch_agg();
  GPU_CHECK(gpuDeviceSynchronize());
  check_bins("scatter_aggregated");
  gklab::report("scatter_block_agg", gklab::benchmarkKernel(launch_agg, moved, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(dev_idx));
  GPU_CHECK(gpuFree(dev_table));
  GPU_CHECK(gpuFree(dev_out));
  GPU_CHECK(gpuFree(dev_bins));
  return EXIT_SUCCESS;
}
