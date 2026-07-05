// gpu_histogram_race.cpp
//
// GPU twin of race_demo / race_fixed_atomic: thousands of threads add into a
// small histogram. The racy kernel uses a plain read-modify-write on global
// memory and silently loses updates; the fixed kernel uses atomicAdd. Run it
// and compare the bin totals - the racy version is wrong and varies between
// runs, the atomic version always matches the host reference.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kNumBins = 16;
constexpr int kNumItems = 1 << 22;
constexpr int kTrials = 3;

// BROKEN on purpose: bins[b] += 1 is a separate load, add, store. Any two
// threads hitting the same bin at once can lose one of the increments.
__global__ void histogramRacy(const int* items, int* bins, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int b = items[i];
    bins[b] = bins[b] + 1;  // RACE: unsynchronized read-modify-write
  }
}

// Correct: atomicAdd makes the read-modify-write indivisible. Contention on
// 16 bins is high; chapter 26 shows warp/block aggregation to cut it down.
__global__ void histogramAtomic(const int* items, int* bins, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&bins[items[i]], 1);
  }
}

}  // namespace

int main() {
  std::vector<int> items(kNumItems);
  std::vector<long long> reference(kNumBins, 0);
  for (int i = 0; i < kNumItems; ++i) {
    items[i] = (i * 2654435761u) % kNumBins;  // scrambled but deterministic
    ++reference[items[i]];
  }

  int* dev_items = nullptr;
  int* dev_bins = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_items), kNumItems * sizeof(int)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_bins), kNumBins * sizeof(int)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_items, items.data(), kNumItems * sizeof(int)));

  const int grid = (kNumItems + kBlockSize - 1) / kBlockSize;
  std::vector<int> bins(kNumBins);

  auto run_and_check = [&](bool racy, int trial) {
    GPU_CHECK(gpuMemset(dev_bins, 0, kNumBins * sizeof(int)));
    if (racy) {
      GPU_LAUNCH(histogramRacy, grid, kBlockSize, 0, dev_items, dev_bins, kNumItems);
    } else {
      GPU_LAUNCH(histogramAtomic, grid, kBlockSize, 0, dev_items, dev_bins, kNumItems);
    }
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(bins.data(), dev_bins, kNumBins * sizeof(int)));

    long long total = 0;
    long long max_missing = 0;
    for (int b = 0; b < kNumBins; ++b) {
      total += bins[b];
      max_missing = std::max<long long>(max_missing, reference[b] - bins[b]);
    }
    std::printf("  %-6s trial %d: counted %10lld / %d  (max missing per bin %lld) %s\n",
                racy ? "racy" : "atomic", trial, total, kNumItems, max_missing,
                total == kNumItems ? "OK" : "WRONG");
  };

  std::printf("gpu_histogram_race: %d items into %d bins, expected total %d\n",
              kNumItems, kNumBins, kNumItems);
  for (int trial = 0; trial < kTrials; ++trial) run_and_check(/*racy=*/true, trial);
  for (int trial = 0; trial < kTrials; ++trial) run_and_check(/*racy=*/false, trial);

  GPU_CHECK(gpuFree(dev_items));
  GPU_CHECK(gpuFree(dev_bins));
  return 0;
}
