// embedding_lookup.cpp
//
// Embedding gather: out[i][:] = table[idx[i]][:], one warp per lookup row.
// Real batches have skewed, repeated indices. Sorting the batch groups
// duplicate/nearby rows so consecutive warps hit the same table rows while
// they are still resident in L2; an inverse permutation then writes results
// back in the original batch order. Compares random-order vs sorted+permuted
// lookups and verifies both outputs match exactly.
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kWarpSize = 32;
constexpr int kDim = 64;             // embedding width: 64 floats = 256 bytes
constexpr int kVocab = 1 << 18;      // 262,144 rows x 256 B = 64 MiB table
constexpr int kBatch = 1 << 20;      // 1M lookups
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockSize = kWarpsPerBlock * kWarpSize;

// One warp per lookup: lane L copies elements L, L+32 of the row. Both the
// table read and the out write are coalesced within the warp; locality across
// warps depends purely on the order of idx[].
__global__ void embeddingGather(const float* table, const int* idx, const int* out_pos,
                                float* out, int batch) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  if (warp >= batch) return;
  const float* src = table + static_cast<size_t>(idx[warp]) * kDim;
  // out_pos maps the (possibly sorted) processing order back to batch order.
  float* dst = out + static_cast<size_t>(out_pos[warp]) * kDim;
  for (int d = lane; d < kDim; d += kWarpSize) dst[d] = src[d];
}

}  // namespace

int main() {
  // Skewed distribution: 90% of lookups land in a 4096-row "hot" set, like
  // frequent tokens in a language-model batch.
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> hot(0, 4095);
  std::uniform_int_distribution<int> cold(0, kVocab - 1);
  std::uniform_real_distribution<float> coin(0.0f, 1.0f);

  std::vector<int> idx(kBatch);
  for (int i = 0; i < kBatch; ++i) idx[i] = (coin(rng) < 0.9f) ? hot(rng) : cold(rng);

  // Sorted order: process lookups grouped by row id; out_pos restores the
  // original position of each lookup (the inverse permutation).
  std::vector<int> order(kBatch);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return idx[a] < idx[b]; });
  std::vector<int> sorted_idx(kBatch), sorted_out_pos(kBatch);
  for (int i = 0; i < kBatch; ++i) {
    sorted_idx[i] = idx[order[i]];
    sorted_out_pos[i] = order[i];
  }
  std::vector<int> identity(kBatch);
  std::iota(identity.begin(), identity.end(), 0);

  std::vector<float> table(static_cast<size_t>(kVocab) * kDim);
  for (size_t i = 0; i < table.size(); ++i) table[i] = static_cast<float>(i % 997);

  float* dev_table = nullptr;
  float* dev_out = nullptr;
  int* dev_idx = nullptr;
  int* dev_pos = nullptr;
  const size_t out_bytes = static_cast<size_t>(kBatch) * kDim * sizeof(float);
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_table), table.size() * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), out_bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_idx), kBatch * sizeof(int)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_pos), kBatch * sizeof(int)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_table, table.data(), table.size() * sizeof(float)));

  constexpr double kPeakGbPerSec = 1555.0;
  const int grid = (kBatch * kWarpSize + kBlockSize - 1) / kBlockSize;
  // Useful traffic: one row read + one row written per lookup.
  const size_t moved = 2 * out_bytes;

  std::vector<float> out_random(static_cast<size_t>(kBatch) * kDim);
  std::vector<float> out_sorted(static_cast<size_t>(kBatch) * kDim);

  // Random (original batch) order.
  GPU_CHECK(gpuMemcpyHostToDevice(dev_idx, idx.data(), kBatch * sizeof(int)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_pos, identity.data(), kBatch * sizeof(int)));
  auto launch_random = [&]() { GPU_LAUNCH(embeddingGather, grid, kBlockSize, 0, dev_table, dev_idx, dev_pos, dev_out, kBatch); };
  launch_random();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(out_random.data(), dev_out, out_bytes));
  gklab::report("lookup_random", gklab::benchmarkKernel(launch_random, moved, 0.0), kPeakGbPerSec, 0.0);

  // Sorted order + inverse permutation on the write side.
  GPU_CHECK(gpuMemcpyHostToDevice(dev_idx, sorted_idx.data(), kBatch * sizeof(int)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_pos, sorted_out_pos.data(), kBatch * sizeof(int)));
  auto launch_sorted = [&]() { GPU_LAUNCH(embeddingGather, grid, kBlockSize, 0, dev_table, dev_idx, dev_pos, dev_out, kBatch); };
  launch_sorted();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(out_sorted.data(), dev_out, out_bytes));
  gklab::report("lookup_sorted", gklab::benchmarkKernel(launch_sorted, moved, 0.0), kPeakGbPerSec, 0.0);

  if (!gklab::verifyClose(out_sorted, out_random)) {
    std::fprintf(stderr, "sorted+inverse-permutation output differs from random-order output\n");
    return EXIT_FAILURE;
  }
  std::printf("sorted + inverse permutation matches original batch order output\n");

  GPU_CHECK(gpuFree(dev_table));
  GPU_CHECK(gpuFree(dev_out));
  GPU_CHECK(gpuFree(dev_idx));
  GPU_CHECK(gpuFree(dev_pos));
  return EXIT_SUCCESS;
}
