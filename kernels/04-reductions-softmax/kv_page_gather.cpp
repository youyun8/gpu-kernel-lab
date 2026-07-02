// kv_page_gather.cpp
//
// The address indirection at the heart of PagedAttention (chapter 25): the KV
// cache is stored in fixed-size pages scattered across a pool, and a per-
// sequence page table maps logical token positions to physical pages. The
// kernel gathers one sequence's K vectors into contiguous output — exactly
// what an attention kernel does on the fly — and is verified against a host
// gather. Compare with the contiguous baseline to see the indirection cost.
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kPageSize = 16;        // tokens per page (vLLM-style block)
constexpr int kHeadDim = 64;
constexpr int kSeqLen = 4096;        // tokens in the sequence
constexpr int kPoolPages = 8192;     // physical pages in the KV pool
constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;

// One warp per token. Logical token t lives at physical page
// page_table[t / kPageSize], slot t % kPageSize inside that page.
__global__ void gatherPagedK(const float* pool, const int* page_table, float* out,
                             int seq_len) {
  int token = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  if (token >= seq_len) return;

  int page = page_table[token / kPageSize];
  int slot = token % kPageSize;
  const float* src = pool + (static_cast<size_t>(page) * kPageSize + slot) * kHeadDim;
  float* dst = out + static_cast<size_t>(token) * kHeadDim;
  for (int e = lane; e < kHeadDim; e += kWarpSize) dst[e] = src[e];
}

// Baseline: the same copy from a contiguous (non-paged) KV layout.
__global__ void gatherContiguousK(const float* kv, float* out, int seq_len) {
  int token = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  if (token >= seq_len) return;
  const float* src = kv + static_cast<size_t>(token) * kHeadDim;
  float* dst = out + static_cast<size_t>(token) * kHeadDim;
  for (int e = lane; e < kHeadDim; e += kWarpSize) dst[e] = src[e];
}

}  // namespace

int main() {
  const int num_pages = kSeqLen / kPageSize;
  const size_t pool_elems = static_cast<size_t>(kPoolPages) * kPageSize * kHeadDim;
  const size_t seq_elems = static_cast<size_t>(kSeqLen) * kHeadDim;

  // Scatter the sequence's pages randomly through the pool, like an allocator
  // serving many concurrent sequences would.
  std::vector<int> page_table(num_pages);
  {
    std::vector<int> pages(kPoolPages);
    std::iota(pages.begin(), pages.end(), 0);
    std::mt19937 rng(7);
    std::shuffle(pages.begin(), pages.end(), rng);
    std::copy(pages.begin(), pages.begin() + num_pages, page_table.begin());
  }

  std::vector<float> pool(pool_elems);
  for (size_t i = 0; i < pool_elems; ++i) pool[i] = static_cast<float>(i % 911) * 0.01f;

  // Host reference gather + the equivalent contiguous layout.
  std::vector<float> expected(seq_elems), contiguous(seq_elems);
  for (int t = 0; t < kSeqLen; ++t) {
    int page = page_table[t / kPageSize];
    int slot = t % kPageSize;
    for (int e = 0; e < kHeadDim; ++e) {
      expected[static_cast<size_t>(t) * kHeadDim + e] =
          pool[(static_cast<size_t>(page) * kPageSize + slot) * kHeadDim + e];
    }
  }
  contiguous = expected;

  float *dev_pool = nullptr, *dev_contig = nullptr, *dev_out = nullptr;
  int* dev_table = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_pool), pool_elems * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_contig), seq_elems * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), seq_elems * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_table), num_pages * sizeof(int)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_pool, pool.data(), pool_elems * sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_contig, contiguous.data(), seq_elems * sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_table, page_table.data(), num_pages * sizeof(int)));

  constexpr double kPeakGbPerSec = 1555.0;
  const int grid = (kSeqLen * kWarpSize + kBlockSize - 1) / kBlockSize;
  const size_t moved = 2 * seq_elems * sizeof(float);

  auto launch_paged = [&]() { GPU_LAUNCH(gatherPagedK, grid, kBlockSize, 0, dev_pool, dev_table, dev_out, kSeqLen); };
  launch_paged();
  GPU_CHECK(gpuDeviceSynchronize());
  std::vector<float> result(seq_elems);
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, seq_elems * sizeof(float)));
  if (!gklab::verifyClose(result, expected)) return EXIT_FAILURE;
  gklab::report("gather_paged_kv", gklab::benchmarkKernel(launch_paged, moved, 0.0), kPeakGbPerSec, 0.0);

  auto launch_contig = [&]() { GPU_LAUNCH(gatherContiguousK, grid, kBlockSize, 0, dev_contig, dev_out, kSeqLen); };
  gklab::report("gather_contiguous", gklab::benchmarkKernel(launch_contig, moved, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(dev_pool));
  GPU_CHECK(gpuFree(dev_contig));
  GPU_CHECK(gpuFree(dev_out));
  GPU_CHECK(gpuFree(dev_table));
  return EXIT_SUCCESS;
}
