// histogram_atomics.cpp
//
// Chapter 26 reference solution: compare global atomics, warp-aggregated
// atomics, and block-private shared-memory histograms for a 256-bin byte
// histogram. The input distribution is intentionally skewed to expose
// contention.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBins = 256;
constexpr int kBlockSize = 256;

__global__ void histogramGlobalAtomic(const unsigned char* input, unsigned int* bins, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    atomicAdd(&bins[input[i]], 1u);
  }
}

__global__ void histogramWarpAggregated(const unsigned char* input, unsigned int* bins, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    unsigned int key = input[i];
#if defined(USE_CUDA)
    unsigned int mask = __match_any_sync(0xffffffffu, key);
    int lane = threadIdx.x & 31;
    int leader = __ffs(mask) - 1;
    unsigned int count = __popc(mask);
    if (lane == leader) atomicAdd(&bins[key], count);
#else
    // HIP portability fallback. AMD wave-level grouped-match support varies by
    // target, so keep the reference target buildable and compare against the
    // shared-memory privatized version below.
    atomicAdd(&bins[key], 1u);
#endif
  }
}

__global__ void histogramBlockPrivate(const unsigned char* input, unsigned int* bins, int n) {
  __shared__ unsigned int local[kBins];

  for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) local[bin] = 0;
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    atomicAdd(&local[input[i]], 1u);
  }
  __syncthreads();

  for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
    unsigned int count = local[bin];
    if (count != 0) atomicAdd(&bins[bin], count);
  }
}

void cpuHistogram(const std::vector<unsigned char>& input, std::vector<unsigned int>& bins) {
  std::fill(bins.begin(), bins.end(), 0u);
  for (unsigned char value : input) ++bins[value];
}

bool verifyHistogram(const char* name, const std::vector<unsigned int>& got,
                     const std::vector<unsigned int>& want) {
  for (int i = 0; i < kBins; ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "%s FAILED: bin %d got %u want %u\n", name, i, got[i], want[i]);
      return false;
    }
  }
  std::printf("%s correctness OK\n", name);
  return true;
}

}  // namespace

int main() {
  constexpr int kSizeN = 1 << 24;
  constexpr int kGrid = 1024;
  const size_t input_bytes = static_cast<size_t>(kSizeN) * sizeof(unsigned char);
  const size_t bin_bytes = kBins * sizeof(unsigned int);

  std::vector<unsigned char> input(kSizeN);
  for (int i = 0; i < kSizeN; ++i) {
    // Skewed distribution: half the values hit bin 7, most of the rest hit a
    // small hot set, and a tail covers all bins.
    if ((i & 1) == 0) {
      input[i] = 7;
    } else if ((i % 8) != 0) {
      input[i] = static_cast<unsigned char>((i * 13) & 31);
    } else {
      input[i] = static_cast<unsigned char>((i * 37) & 255);
    }
  }

  std::vector<unsigned int> reference(kBins);
  std::vector<unsigned int> result(kBins);
  cpuHistogram(input, reference);

  unsigned char* dev_input = nullptr;
  unsigned int* dev_bins = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_input), input_bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_bins), bin_bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_input, input.data(), input_bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const size_t bytes_moved = input_bytes + bin_bytes;

  auto run = [&](const char* name, const std::function<void()>& kernel_launch) -> bool {
    auto launch = [&]() {
      GPU_CHECK(gpuMemset(dev_bins, 0, bin_bytes));
      kernel_launch();
    };
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_bins, bin_bytes));
    if (!verifyHistogram(name, result, reference)) return false;
    gklab::report(name, gklab::benchmarkKernel(launch, bytes_moved, 0.0), kPeakGbPerSec, 0.0);
    return true;
  };

  if (!run("hist_global_atomic", [&]() {
        GPU_LAUNCH(histogramGlobalAtomic, kGrid, kBlockSize, 0, dev_input, dev_bins, kSizeN);
      }))
    return EXIT_FAILURE;

  if (!run("hist_warp_aggregated", [&]() {
        GPU_LAUNCH(histogramWarpAggregated, kGrid, kBlockSize, 0, dev_input, dev_bins, kSizeN);
      }))
    return EXIT_FAILURE;

  if (!run("hist_block_private", [&]() {
        GPU_LAUNCH(histogramBlockPrivate, kGrid, kBlockSize, 0, dev_input, dev_bins, kSizeN);
      }))
    return EXIT_FAILURE;

  GPU_CHECK(gpuFree(dev_input));
  GPU_CHECK(gpuFree(dev_bins));
  return EXIT_SUCCESS;
}
