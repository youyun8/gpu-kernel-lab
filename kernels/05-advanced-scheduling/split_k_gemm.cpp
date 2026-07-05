// split_k_gemm.cpp
//
// Compares a plain tiled GEMM against a Split-K GEMM on a skinny problem (small
// M, N; large K) where too few output tiles under-fill the GPU (chapter 13).
// Split-K partitions K across multiple CTAs that produce partial sums combined
// with atomics, trading an extra reduction for higher occupancy.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kTile = 16;

// Plain tiled GEMM: one output tile per block, loops over all of K.
__global__ void gemmPlain(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float s_a[kTile][kTile];
  __shared__ float s_b[kTile][kTile];
  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < k; t += kTile) {
    s_a[threadIdx.y][threadIdx.x] = (row < m) ? a[row * k + (t + threadIdx.x)] : 0.0f;
    s_b[threadIdx.y][threadIdx.x] = (col < n) ? b[(t + threadIdx.y) * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int p = 0; p < kTile; ++p) acc += s_a[threadIdx.y][p] * s_b[p][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = acc;
}

// Split-K: gridDim.z CTAs each handle a K-slice and atomically add partials.
__global__ void gemmSplitK(const float* a, const float* b, float* c, int m, int n, int k, int splits) {
  __shared__ float s_a[kTile][kTile];
  __shared__ float s_b[kTile][kTile];
  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  int kPerSplit = (k + splits - 1) / splits;
  int kStart = blockIdx.z * kPerSplit;
  int kEnd = min(kStart + kPerSplit, k);
  float acc = 0.0f;
  for (int t = kStart; t < kEnd; t += kTile) {
    int a_col = t + threadIdx.x;
    int b_row = t + threadIdx.y;
    s_a[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    s_b[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int p = 0; p < kTile; ++p) acc += s_a[threadIdx.y][p] * s_b[p][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) atomicAdd(&c[row * n + col], acc);
}

void cpuGemm(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c,
             int m, int n, int k) {
  for (int r = 0; r < m; ++r)
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) acc += a[r * k + p] * b[p * n + col];
      c[r * n + col] = acc;
    }
}

}  // namespace

int main() {
  constexpr int kSizeM = 128;
  constexpr int kSizeN = 128;
  constexpr int kSizeK = 4096;  // large K, few output tiles -> under-filled GPU
  constexpr int kSplits = 8;

  std::vector<float> host_a(static_cast<size_t>(kSizeM) * kSizeK);
  std::vector<float> host_b(static_cast<size_t>(kSizeK) * kSizeN);
  std::vector<float> reference(static_cast<size_t>(kSizeM) * kSizeN);
  std::vector<float> result(static_cast<size_t>(kSizeM) * kSizeN);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>((i % 9) - 4) * 0.05f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>((i % 5) - 2) * 0.1f;
  cpuGemm(host_a, host_b, reference, kSizeM, kSizeN, kSizeK);

  float* dev_a = nullptr;
  float* dev_b = nullptr;
  float* dev_c = nullptr;
  const size_t bytes_a = host_a.size() * sizeof(float);
  const size_t bytes_b = host_b.size() * sizeof(float);
  const size_t bytes_c = reference.size() * sizeof(float);
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_a), bytes_a));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_b), bytes_b));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_c), bytes_c));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_a, host_a.data(), bytes_a));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_b, host_b.data(), bytes_b));

  dim3 block(kTile, kTile);
  dim3 grid_plain(kSizeN / kTile, kSizeM / kTile);
  dim3 grid_split(kSizeN / kTile, kSizeM / kTile, kSplits);
  const double flops = 2.0 * kSizeM * kSizeN * kSizeK;
  const size_t bytes_moved = bytes_a + bytes_b + bytes_c;
  constexpr double kPeakGflopPerSec = 19500.0;

  auto launch_plain = [&]() {
    GPU_CHECK(gpuMemset(dev_c, 0, bytes_c));
    GPU_LAUNCH(gemmPlain, grid_plain, block, 0, dev_a, dev_b, dev_c, kSizeM, kSizeN, kSizeK);
  };
  auto launch_split = [&]() {
    GPU_CHECK(gpuMemset(dev_c, 0, bytes_c));
    GPU_LAUNCH(gemmSplitK, grid_split, block, 0, dev_a, dev_b, dev_c, kSizeM, kSizeN, kSizeK, kSplits);
  };

  launch_plain();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_c, bytes_c));
  if (!gklab::verifyClose(result, reference, 1.0e-2f)) return EXIT_FAILURE;
  gklab::report("gemm_plain", gklab::benchmarkKernel(launch_plain, bytes_moved, flops), 0.0, kPeakGflopPerSec);

  launch_split();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_c, bytes_c));
  if (!gklab::verifyClose(result, reference, 1.0e-2f)) return EXIT_FAILURE;
  gklab::report("gemm_split_k", gklab::benchmarkKernel(launch_split, bytes_moved, flops), 0.0, kPeakGflopPerSec);

  GPU_CHECK(gpuFree(dev_a));
  GPU_CHECK(gpuFree(dev_b));
  GPU_CHECK(gpuFree(dev_c));
  return EXIT_SUCCESS;
}
