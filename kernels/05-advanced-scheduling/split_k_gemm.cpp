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
  __shared__ float sA[kTile][kTile];
  __shared__ float sB[kTile][kTile];
  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < k; t += kTile) {
    sA[threadIdx.y][threadIdx.x] = (row < m) ? a[row * k + (t + threadIdx.x)] : 0.0f;
    sB[threadIdx.y][threadIdx.x] = (col < n) ? b[(t + threadIdx.y) * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int p = 0; p < kTile; ++p) acc += sA[threadIdx.y][p] * sB[p][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = acc;
}

// Split-K: gridDim.z CTAs each handle a K-slice and atomically add partials.
__global__ void gemmSplitK(const float* a, const float* b, float* c, int m, int n, int k, int splits) {
  __shared__ float sA[kTile][kTile];
  __shared__ float sB[kTile][kTile];
  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  int kPerSplit = (k + splits - 1) / splits;
  int kStart = blockIdx.z * kPerSplit;
  int kEnd = min(kStart + kPerSplit, k);
  float acc = 0.0f;
  for (int t = kStart; t < kEnd; t += kTile) {
    int aCol = t + threadIdx.x;
    int bRow = t + threadIdx.y;
    sA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
    sB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int p = 0; p < kTile; ++p) acc += sA[threadIdx.y][p] * sB[p][threadIdx.x];
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
  constexpr int kM = 128;
  constexpr int kN = 128;
  constexpr int kK = 4096;  // large K, few output tiles -> under-filled GPU
  constexpr int kSplits = 8;

  std::vector<float> hostA(static_cast<size_t>(kM) * kK);
  std::vector<float> hostB(static_cast<size_t>(kK) * kN);
  std::vector<float> reference(static_cast<size_t>(kM) * kN);
  std::vector<float> result(static_cast<size_t>(kM) * kN);
  for (size_t i = 0; i < hostA.size(); ++i) hostA[i] = static_cast<float>((i % 9) - 4) * 0.05f;
  for (size_t i = 0; i < hostB.size(); ++i) hostB[i] = static_cast<float>((i % 5) - 2) * 0.1f;
  cpuGemm(hostA, hostB, reference, kM, kN, kK);

  float* devA = nullptr;
  float* devB = nullptr;
  float* devC = nullptr;
  const size_t bytesA = hostA.size() * sizeof(float);
  const size_t bytesB = hostB.size() * sizeof(float);
  const size_t bytesC = reference.size() * sizeof(float);
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devA), bytesA));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devB), bytesB));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devC), bytesC));
  GPU_CHECK(gpuMemcpyHostToDevice(devA, hostA.data(), bytesA));
  GPU_CHECK(gpuMemcpyHostToDevice(devB, hostB.data(), bytesB));

  dim3 block(kTile, kTile);
  dim3 gridPlain(kN / kTile, kM / kTile);
  dim3 gridSplit(kN / kTile, kM / kTile, kSplits);
  const double flops = 2.0 * kM * kN * kK;
  const size_t bytesMoved = bytesA + bytesB + bytesC;
  constexpr double kPeakGflopPerSec = 19500.0;

  auto launchPlain = [&]() {
    GPU_CHECK(gpuMemset(devC, 0, bytesC));
    GPU_LAUNCH(gemmPlain, gridPlain, block, 0, devA, devB, devC, kM, kN, kK);
  };
  auto launchSplit = [&]() {
    GPU_CHECK(gpuMemset(devC, 0, bytesC));
    GPU_LAUNCH(gemmSplitK, gridSplit, block, 0, devA, devB, devC, kM, kN, kK, kSplits);
  };

  launchPlain();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devC, bytesC));
  if (!gklab::verifyClose(result, reference, 1.0e-2f)) return EXIT_FAILURE;
  gklab::report("gemm_plain", gklab::benchmarkKernel(launchPlain, bytesMoved, flops), 0.0, kPeakGflopPerSec);

  launchSplit();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devC, bytesC));
  if (!gklab::verifyClose(result, reference, 1.0e-2f)) return EXIT_FAILURE;
  gklab::report("gemm_split_k", gklab::benchmarkKernel(launchSplit, bytesMoved, flops), 0.0, kPeakGflopPerSec);

  GPU_CHECK(gpuFree(devA));
  GPU_CHECK(gpuFree(devB));
  GPU_CHECK(gpuFree(devC));
  return EXIT_SUCCESS;
}
