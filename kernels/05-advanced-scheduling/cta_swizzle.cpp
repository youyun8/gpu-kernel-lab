// cta_swizzle.cpp
//
// Compares default row-major CTA-to-tile mapping against a grouped swizzle that
// improves L2 locality between concurrently executing CTAs (chapter 13). Both
// compute the same tiled GEMM and are validated against a CPU reference.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kTile = 16;
constexpr int kSwizzleGroup = 8;  // tiles per swizzle group along N

__device__ void gemmTileAt(const float* a, const float* b, float* c, int m, int n, int k,
                           int tileRow, int tileCol) {
  __shared__ float sA[kTile][kTile];
  __shared__ float sB[kTile][kTile];
  int row = tileRow * kTile + threadIdx.y;
  int col = tileCol * kTile + threadIdx.x;
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

__global__ void gemmRowMajor(const float* a, const float* b, float* c, int m, int n, int k) {
  gemmTileAt(a, b, c, m, n, k, blockIdx.y, blockIdx.x);
}

// Swizzled: remap the linear block id so nearby CTAs share more A/B rows/cols.
__global__ void gemmSwizzled(const float* a, const float* b, float* c, int m, int n, int k,
                             int tilesN) {
  int linear = blockIdx.y * gridDim.x + blockIdx.x;
  int tilesInGroup = kSwizzleGroup;
  int groupSizeTiles = tilesInGroup * (m / kTile > 0 ? (m / kTile) : 1);
  int group = linear / groupSizeTiles;
  int inGroup = linear % groupSizeTiles;
  int tileRow = inGroup % (m / kTile);
  int tileCol = group * tilesInGroup + inGroup / (m / kTile);
  if (tileCol >= tilesN) return;
  gemmTileAt(a, b, c, m, n, k, tileRow, tileCol);
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
  constexpr int kM = 512;
  constexpr int kN = 512;
  constexpr int kK = 512;
  const int tilesM = kM / kTile;
  const int tilesN = kN / kTile;

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
  dim3 grid(tilesN, tilesM);
  const double flops = 2.0 * kM * kN * kK;
  const size_t bytesMoved = bytesA + bytesB + bytesC;
  constexpr double kPeakGflopPerSec = 19500.0;

  auto launchRow = [&]() { GPU_LAUNCH(gemmRowMajor, grid, block, 0, devA, devB, devC, kM, kN, kK); };
  auto launchSwz = [&]() { GPU_LAUNCH(gemmSwizzled, grid, block, 0, devA, devB, devC, kM, kN, kK, tilesN); };

  launchRow();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devC, bytesC));
  if (!gklab::verifyClose(result, reference, 1.0e-2f)) return EXIT_FAILURE;
  gklab::report("gemm_row_major", gklab::benchmarkKernel(launchRow, bytesMoved, flops), 0.0, kPeakGflopPerSec);

  GPU_CHECK(gpuMemset(devC, 0, bytesC));
  launchSwz();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devC, bytesC));
  if (!gklab::verifyClose(result, reference, 1.0e-2f)) return EXIT_FAILURE;
  gklab::report("gemm_swizzled", gklab::benchmarkKernel(launchSwz, bytesMoved, flops), 0.0, kPeakGflopPerSec);

  GPU_CHECK(gpuFree(devA));
  GPU_CHECK(gpuFree(devB));
  GPU_CHECK(gpuFree(devC));
  return EXIT_SUCCESS;
}
