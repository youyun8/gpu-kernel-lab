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
                           int tile_row, int tile_col) {
  __shared__ float s_a[kTile][kTile];
  __shared__ float s_b[kTile][kTile];
  int row = tile_row * kTile + threadIdx.y;
  int col = tile_col * kTile + threadIdx.x;
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

__global__ void gemmRowMajor(const float* a, const float* b, float* c, int m, int n, int k) {
  gemmTileAt(a, b, c, m, n, k, blockIdx.y, blockIdx.x);
}

// Swizzled: remap the linear block id so nearby CTAs share more A/B rows/cols.
__global__ void gemmSwizzled(const float* a, const float* b, float* c, int m, int n, int k,
                             int tiles_n) {
  int linear = blockIdx.y * gridDim.x + blockIdx.x;
  int tiles_in_group = kSwizzleGroup;
  int group_size_tiles = tiles_in_group * (m / kTile > 0 ? (m / kTile) : 1);
  int group = linear / group_size_tiles;
  int in_group = linear % group_size_tiles;
  int tile_row = in_group % (m / kTile);
  int tile_col = group * tiles_in_group + in_group / (m / kTile);
  if (tile_col >= tiles_n) return;
  gemmTileAt(a, b, c, m, n, k, tile_row, tile_col);
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
  constexpr int kSizeM = 512;
  constexpr int kSizeN = 512;
  constexpr int kSizeK = 512;
  const int tiles_m = kSizeM / kTile;
  const int tiles_n = kSizeN / kTile;

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
  dim3 grid(tiles_n, tiles_m);
  const double flops = 2.0 * kSizeM * kSizeN * kSizeK;
  const size_t bytes_moved = bytes_a + bytes_b + bytes_c;
  constexpr double kPeakGflopPerSec = 19500.0;

  auto launch_row = [&]() { GPU_LAUNCH(gemmRowMajor, grid, block, 0, dev_a, dev_b, dev_c, kSizeM, kSizeN, kSizeK); };
  auto launch_swz = [&]() { GPU_LAUNCH(gemmSwizzled, grid, block, 0, dev_a, dev_b, dev_c, kSizeM, kSizeN, kSizeK, tiles_n); };

  launch_row();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_c, bytes_c));
  if (!gklab::verifyClose(result, reference, 1.0e-2f)) return EXIT_FAILURE;
  gklab::report("gemm_row_major", gklab::benchmarkKernel(launch_row, bytes_moved, flops), 0.0, kPeakGflopPerSec);

  GPU_CHECK(gpuMemset(dev_c, 0, bytes_c));
  launch_swz();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_c, bytes_c));
  if (!gklab::verifyClose(result, reference, 1.0e-2f)) return EXIT_FAILURE;
  gklab::report("gemm_swizzled", gklab::benchmarkKernel(launch_swz, bytes_moved, flops), 0.0, kPeakGflopPerSec);

  GPU_CHECK(gpuFree(dev_a));
  GPU_CHECK(gpuFree(dev_b));
  GPU_CHECK(gpuFree(dev_c));
  return EXIT_SUCCESS;
}
