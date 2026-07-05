// sgemm_step1_shared.cpp
//
// Step 1: shared-memory tiling. Each block cooperatively loads kTile x kTile
// tiles of A and B into shared memory and reuses them, cutting global traffic.
#include "gemm_common.h"

namespace {

constexpr int kTile = 32;

__global__ void sgemmShared(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float s_a[kTile][kTile];
  __shared__ float s_b[kTile][kTile];
  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < k; t += kTile) {
    int a_col = t + threadIdx.x;
    int b_row = t + threadIdx.y;
    s_a[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    s_b[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int p = 0; p < kTile; ++p) acc += s_a[threadIdx.y][p] * s_b[p][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = acc;
}

}  // namespace

int main() {
  constexpr int kSizeM = 1024;
  constexpr int kSizeN = 1024;
  constexpr int kSizeK = 1024;
  dim3 block(kTile, kTile);
  dim3 grid((kSizeN + kTile - 1) / kTile, (kSizeM + kTile - 1) / kTile);
  return gklab::runGemm("sgemm_step1_shared", kSizeM, kSizeN, kSizeK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmShared, grid, block, 0, buf.a, buf.b, buf.c, kSizeM, kSizeN, kSizeK);
  });
}
