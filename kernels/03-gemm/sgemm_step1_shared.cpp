// sgemm_step1_shared.cpp
//
// Step 1: shared-memory tiling. Each block cooperatively loads kTile x kTile
// tiles of A and B into shared memory and reuses them, cutting global traffic.
#include "gemm_common.h"

namespace {

constexpr int kTile = 32;

__global__ void sgemmShared(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float sA[kTile][kTile];
  __shared__ float sB[kTile][kTile];
  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < k; t += kTile) {
    int aCol = t + threadIdx.x;
    int bRow = t + threadIdx.y;
    sA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
    sB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int p = 0; p < kTile; ++p) acc += sA[threadIdx.y][p] * sB[p][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = acc;
}

}  // namespace

int main() {
  constexpr int kM = 1024;
  constexpr int kN = 1024;
  constexpr int kK = 1024;
  dim3 block(kTile, kTile);
  dim3 grid((kN + kTile - 1) / kTile, (kM + kTile - 1) / kTile);
  return gklab::runGemm("sgemm_step1_shared", kM, kN, kK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmShared, grid, block, 0, buf.a, buf.b, buf.c, kM, kN, kK);
  });
}
