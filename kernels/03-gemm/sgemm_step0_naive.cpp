// sgemm_step0_naive.cpp
//
// Step 0 of the GEMM series: one thread per output element. Zero data reuse,
// arithmetic intensity ~ 0.25 FLOP/byte, so it is firmly bandwidth-bound.
#include "gemm_common.h"

namespace {

constexpr int kBlockDim = 16;

__global__ void sgemmNaive(const float* a, const float* b, float* c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += a[row * k + p] * b[p * n + col];
    c[row * n + col] = acc;
  }
}

}  // namespace

int main() {
  constexpr int kM = 1024;
  constexpr int kN = 1024;
  constexpr int kK = 1024;
  dim3 block(kBlockDim, kBlockDim);
  dim3 grid((kN + kBlockDim - 1) / kBlockDim, (kM + kBlockDim - 1) / kBlockDim);
  return gklab::runGemm("sgemm_step0_naive", kM, kN, kK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmNaive, grid, block, 0, buf.a, buf.b, buf.c, kM, kN, kK);
  });
}
