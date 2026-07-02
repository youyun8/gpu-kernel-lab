// sgemm_step2_register_tiling.cpp
//
// Step 2: block tiling in shared memory plus register (thread-level) tiling.
// Each thread computes a kTM x kTN micro-tile of C, holding operands in
// registers to amortize shared-memory bandwidth. Assumes dimensions divisible
// by the block tile sizes (true for the 1024 case used here).
#include "gemm_common.h"

namespace {

constexpr int kBM = 64;  // block tile rows of C
constexpr int kBN = 64;  // block tile cols of C
constexpr int kBK = 16;  // block tile depth
constexpr int kTM = 4;   // per-thread rows
constexpr int kTN = 4;   // per-thread cols
// Threads per block: (kBM/kTM) x (kBN/kTN) = 16 x 16 = 256.

__global__ void sgemmRegisterTiling(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float sA[kBK][kBM];  // transposed A tile for coalesced register reads
  __shared__ float sB[kBK][kBN];

  const int threadCol = threadIdx.x % (kBN / kTN);
  const int threadRow = threadIdx.x / (kBN / kTN);
  const int blockRow = blockIdx.y * kBM;
  const int blockCol = blockIdx.x * kBN;
  const int threadsPerBlock = (kBM / kTM) * (kBN / kTN);

  float acc[kTM][kTN] = {};

  for (int t = 0; t < k; t += kBK) {
    // Cooperatively load A (kBM x kBK) into sA transposed, and B (kBK x kBN).
    for (int idx = threadIdx.x; idx < kBM * kBK; idx += threadsPerBlock) {
      int r = idx / kBK;
      int col = idx % kBK;
      sA[col][r] = a[(blockRow + r) * k + (t + col)];
    }
    for (int idx = threadIdx.x; idx < kBK * kBN; idx += threadsPerBlock) {
      int r = idx / kBN;
      int col = idx % kBN;
      sB[r][col] = b[(t + r) * n + (blockCol + col)];
    }
    __syncthreads();

#pragma unroll
    for (int p = 0; p < kBK; ++p) {
      float regA[kTM];
      float regB[kTN];
#pragma unroll
      for (int i = 0; i < kTM; ++i) regA[i] = sA[p][threadRow * kTM + i];
#pragma unroll
      for (int j = 0; j < kTN; ++j) regB[j] = sB[p][threadCol * kTN + j];
#pragma unroll
      for (int i = 0; i < kTM; ++i)
#pragma unroll
        for (int j = 0; j < kTN; ++j) acc[i][j] += regA[i] * regB[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kTM; ++i)
#pragma unroll
    for (int j = 0; j < kTN; ++j) {
      int row = blockRow + threadRow * kTM + i;
      int col = blockCol + threadCol * kTN + j;
      if (row < m && col < n) c[row * n + col] = acc[i][j];
    }
}

}  // namespace

int main() {
  constexpr int kM = 1024;
  constexpr int kN = 1024;
  constexpr int kK = 1024;
  const int threadsPerBlock = (kBM / kTM) * (kBN / kTN);
  dim3 block(threadsPerBlock);
  dim3 grid(kN / kBN, kM / kBM);
  return gklab::runGemm("sgemm_step2_register_tiling", kM, kN, kK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmRegisterTiling, grid, block, 0, buf.a, buf.b, buf.c, kM, kN, kK);
  });
}
