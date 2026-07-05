// sgemm_step2_register_tiling.cpp
//
// Step 2: block tiling in shared memory plus register (thread-level) tiling.
// Each thread computes a kThreadM x kThreadN micro-tile of C, holding operands in
// registers to amortize shared-memory bandwidth. Assumes dimensions divisible
// by the block tile sizes (true for the 1024 case used here).
#include "gemm_common.h"

namespace {

constexpr int kBlockM = 64;  // block tile rows of C
constexpr int kBlockN = 64;  // block tile cols of C
constexpr int kBlockK = 16;  // block tile depth
constexpr int kThreadM = 4;   // per-thread rows
constexpr int kThreadN = 4;   // per-thread cols
// Threads per block: (kBlockM/kThreadM) x (kBlockN/kThreadN) = 16 x 16 = 256.

__global__ void sgemmRegisterTiling(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float s_a[kBlockK][kBlockM];  // transposed A tile for coalesced register reads
  __shared__ float s_b[kBlockK][kBlockN];

  const int thread_col = threadIdx.x % (kBlockN / kThreadN);
  const int thread_row = threadIdx.x / (kBlockN / kThreadN);
  const int block_row = blockIdx.y * kBlockM;
  const int block_col = blockIdx.x * kBlockN;
  const int threads_per_block = (kBlockM / kThreadM) * (kBlockN / kThreadN);

  float acc[kThreadM][kThreadN] = {};

  for (int t = 0; t < k; t += kBlockK) {
    // Cooperatively load A (kBlockM x kBlockK) into s_a transposed, and B (kBlockK x kBlockN).
    for (int idx = threadIdx.x; idx < kBlockM * kBlockK; idx += threads_per_block) {
      int r = idx / kBlockK;
      int col = idx % kBlockK;
      s_a[col][r] = a[(block_row + r) * k + (t + col)];
    }
    for (int idx = threadIdx.x; idx < kBlockK * kBlockN; idx += threads_per_block) {
      int r = idx / kBlockN;
      int col = idx % kBlockN;
      s_b[r][col] = b[(t + r) * n + (block_col + col)];
    }
    __syncthreads();

#pragma unroll
    for (int p = 0; p < kBlockK; ++p) {
      float reg_a[kThreadM];
      float reg_b[kThreadN];
#pragma unroll
      for (int i = 0; i < kThreadM; ++i) reg_a[i] = s_a[p][thread_row * kThreadM + i];
#pragma unroll
      for (int j = 0; j < kThreadN; ++j) reg_b[j] = s_b[p][thread_col * kThreadN + j];
#pragma unroll
      for (int i = 0; i < kThreadM; ++i)
#pragma unroll
        for (int j = 0; j < kThreadN; ++j) acc[i][j] += reg_a[i] * reg_b[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kThreadM; ++i)
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      int row = block_row + thread_row * kThreadM + i;
      int col = block_col + thread_col * kThreadN + j;
      if (row < m && col < n) c[row * n + col] = acc[i][j];
    }
}

}  // namespace

int main() {
  constexpr int kSizeM = 1024;
  constexpr int kSizeN = 1024;
  constexpr int kSizeK = 1024;
  const int threads_per_block = (kBlockM / kThreadM) * (kBlockN / kThreadN);
  dim3 block(threads_per_block);
  dim3 grid(kSizeN / kBlockN, kSizeM / kBlockM);
  return gklab::runGemm("sgemm_step2_register_tiling", kSizeM, kSizeN, kSizeK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmRegisterTiling, grid, block, 0, buf.a, buf.b, buf.c, kSizeM, kSizeN, kSizeK);
  });
}
