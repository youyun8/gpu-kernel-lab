// sgemm_step3_vectorized.cpp
//
// Step 3: same block/register tiling as step 2, but global loads of B use
// float4 (128-bit) vectorized access to cut load instructions and improve
// memory throughput. B tile width is a multiple of 4.
#include "gemm_common.h"

namespace {

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 16;
constexpr int kThreadM = 4;
constexpr int kThreadN = 4;

__global__ void sgemmVectorized(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float s_a[kBlockK][kBlockM];
  __shared__ float s_b[kBlockK][kBlockN];

  const int thread_col = threadIdx.x % (kBlockN / kThreadN);
  const int thread_row = threadIdx.x / (kBlockN / kThreadN);
  const int block_row = blockIdx.y * kBlockM;
  const int block_col = blockIdx.x * kBlockN;
  const int threads_per_block = (kBlockM / kThreadM) * (kBlockN / kThreadN);

  float acc[kThreadM][kThreadN] = {};

  for (int t = 0; t < k; t += kBlockK) {
    for (int idx = threadIdx.x; idx < kBlockM * kBlockK; idx += threads_per_block) {
      int r = idx / kBlockK;
      int col = idx % kBlockK;
      s_a[col][r] = a[(block_row + r) * k + (t + col)];
    }
    // Vectorized load of B: each float4 covers 4 consecutive columns.
    for (int idx = threadIdx.x; idx < (kBlockK * kBlockN) / 4; idx += threads_per_block) {
      int r = idx / (kBlockN / 4);
      int col4 = idx % (kBlockN / 4);
      const float4 v = reinterpret_cast<const float4*>(b)[((t + r) * n + block_col) / 4 + col4];
      s_b[r][col4 * 4 + 0] = v.x;
      s_b[r][col4 * 4 + 1] = v.y;
      s_b[r][col4 * 4 + 2] = v.z;
      s_b[r][col4 * 4 + 3] = v.w;
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
  return gklab::runGemm("sgemm_step3_vectorized", kSizeM, kSizeN, kSizeK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmVectorized, grid, block, 0, buf.a, buf.b, buf.c, kSizeM, kSizeN, kSizeK);
  });
}
