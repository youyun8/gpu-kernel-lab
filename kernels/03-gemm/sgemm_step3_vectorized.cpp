// sgemm_step3_vectorized.cpp
//
// Step 3: same block/register tiling as step 2, but global loads of B use
// float4 (128-bit) vectorized access to cut load instructions and improve
// memory throughput. B tile width is a multiple of 4.
#include "gemm_common.h"

namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;

__global__ void sgemmVectorized(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float sA[kBK][kBM];
  __shared__ float sB[kBK][kBN];

  const int threadCol = threadIdx.x % (kBN / kTN);
  const int threadRow = threadIdx.x / (kBN / kTN);
  const int blockRow = blockIdx.y * kBM;
  const int blockCol = blockIdx.x * kBN;
  const int threadsPerBlock = (kBM / kTM) * (kBN / kTN);

  float acc[kTM][kTN] = {};

  for (int t = 0; t < k; t += kBK) {
    for (int idx = threadIdx.x; idx < kBM * kBK; idx += threadsPerBlock) {
      int r = idx / kBK;
      int col = idx % kBK;
      sA[col][r] = a[(blockRow + r) * k + (t + col)];
    }
    // Vectorized load of B: each float4 covers 4 consecutive columns.
    for (int idx = threadIdx.x; idx < (kBK * kBN) / 4; idx += threadsPerBlock) {
      int r = idx / (kBN / 4);
      int col4 = idx % (kBN / 4);
      const float4 v = reinterpret_cast<const float4*>(b)[((t + r) * n + blockCol) / 4 + col4];
      sB[r][col4 * 4 + 0] = v.x;
      sB[r][col4 * 4 + 1] = v.y;
      sB[r][col4 * 4 + 2] = v.z;
      sB[r][col4 * 4 + 3] = v.w;
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
  return gklab::runGemm("sgemm_step3_vectorized", kM, kN, kK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmVectorized, grid, block, 0, buf.a, buf.b, buf.c, kM, kN, kK);
  });
}
