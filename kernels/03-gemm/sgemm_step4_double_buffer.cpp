// sgemm_step4_double_buffer.cpp
//
// Step 4: double buffering (software pipelining). Two shared-memory buffers let
// the load of the next k-tile overlap with computation on the current tile,
// hiding global-memory latency behind the FMA work.
#include "gemm_common.h"

namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;

__global__ void sgemmDoubleBuffer(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float sA[2][kBK][kBM];
  __shared__ float sB[2][kBK][kBN];

  const int threadCol = threadIdx.x % (kBN / kTN);
  const int threadRow = threadIdx.x / (kBN / kTN);
  const int blockRow = blockIdx.y * kBM;
  const int blockCol = blockIdx.x * kBN;
  const int threadsPerBlock = (kBM / kTM) * (kBN / kTN);

  float acc[kTM][kTN] = {};

  auto loadTile = [&](int t, int stage) {
    for (int idx = threadIdx.x; idx < kBM * kBK; idx += threadsPerBlock) {
      int r = idx / kBK;
      int col = idx % kBK;
      sA[stage][col][r] = a[(blockRow + r) * k + (t + col)];
    }
    for (int idx = threadIdx.x; idx < kBK * kBN; idx += threadsPerBlock) {
      int r = idx / kBN;
      int col = idx % kBN;
      sB[stage][r][col] = b[(t + r) * n + (blockCol + col)];
    }
  };

  int stage = 0;
  loadTile(0, stage);
  __syncthreads();

  for (int t = 0; t < k; t += kBK) {
    const int nextT = t + kBK;
    const int nextStage = stage ^ 1;
    if (nextT < k) {
      loadTile(nextT, nextStage);  // prefetch next tile into the other buffer
    }

#pragma unroll
    for (int p = 0; p < kBK; ++p) {
      float regA[kTM];
      float regB[kTN];
#pragma unroll
      for (int i = 0; i < kTM; ++i) regA[i] = sA[stage][p][threadRow * kTM + i];
#pragma unroll
      for (int j = 0; j < kTN; ++j) regB[j] = sB[stage][p][threadCol * kTN + j];
#pragma unroll
      for (int i = 0; i < kTM; ++i)
#pragma unroll
        for (int j = 0; j < kTN; ++j) acc[i][j] += regA[i] * regB[j];
    }
    __syncthreads();
    stage = nextStage;
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
  return gklab::runGemm("sgemm_step4_double_buffer", kM, kN, kK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmDoubleBuffer, grid, block, 0, buf.a, buf.b, buf.c, kM, kN, kK);
  });
}
