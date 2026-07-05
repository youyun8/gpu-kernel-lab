// sgemm_step4_double_buffer.cpp
//
// Step 4: double buffering (software pipelining). Two shared-memory buffers let
// the load of the next k-tile overlap with computation on the current tile,
// hiding global-memory latency behind the FMA work.
#include "gemm_common.h"

namespace {

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 16;
constexpr int kThreadM = 4;
constexpr int kThreadN = 4;

__global__ void sgemmDoubleBuffer(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float s_a[2][kBlockK][kBlockM];
  __shared__ float s_b[2][kBlockK][kBlockN];

  const int thread_col = threadIdx.x % (kBlockN / kThreadN);
  const int thread_row = threadIdx.x / (kBlockN / kThreadN);
  const int block_row = blockIdx.y * kBlockM;
  const int block_col = blockIdx.x * kBlockN;
  const int threads_per_block = (kBlockM / kThreadM) * (kBlockN / kThreadN);

  float acc[kThreadM][kThreadN] = {};

  auto loadTile = [&](int t, int stage) {
    for (int idx = threadIdx.x; idx < kBlockM * kBlockK; idx += threads_per_block) {
      int r = idx / kBlockK;
      int col = idx % kBlockK;
      s_a[stage][col][r] = a[(block_row + r) * k + (t + col)];
    }
    for (int idx = threadIdx.x; idx < kBlockK * kBlockN; idx += threads_per_block) {
      int r = idx / kBlockN;
      int col = idx % kBlockN;
      s_b[stage][r][col] = b[(t + r) * n + (block_col + col)];
    }
  };

  int stage = 0;
  loadTile(0, stage);
  __syncthreads();

  for (int t = 0; t < k; t += kBlockK) {
    const int next_t = t + kBlockK;
    const int next_stage = stage ^ 1;
    if (next_t < k) {
      loadTile(next_t, next_stage);  // prefetch next tile into the other buffer
    }

#pragma unroll
    for (int p = 0; p < kBlockK; ++p) {
      float reg_a[kThreadM];
      float reg_b[kThreadN];
#pragma unroll
      for (int i = 0; i < kThreadM; ++i) reg_a[i] = s_a[stage][p][thread_row * kThreadM + i];
#pragma unroll
      for (int j = 0; j < kThreadN; ++j) reg_b[j] = s_b[stage][p][thread_col * kThreadN + j];
#pragma unroll
      for (int i = 0; i < kThreadM; ++i)
#pragma unroll
        for (int j = 0; j < kThreadN; ++j) acc[i][j] += reg_a[i] * reg_b[j];
    }
    __syncthreads();
    stage = next_stage;
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
  return gklab::runGemm("sgemm_step4_double_buffer", kSizeM, kSizeN, kSizeK, [&](const gklab::GemmBuffers& buf) {
    GPU_LAUNCH(sgemmDoubleBuffer, grid, block, 0, buf.a, buf.b, buf.c, kSizeM, kSizeN, kSizeK);
  });
}
