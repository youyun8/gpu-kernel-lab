// autotune_register_tiling.cpp
//
// Chapter 30 reference solution: build several compile-time register-tiling
// variants and benchmark them under one executable. This is a minimal
// autotuning loop: each candidate is a specialized kernel with different
// per-thread micro-tile shape.
#include "gemm_common.h"

namespace {

constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kBlockK = 16;

#define DEFINE_SGEMM_CONFIG(NAME, TM_VALUE, TN_VALUE)                                           \
  __global__ void NAME(const float* a, const float* b, float* c, int m, int n, int k) {          \
    constexpr int kThreadM = TM_VALUE;                                                               \
    constexpr int kThreadN = TN_VALUE;                                                               \
    constexpr int kThreads = (kBlockM / kThreadM) * (kBlockN / kThreadN);                                         \
    __shared__ float s_a[kBlockM][kBlockK];                                                              \
    __shared__ float s_b[kBlockK][kBlockN];                                                              \
    int thread_col = threadIdx.x % (kBlockN / kThreadN);                                                  \
    int thread_row = threadIdx.x / (kBlockN / kThreadN);                                                  \
    int block_row = blockIdx.y * kBlockM;                                                            \
    int block_col = blockIdx.x * kBlockN;                                                            \
    float acc[kThreadM][kThreadN] = {};                                                                   \
    for (int t = 0; t < k; t += kBlockK) {                                                          \
      for (int idx = threadIdx.x; idx < kBlockM * kBlockK; idx += kThreads) {                           \
        int r = idx / kBlockK;                                                                      \
        int col = idx % kBlockK;                                                                    \
        int gr = block_row + r;                                                                  \
        int gc = t + col;                                                                       \
        s_a[r][col] = (gr < m && gc < k) ? a[gr * k + gc] : 0.0f;                                \
      }                                                                                         \
      for (int idx = threadIdx.x; idx < kBlockK * kBlockN; idx += kThreads) {                           \
        int r = idx / kBlockN;                                                                      \
        int col = idx % kBlockN;                                                                    \
        int gr = t + r;                                                                         \
        int gc = block_col + col;                                                                \
        s_b[r][col] = (gr < k && gc < n) ? b[gr * n + gc] : 0.0f;                                \
      }                                                                                         \
      __syncthreads();                                                                          \
      for (int p = 0; p < kBlockK; ++p) {                                                           \
        float reg_a[kThreadM];                                                                        \
        float reg_b[kThreadN];                                                                        \
        for (int i = 0; i < kThreadM; ++i) reg_a[i] = s_a[thread_row * kThreadM + i][p];                     \
        for (int j = 0; j < kThreadN; ++j) reg_b[j] = s_b[p][thread_col * kThreadN + j];                     \
        for (int i = 0; i < kThreadM; ++i)                                                           \
          for (int j = 0; j < kThreadN; ++j) acc[i][j] += reg_a[i] * reg_b[j];                         \
      }                                                                                         \
      __syncthreads();                                                                          \
    }                                                                                           \
    for (int i = 0; i < kThreadM; ++i) {                                                             \
      for (int j = 0; j < kThreadN; ++j) {                                                           \
        int row = block_row + thread_row * kThreadM + i;                                               \
        int col = block_col + thread_col * kThreadN + j;                                               \
        if (row < m && col < n) c[row * n + col] = acc[i][j];                                   \
      }                                                                                         \
    }                                                                                           \
  }

DEFINE_SGEMM_CONFIG(sgemmTm2Tn2, 2, 2)
DEFINE_SGEMM_CONFIG(sgemmTm4Tn4, 4, 4)
DEFINE_SGEMM_CONFIG(sgemmTm8Tn4, 8, 4)

#undef DEFINE_SGEMM_CONFIG

}  // namespace

int main() {
  constexpr int kSizeM = 256;
  constexpr int kSizeN = 256;
  constexpr int kSizeK = 256;
  dim3 grid((kSizeN + kBlockN - 1) / kBlockN, (kSizeM + kBlockM - 1) / kBlockM);

  int status = EXIT_SUCCESS;

  {
    constexpr int kThreadM = 2;
    constexpr int kThreadN = 2;
    constexpr int threads = (kBlockM / kThreadM) * (kBlockN / kThreadN);
    dim3 block(threads);
    status |= gklab::runGemm("autotune_tm2_tn2", kSizeM, kSizeN, kSizeK, [&](const gklab::GemmBuffers& buf) {
      GPU_LAUNCH(sgemmTm2Tn2, grid, block, 0, buf.a, buf.b, buf.c, kSizeM, kSizeN, kSizeK);
    });
  }

  {
    constexpr int kThreadM = 4;
    constexpr int kThreadN = 4;
    constexpr int threads = (kBlockM / kThreadM) * (kBlockN / kThreadN);
    dim3 block(threads);
    status |= gklab::runGemm("autotune_tm4_tn4", kSizeM, kSizeN, kSizeK, [&](const gklab::GemmBuffers& buf) {
      GPU_LAUNCH(sgemmTm4Tn4, grid, block, 0, buf.a, buf.b, buf.c, kSizeM, kSizeN, kSizeK);
    });
  }

  {
    constexpr int kThreadM = 8;
    constexpr int kThreadN = 4;
    constexpr int threads = (kBlockM / kThreadM) * (kBlockN / kThreadN);
    dim3 block(threads);
    status |= gklab::runGemm("autotune_tm8_tn4", kSizeM, kSizeN, kSizeK, [&](const gklab::GemmBuffers& buf) {
      GPU_LAUNCH(sgemmTm8Tn4, grid, block, 0, buf.a, buf.b, buf.c, kSizeM, kSizeN, kSizeK);
    });
  }

  return status;
}
