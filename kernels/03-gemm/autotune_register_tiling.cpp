// autotune_register_tiling.cpp
//
// Chapter 30 reference solution: build several compile-time register-tiling
// variants and benchmark them under one executable. This is a minimal
// autotuning loop: each candidate is a specialized kernel with different
// per-thread micro-tile shape.
#include "gemm_common.h"

namespace {

constexpr int kBM = 32;
constexpr int kBN = 32;
constexpr int kBK = 16;

#define DEFINE_SGEMM_CONFIG(NAME, TM_VALUE, TN_VALUE)                                           \
  __global__ void NAME(const float* a, const float* b, float* c, int m, int n, int k) {          \
    constexpr int kTM = TM_VALUE;                                                               \
    constexpr int kTN = TN_VALUE;                                                               \
    constexpr int kThreads = (kBM / kTM) * (kBN / kTN);                                         \
    __shared__ float sA[kBM][kBK];                                                              \
    __shared__ float sB[kBK][kBN];                                                              \
    int threadCol = threadIdx.x % (kBN / kTN);                                                  \
    int threadRow = threadIdx.x / (kBN / kTN);                                                  \
    int blockRow = blockIdx.y * kBM;                                                            \
    int blockCol = blockIdx.x * kBN;                                                            \
    float acc[kTM][kTN] = {};                                                                   \
    for (int t = 0; t < k; t += kBK) {                                                          \
      for (int idx = threadIdx.x; idx < kBM * kBK; idx += kThreads) {                           \
        int r = idx / kBK;                                                                      \
        int col = idx % kBK;                                                                    \
        int gr = blockRow + r;                                                                  \
        int gc = t + col;                                                                       \
        sA[r][col] = (gr < m && gc < k) ? a[gr * k + gc] : 0.0f;                                \
      }                                                                                         \
      for (int idx = threadIdx.x; idx < kBK * kBN; idx += kThreads) {                           \
        int r = idx / kBN;                                                                      \
        int col = idx % kBN;                                                                    \
        int gr = t + r;                                                                         \
        int gc = blockCol + col;                                                                \
        sB[r][col] = (gr < k && gc < n) ? b[gr * n + gc] : 0.0f;                                \
      }                                                                                         \
      __syncthreads();                                                                          \
      for (int p = 0; p < kBK; ++p) {                                                           \
        float regA[kTM];                                                                        \
        float regB[kTN];                                                                        \
        for (int i = 0; i < kTM; ++i) regA[i] = sA[threadRow * kTM + i][p];                     \
        for (int j = 0; j < kTN; ++j) regB[j] = sB[p][threadCol * kTN + j];                     \
        for (int i = 0; i < kTM; ++i)                                                           \
          for (int j = 0; j < kTN; ++j) acc[i][j] += regA[i] * regB[j];                         \
      }                                                                                         \
      __syncthreads();                                                                          \
    }                                                                                           \
    for (int i = 0; i < kTM; ++i) {                                                             \
      for (int j = 0; j < kTN; ++j) {                                                           \
        int row = blockRow + threadRow * kTM + i;                                               \
        int col = blockCol + threadCol * kTN + j;                                               \
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
  constexpr int kM = 256;
  constexpr int kN = 256;
  constexpr int kK = 256;
  dim3 grid((kN + kBN - 1) / kBN, (kM + kBM - 1) / kBM);

  int status = EXIT_SUCCESS;

  {
    constexpr int kTM = 2;
    constexpr int kTN = 2;
    constexpr int threads = (kBM / kTM) * (kBN / kTN);
    dim3 block(threads);
    status |= gklab::runGemm("autotune_tm2_tn2", kM, kN, kK, [&](const gklab::GemmBuffers& buf) {
      GPU_LAUNCH(sgemmTm2Tn2, grid, block, 0, buf.a, buf.b, buf.c, kM, kN, kK);
    });
  }

  {
    constexpr int kTM = 4;
    constexpr int kTN = 4;
    constexpr int threads = (kBM / kTM) * (kBN / kTN);
    dim3 block(threads);
    status |= gklab::runGemm("autotune_tm4_tn4", kM, kN, kK, [&](const gklab::GemmBuffers& buf) {
      GPU_LAUNCH(sgemmTm4Tn4, grid, block, 0, buf.a, buf.b, buf.c, kM, kN, kK);
    });
  }

  {
    constexpr int kTM = 8;
    constexpr int kTN = 4;
    constexpr int threads = (kBM / kTM) * (kBN / kTN);
    dim3 block(threads);
    status |= gklab::runGemm("autotune_tm8_tn4", kM, kN, kK, [&](const gklab::GemmBuffers& buf) {
      GPU_LAUNCH(sgemmTm8Tn4, grid, block, 0, buf.a, buf.b, buf.c, kM, kN, kK);
    });
  }

  return status;
}
