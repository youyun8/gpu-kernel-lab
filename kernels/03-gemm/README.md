# 03-gemm — SGEMM 優化全流程

對應網站第 12 章。 每個 step 都對 CPU reference 做 correctness check, 並用共用 harness 報告 GFLOP/s 與 % of peak。

## 由 CMake 建置的 targets

- `sgemm_step0_naive` — 每個 output 一個 thread, 零 reuse, bandwidth-bound。
- `sgemm_step1_shared` — shared memory tiling。
- `sgemm_step2_register_tiling` — block tiling + 每 thread 算 kTM×kTN micro-tile。
- `sgemm_step3_vectorized` — 在 step 2 基礎上對 B 做 float4 vectorized load。
- `sgemm_step4_double_buffer` — double buffering / software pipelining。
- `quantized_epilogue` — fp32 accumulation + bias + per-column int8 requantization。
- `autotune_register_tiling` — 多個 compile-time register tile 變體的最小 autotuning loop。

全部使用 portability header, 同一份 `.cpp` 在 CUDA (nvcc) 與 HIP (hipcc) 皆可編譯。

## Step 5 與 library 比較 (平台特定, 不在預設 build)

Step 5 (Tensor Core WMMA / AMD MFMA) 與 Step 6 (cuBLAS / hipBLASLt 比較) 高度依賴各平台的 arch 與 library, 故不放進跨平台的預設 build, 以免在缺對應 SDK 的機器上 configure 失敗。 實作指引:

- **CUDA WMMA**: 使用 `#include <mma.h>` 與 `nvcuda::wmma` API, 以 `half` 輸入、`float` accumulator, tile 為 16×16×16。 需 `-arch=sm_70` 以上。
- **AMD MFMA**: 使用 `__builtin_amdgcn_mfma_f32_16x16x4f32` 等 intrinsics (或透過 rocWMMA), 需 CDNA 架構 (gfx908/gfx90a/gfx942)。
- **Library 參考**: CUDA 連結 `-lcublas` 呼叫 `cublasSgemm`; ROCm 連結 hipBLASLt 呼叫對應 API。 把回報的 GFLOP/s 當成 100% 基準。

若要啟用, 於對應平台安裝 SDK 後, 參考 `bench_all.py` 的 `--with-library` 說明擴充。
