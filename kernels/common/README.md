# kernels/common

這裡是所有 kernel 範例共用的 header。

## 檔案

- `gpu_portability.h` — 一層薄薄的 portability layer,把 `gpuMalloc` / `gpuMemcpy...` / event API 對映到 CUDA (`cudaXxx`) 或 HIP (`hipXxx`)。build 系統會定義 `USE_CUDA` 或 `USE_HIP` 其中之一。
- `benchmark.h` — 計時 (warmup + GPU event)、正確性驗證 (`verifyClose`)、以及 achieved bandwidth / GFLOP/s / % of peak 的報告函式。

## 命名規範

全 repo 的 C++ 遵守:PascalCase 用於 class/struct(如 `BenchResult`)、camelCase 用於函式(如 `benchmarkKernel`)、`kCamelCase` 用於 constexpr/const(如 `kDefaultIters`)、snake_case 用於一般變數與參數。所有註解為英文。

## 驗證層級

若機器上有 CUDA 或 ROCm toolchain,`kernels/CMakeLists.txt` 會自動偵測並用 `nvcc` / `hipcc` 編譯。若兩者皆無,可用 syntax-only 檢查:

```bash
# HIP 語法檢查 (需要 ROCm/hipcc,但不需要 GPU)
hipcc -DUSE_HIP -std=c++17 -I kernels/common -fsyntax-only kernels/01-basics/vector_add.cpp

# 純 host 端 syntax 檢查 (無任何 GPU toolchain 時的最低驗證)
clang++ -DUSE_HIP -std=c++17 -I kernels/common -fsyntax-only ... (需要對應 headers)
```

`VERIFICATION_LOG.md` 會標註每個檔案實際達到的驗證層級。
