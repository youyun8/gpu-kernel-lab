# kernels/exercises — programming 練習參考解

對應網站 `/exercises` 的 programming 題參考解答。與其他 kernel 一樣用 CMake 建置(CUDA/HIP 自動偵測),各自附 correctness check 與 benchmark 報告。

## Targets

- `ex_a_saxpy` — 練習 A7:SAXPY (`y = a*x + y`),memory-bound,報告 achieved bandwidth。
- `ex_b_block_reduce` — 練習 B9:block sum reduction 兩種收尾(shared-memory tree vs. tree + warp shuffle),對照正確性與時間。

## 建置與執行

```bash
cd kernels
cmake -B build -S . && cmake --build build -j
./build/exercises/ex_a_saxpy
./build/exercises/ex_b_block_reduce
```

其餘 programming 題(Track C 的 GEMM 階梯 / Split-K、Track D 的 PyTorch 題)直接使用既有的 `03-gemm`、`05-advanced-scheduling`、`06-pytorch-integration` 目錄,練習頁已標明對應路徑。
