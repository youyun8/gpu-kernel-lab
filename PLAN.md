# PLAN

site map、章節大綱與建置 checklist。完成的項目以 `- [x]` 標記。

## Site map

- `/` 首頁:hero、目標讀者 / 環境需求 / 數據誠實三卡、10 分鐘快速開始、學習路線圖 (Track A→D)。
- `/roadmap` 學習路線圖完整頁。
- `/chapters/[slug]` 21 章動態路由,含側邊目錄與 prev/next 導覽。

## 互動元件

- [x] `MemoryCoalescingVisualizer` — stride/offset → transactions 與 bandwidth efficiency
- [x] `OccupancyCalculator` — regs/smem/blockSize/arch → occupancy + limiter + 曲線
- [x] `TilingAnimator` — GEMM tiling 動畫 + reuse factor
- [x] `RooflineChart` — 互動 log-log roofline,載入 benchmark JSON
- [x] `BenchmarkComparison` — GEMM 各步 bar chart,CUDA/ROCm 切換
- [x] `Quiz` — 選擇題,答錯顯示解釋
- [x] `PlatformTabs` — CUDA/HIP 程式碼並排切換
- [x] 輔助元件:`Callout`、`LabBox`、`FurtherReading`

## Track A — 入門

- [x] Ch1 GPU 是什麼樣的機器? (lab: 01-basics)
- [x] Ch2 第一個 kernel (lab: 01-basics/vector_add)
- [x] Ch3 Memory hierarchy 全景圖 (lab: 02-memory/bandwidth_probe)
- [x] Ch4 如何測量效能 (lab: 02-memory)
- [x] Ch5 Roofline model (lab: 02-memory/roofline_probe)

## Track B — 進階

- [x] Ch6 Memory coalescing (lab: 02-memory/stride_bandwidth)
- [x] Ch7 Shared memory 與 bank conflicts (lab: 02-memory/transpose)
- [x] Ch8 Occupancy 與 latency hiding (lab: 01-basics/occupancy_experiment)
- [x] Ch9 Warp-level programming (lab: 04-reductions-softmax/warp_reduce)
- [x] Ch10 Reduction 與 softmax (lab: 04-reductions-softmax)
- [x] Ch11 Instruction-level 優化 (lab: 02-memory/vectorized_copy, saxpy_unroll)

## Track C — 專家

- [x] Ch12 GEMM 優化全流程 (lab: 03-gemm/sgemm_step0–4)
- [x] Ch13 小矩陣與尾端效應 (lab: 05-advanced-scheduling/split_k_gemm, cta_swizzle)
- [x] Ch14 Warp specialization 與 pipelines (lab: 05-advanced-scheduling/async_pipeline)
- [x] Ch15 認識 kernel library 生態 (lab: 03-gemm)
- [x] Ch16 Profiling 深入 (lab: 05-advanced-scheduling + scripts/profile_*)

## Track D — PyTorch 實戰

- [x] Ch17 找到值得優化的 kernel (lab: 06/profile_model.py)
- [x] Ch18 第一個 custom extension (lab: 06/cpp_extension)
- [x] Ch19 fused elementwise + reduction (lab: 06/load_inline)
- [x] Ch20 Triton 入門與比較 (lab: 06/triton)
- [x] Ch21 整合回訓練/推論流程 (lab: 06/custom_op_autograd)

## kernels

- [x] common:gpu_portability.h、benchmark.h、README
- [x] 01-basics:vector_add、occupancy_experiment
- [x] 02-memory:bandwidth_probe、stride_bandwidth、roofline_probe、transpose、vectorized_copy、saxpy_unroll
- [x] 03-gemm:sgemm_step0–4、gemm_common.h、README (WMMA/MFMA 指引)
- [x] 04-reductions-softmax:reduction、warp_reduce、softmax
- [x] 05-advanced-scheduling:split_k_gemm、cta_swizzle、async_pipeline
- [x] 06-pytorch-integration:profile_model、cpp_extension、load_inline、triton、custom_op_autograd

## 基礎建設

- [x] scripts:bench_all.py、profile_ncu.sh、profile_rocprof.sh
- [x] docker:Dockerfile.cuda、Dockerfile.rocm
- [x] CI:.github/workflows/deploy.yml (typecheck + lint + build + Pages 部署)
- [x] docs:README、DECISIONS、VERIFICATION_LOG、LICENSE、.gitignore

## 驗證 gate

- [x] website:tsc、lint、build (static export,26 頁) 全通過
- [x] kernels:CMake configure + build (HIP) 成功,19 target 實機執行通過 correctness
- [x] Python:py_compile 全通過
- [x] Final Audit (見 VERIFICATION_LOG.md)

## 練習與解答 (Exercises)

- [x] Exercise / Solution 元件(collapsible <details>,鍵盤可操作)+ 併入 mdx-components
- [x] `/exercises` 索引頁 + `/exercises/[track]` 動態路由 + header/首頁連結
- [x] Track A 練習(8 題,paper + programming)含完整解答
- [x] Track B 練習(9 題)含完整解答
- [x] Track C 練習(8 題)含完整解答
- [x] Track D 練習(7 題)含完整解答
- [x] programming 參考解 kernels:exercises/ex_a_saxpy、ex_b_block_reduce (實機 build + run 通過)
