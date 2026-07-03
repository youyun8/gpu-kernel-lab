# VERIFICATION_LOG

每個階段的驗證結果與修復記錄 (依 prompt 的 Verification Gate 要求)。

## 2026-07-03 Update — 全量 clean re-verification (獨立重跑, 非信任前次 log)

環境: 8 × AMD Instinct MI355X (gfx950)、ROCm 7.2 (hipcc / AMD clang 22.0.0)、RCCL、cmake 3.31、g++ 11.4、PyTorch 2.9.1+rocm7.2.0 (8 GPUs visible)、Triton 3.6.0。

方法: 從乾淨 build dir 重新 `cmake -B <dir> -S kernels -DCMAKE_BUILD_TYPE=Release` (自動偵測 HIP) + `cmake --build -j`, 逐一執行所有產出的 executable, 不信任前次 log 的宣稱。

### Compiled kernels — 49/49 pass (修完 1 個 bug 後)

- Clean build 0 error, 產生 49 個 executable (tracks 00–07 + exercises + access-patterns)。
- 逐一實跑結果: 48 支首輪即 OK; **`05-advanced-scheduling/persistent_scheduler` 首輪 FAIL** (見下), 修復後 49/49 全 pass。
- `07-multi-gpu-collectives/allreduce_bench` (RCCL) 在 8 GPU 實跑, correctness OK on all ranks, busbw 由 ~35 GB/s 升至 ~366 GB/s (1MB→256MB)。
- data-race demos: 壞版仍可觀察 race, 修復版穩定正確 (與前次一致)。

### Bug 1 (已修) — persistent_scheduler 正確性錯誤

- 症狀: `tasks_persistent FAILED: task 0 got 204482 want 133693440` (static 版正確)。
- 根因: `runTasksPersistent` 讓 block 內**每個 thread** 都呼叫 `atomicAdd(nextTask, kChunkSize)`, 造成各 thread 拿到不同 chunk、control flow 分歧, 導致 `blockReduceSum` 內的 `__syncthreads()` 執行次數不一致 → reduction 損壞 (亦有 barrier hang 風險)。
- 修法: 僅 thread 0 做 atomic dequeue 寫入 `__shared__ int beginShared`, `__syncthreads()` 後全 block 讀同一 `begin`。 修復後 static/persistent 皆 correctness OK, 且 persistent 略快 (0.110 ms vs 0.125 ms), 正好印證章節論點。

### Bug 2 (已修) — 目錄名 `triton/` 遮蔽已安裝的 triton 套件

- 症狀: 依 README 於 `06-pytorch-integration/` 執行時, `import triton` 解析到 repo 目錄 (namespace package, `__file__=None`) 而非套件。 Triton 範例因此**誤報 "not installed; skipping"** (kernel 根本沒跑); 且該目錄下任何 torch 程式會在 `torch._dynamo` → `import triton.language` 崩潰 (`AttributeError`)。
- 修法: `git mv kernels/06-pytorch-integration/triton kernels/.../triton_examples`, 並更新 4 處引用 (README、`d20-triton-intro.mdx` 兩處含 GitHub 連結、`exercises/track-d.mdx`)。

### PyTorch integration (ch. 06) — 全 pass (CMake 外, 另跑)

於 documented 目錄實跑全部通過: `load_inline/fused_bias_gelu` (fused 2.66×, torch.compile 3.63×)、`triton_examples/softmax_triton` (max abs err 7.5e-09)、`triton_examples/matmul_triton` (8.4e-05)、`custom_op_autograd/test_custom_op` (forward + torch.compile OK)、`profile_model` (輸出 trace)、`cpp_extension/test_gelu` (JIT 以 `--offload-arch=gfx950` 編譯, forward/backward max abs err 0)。

### 清理 (stale files)

移除 stale 產物: 舊 `build/` (6.7M)、`website/.next`、`website/out`、`website/tsconfig.tsbuildinfo`、`website/next-env.d.ts`、`kernels/06-pytorch-integration/trace.json`、以及 JIT hipify 產物 `gelu_kernel.hip` 與 test 產物 `baseline.json`。 保留 `node_modules` (live deps) 與 `.codex` (tooling state)。

## 2026-07-03 Update — 收尾批次 (任務 8 補完 + 全面實機驗證)

環境: ROCm 7.2 (hipcc / AMD clang 22)、64 × AMD Instinct MI355X (gfx950)、RCCL `librccl.so.1` (7.2.0)、cmake 3.31.10、g++ 11.4、OpenMP、pthread。 Node v22 (容器內既有), npm 以 registry.npmjs.org tarball bootstrap 出 10.9.2。

### 全域 gate

- 網站 `npm run build` (`output: 'export'`): 通過, 產生 45 靜態頁 (含 34 章 + /exercises + roadmap)。 `npm run typecheck` (tsc --noEmit) 與 `npm run lint` (next lint): 皆通過, 0 warning/error。 加上 basePath 的既有結構未動。
- `grep -rn "your-org"` 於 website/kernels/scripts/docker/README/.github: 0 命中 (僅 prompt 檔本身提到字串)。
- Kernel 全量 build: `cmake -B build -S kernels -DCMAKE_BUILD_TYPE=Release` configure 通過, 平台自動偵測為 HIP; `cmake --build build -j` 成功產生 **56 個 executable**, 無 error。 RCCL / OpenMP / pthread 皆被找到 (無 skip 訊息)。

### 任務 8 (Occupancy Tuning Levers)

- 新增 `kernels/03-occupancy/tuning_levers_demo.cpp` + `CMakeLists.txt`, 已納入上層 `kernels/CMakeLists.txt`。 HIP 實機 build + run 通過, 兩段 correctness check `max relative error 1.192e-07`。
- 實跑輸出 (MI355X, HIP_VISIBLE_DEVICES=0): 三種 block size 與有無 `__launch_bounds__` 均報 100% occupancy (該 GPU register file 極大, 資源未吃緊), 但計時顯示 `low_occ_high_ilp_tiled` **0.067 ms** vs `high_occupancy_scalar` **0.118 ms** (約 1.8×), 印證「in-flight 工作量 (ILP) 決定 latency hiding, 非 occupancy 數字」。
- 章節 `b8-occupancy-latency-hiding.mdx`: 五個旋鈕現各有 before/after code fence (block size / register cap / tile size / accumulators / grid size); stall 診斷表後補上 decision-tree 文字 flowchart, 每葉連回對應 Lever 的 code; `OccupancyCalculator` widget 已嵌入 (line 40); LabBox 改指 `kernels/03-occupancy`; Solution 貼上實機數字。

### 任務 5 (平行化 / data race) — 硬性 gate: 壞版不穩定、正確版穩定

- 修正 `race_demo.cpp`: 改用 `volatile long long` 讓 race 在 Release (-O2) 下仍可觀察, 結論改為依實際 wrong_trials 判斷 (不再無條件宣稱 unstable)。
- 實機執行 (256 核):
  - `race_demo` (壞版): 5/5 trials 損失 ~78–84% 更新, 每次不同 → `5/5 trials lost updates -> data race`。
  - `race_fixed_mutex` / `race_fixed_atomic`: 每 trial 皆 = 8000000, 穩定正確。
  - `openmp_reduction`: racy=WRONG (每次不同), `critical`=正確但慢 (~12 s), `reduction`=正確且快 (~10 ms)。
  - `pthread_race`: racy 版每 trial 不同, mutex 版全部 8000000。
  - GPU twins: `gpu_histogram_race` racy 版 counted ~400/4194304 (WRONG), atomic 版 4194304/4194304 (OK); `gpu_block_reduce` / `gpu_threadfence` correctness OK。

### 任務 6 (NCCL/RCCL) — 硬性 gate: 實機 busbw

- `allreduce_bench` (RCCL, `ncclCommInitAll`, single-process multi-device) 在 **8 × MI355X** 實跑, correctness OK on all ranks。 真實一輪輸出 (float sum, 20 timed iters/size):

  | size(B) | time(us) | algbw(GB/s) | busbw(GB/s) |
  | --- | --- | --- | --- |
  | 1048576 | 53.2 | 19.69 | 34.47 |
  | 2097152 | 52.3 | 40.09 | 70.15 |
  | 4194304 | 57.0 | 73.58 | 128.77 |
  | 8388608 | 78.6 | 106.69 | 186.70 |
  | 16777216 | 126.3 | 132.79 | 232.37 |
  | 33554432 | 198.2 | 169.32 | 296.31 |
  | 67108864 | 348.1 | 192.81 | 337.41 |
  | 134217728 | 657.0 | 204.27 | 357.48 |
  | 268435456 | 1279.0 | 209.87 | 367.28 |

  busbw 依 all-reduce 公式 `busbw = algbw * 2*(n-1)/n` (n=8) 計算; 隨 message size 增大趨近互連峰值, 符合 ring all-reduce bandwidth-bound 的預期。

### 任務 1/2 抽樣實機驗證 (b6 access-patterns, 前批次未實跑)

於 MI355X 實跑 `kernels/02-memory/access-patterns/` 全部 6 支, 皆正常且呈現預期對照:
- `aosoa_layout`: AoS strided (1956 GB/s) < SoA (4905) ≈ AoSoA chunk32 (5138)。
- `vectorized_tail`: scalar/float4 皆 correctness OK, 尾端 3 個 scalar 由 cleanup kernel 處理。
- `cooperative_tile_load`: shared tile (2755 GB/s) 明顯快於 naive 17-load (1032), correctness OK。
- `gather_scatter`: gather (4516 GB/s) 快; scatter naive (27 GB/s) 慢, block aggregation (108 GB/s) 改善。
- `row_padding`: 128B 對齊 ld=1024 (8039 GB/s) 快於未對齊 ld=1000 (5994)。
- `embedding_lookup`: sorted (5344 GB/s) 快於 random (4200), inverse permutation 還原順序正確。

(% of peak 仍以 illustrative 常數計, 故偶見 >100%, 屬預期; 讀者應替換自身硬體峰值。)

## 2026-07-02 Update (previous container)

- Current tools: Node.js v22.22.2, npm 10.9.7, `/usr/bin/g++` present.
- Current missing tools: `cmake`, `hipcc`, and `nvcc` are not installed in this container, so the expanded kernel CMake build could not be executed here.
- Curriculum expansion:
  - Chapters now cover 32 total chapters across Track A-F.
  - Existing chapters 1-25 were enriched with additional optimization techniques, design checklists, failure modes, profiling recipes, and concrete examples.
  - Added Track F chapters 26-32: atomics/histograms, persistent kernels, stream/graph pipeline, low precision/quantization/layouts, autotuning/codegen, correctness/debugging, and a full optimization checklist.
- Programming solutions added to matching directories:
  - `kernels/04-reductions-softmax/histogram_atomics.cpp`.
  - `kernels/05-advanced-scheduling/persistent_scheduler.cpp`.
  - `kernels/05-advanced-scheduling/stream_pipeline.cpp`.
  - `kernels/03-gemm/quantized_epilogue.cpp`.
  - `kernels/03-gemm/autotune_register_tiling.cpp`.
  - `kernels/exercises/ex_c_tail_correctness.cpp`.
- Site/rendering updates:
  - Chapter quizzes render as native `<details>` and are collapsed by default.
  - References render in IEEE-style numbered form with title/source text, `[Online]`, `Available:`, and `Accessed: Jul. 2, 2026`.
  - Markdown label issues with a trailing space before the closing bold marker were normalized, for example the Naive reduction label now renders as `**Naive:**`.
  - Title casing and Chinese/English/punctuation spacing were normalized across visible chapter metadata and content.
- Verification performed in this environment:
  - `npm run typecheck`: passed.
  - `npm run lint`: passed.
  - `npm run build`: passed, generating 43 static pages.
  - Static export check: 32 chapter pages produced under `website/out/chapters`.
  - Quiz export check: 32 chapter pages contain `<details aria-label="章末測驗">`; 0 exported chapter pages contain an open quiz `<details>`.
  - Reference export check: 64 exported chapter artifacts contain `Accessed: Jul. 2, 2026` in rendered citations.
  - `git diff --check`: passed.
  - Kernel build: blocked by missing `cmake` and GPU toolchains in the current container.

## 環境

- Node.js v22、npm 10。
- CMake 3.31。
- ROCm 7.2 (hipcc / clang++）， 偵測到 AMD GPU, kernels 可實機執行。
- 無 nvcc: CUDA 路徑為 portability header 與 CMake 的結構性支援, 未實機編譯。
- Python 3.10。

## 網站 pipeline

- `npx tsc --noEmit`: 通過, 無 TypeScript 錯誤。
- `npm run lint` (next lint): 通過, 無 warning / error。
- `npm run build` (output: 'export'): 成功, 產生 26 個靜態頁 (首頁、roadmap、_not-found、21 章 + slug 索引)。 以 `NEXT_PUBLIC_BASE_PATH=/gpu-kernel-lab` build 亦成功。
- `out/` 檢查: 21 章頁面皆存在; asset 連結帶 `/gpu-kernel-lab` basePath 前綴; chapter 頁含互動元件 markup (Memory Coalescing Visualizer、平台程式碼切換、章末測驗); KaTeX 數學已 render。
- 修復記錄:
  - `PlatformTabs` 原設計用 JSX attribute 傳 fenced code, MDX 無法解析 → 改為 children/`<Platform>` API, 並重寫 6 章受影響檔案。
  - `c12` 內文的 `<5%` 被 MDX 當成 JSX 標籤 → 改寫為「不到 5%」。
  - `PlatformTabs.tsx` 一度重複定義 → 重寫單一實作, tsc 通過。

## 語言規則自我檢查

- 簡體字掃描: 自訂 script 掃 46 個內容/元件/文件檔, 比對常見簡體字集合, 0 命中。
- 技術名詞: kernel、warp、wavefront、shared memory/LDS、occupancy、coalescing、bank conflict、GEMM、Tensor Core、MFMA、roofline、arithmetic intensity 等維持英文, 未翻成中文。
- code comments: 所有 kernel 與元件內註解為英文 (人工抽查 + 撰寫時遵循)。
- placeholder 掃描: TODO / TBD / lorem / 暫定 / FIXME, 0 命中。

## 互動元件

- 7 個必做元件全部實作並在 build 輸出中 render: MemoryCoalescingVisualizer、OccupancyCalculator、TilingAnimator、RooflineChart、BenchmarkComparison、Quiz、PlatformTabs (另含 Callout / LabBox / FurtherReading 輔助元件)。
- 皆有預設 props / 內建資料, 無 runtime 需求; 鍵盤可操作 (radio、button、range、select、tablist role/aria)。

## kernels

- CMake configure: `cmake -B build -S kernels` 成功, 自動偵測平台為 HIP。
- CMake build: `cmake --build build -j` 成功, 產生 19 個 executable。
- syntax-only 檢查: 19 個 `.cpp` 以 `hipcc -DUSE_HIP -fsyntax-only` 全部通過。
- 實機執行 (ROCm GPU): 以下全部通過 correctness check —
  - 01-basics: vector_add、occupancy_experiment
  - 02-memory: bandwidth_probe、stride_bandwidth、roofline_probe、transpose (naive/shared/padded)、vectorized_copy、saxpy_unroll
  - 03-gemm: sgemm_step0_naive、step1_shared、step2_register_tiling、step3_vectorized、step4_double_buffer
  - 04-reductions-softmax: reduction (shared / warp shuffle)、warp_reduce、softmax (three-pass / online, 含大值 overflow 測試)
  - 05-advanced-scheduling: split_k_gemm、cta_swizzle、async_pipeline
- 觀察到的合理現象 (供讀者對照, 非承諾數字): stride 上升時 bandwidth 遞減; float4 copy 明顯快於 scalar; saxpy unroll4 快於 unroll1; split-K 在瘦長矩陣快於 plain; online softmax 快於 three-pass。
- 命名規範: PascalCase (BenchResult / GemmBuffers)、camelCase (benchmarkKernel / warpReduceSum)、kCamelCase (kBlockSize / kTile)、snake_case (區域變數 / 參數)。 註解全英文。
- MDX 與 kernels 的一致性: 章節中的程式碼為對應 kernel 的節錄, 節錄處以註解標明 (例如 GEMM step2 標注 "full kernel in kernels/03-gemm/..."); 核心邏輯與檔案一致。

## PyTorch 章節 (06)

- `python -m py_compile` 通過: profile_model.py、cpp_extension/(setup.py、test_gelu.py)、load_inline/fused_bias_gelu.py、triton/(softmax_triton.py、matmul_triton.py)、custom_op_autograd/test_custom_op.py。
- 執行需 GPU 版 PyTorch (+ 第 20 章需 Triton), 本環境未安裝該套件, 故僅語法驗證。

## scripts / docker / CI

- `python -m py_compile scripts/bench_all.py`: 通過。
- `bash -n scripts/profile_ncu.sh` / `profile_rocprof.sh`: 通過。
- `.github/workflows/deploy.yml`: YAML 解析有效, 含 build (typecheck + lint + build + nojekyll + upload) 與 deploy 兩個 job。
- docker: 提供 CUDA 與 ROCm 兩個 Dockerfile (語法檢視, 未實際 docker build)。

## Final Audit summary

## Exercises 新增 (練習與解答)

- 新增 Exercise / Solution 元件 (Solution 用原生 `<details>`, 預設收合、鍵盤可操作、無 JS 亦可運作), 併入 `mdx-components.tsx`。
- 新增路由: `/exercises` 索引與 `/exercises/[slug]` 動態頁 (4 個 track), 於 header 與首頁加入連結。
- 內容: Track A (8)、Track B (9)、Track C (8)、Track D (7), 共 32 題, 混合 paper-and-pencil 與 programming。 每題附完整解答。
- programming 參考解 kernels: `kernels/exercises/ex_a_saxpy`、`ex_b_block_reduce`, 已在 ROCm 實機 build + run, correctness check 全通過 (saxpy 誤差 1.19e-7; 兩種 reduction 收尾皆通過)。 其餘 programming 題複用既有 03-gemm / 05-advanced-scheduling / 06-pytorch-integration 目錄。
- 驗證: website tsc / lint / build (含 basePath) 再次全通過, 靜態輸出新增 `/exercises` 與 4 個 track 頁面 (solution 以收合 `<details>` render); 簡體字掃描 0 命中、placeholder 掃描 0 命中。
- 修復記錄: `track-c.mdx` 內文的 `<1%` 被 MDX 當成 JSX 標籤 → 以 inline code 包成 `` `<1%` ``。

### 原 Final Audit summary

- 全站 tsc / lint / build 再次全通過 (含 basePath build)。
- PLAN.md 全項目已打勾且對應檔案存在 (21 章 + 7 元件 + 21 lab 目錄/檔案)。
- 全 repo 簡體字掃描 0 命中、placeholder 掃描 0 命中、內部 chapter 連結與 lab 路徑全部有效。
- 已知限制:
  1. CUDA 路徑未實機編譯 (環境無 nvcc); 以 HIP 實機驗證 + portability header/CMake 結構支援。
  2. GEMM Step 5 (WMMA/MFMA) 與 library (cuBLAS/hipBLASLt) 比較不在預設 build, 提供實作指引於 kernels/03-gemm/README.md。
  3. PyTorch (06) 章節僅 py_compile 語法驗證, 未在 GPU + PyTorch 環境實跑。
  4. 網站 benchmark 預設資料為示意數據 (illustrative), 需以 scripts/bench_all.py 產生真實數據替換。
