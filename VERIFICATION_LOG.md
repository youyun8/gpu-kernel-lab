# VERIFICATION_LOG

每個階段的驗證結果與修復記錄 (依 prompt 的 Verification Gate 要求)。

## 環境

- Node.js v22、npm 10。
- CMake 3.31。
- ROCm 7.2 (hipcc / clang++）， 偵測到 AMD GPU, kernels 可實機執行。
- 無 nvcc: CUDA 路徑為 portability header 與 CMake 的結構性支援, 未實機編譯。
- Python 3.10。

## 網站 pipeline

- `npx tsc --noEmit`: 通過, 無 TypeScript 錯誤。
- `npm run lint` (next lint): 通過, 無 warning / error。
- `npm run build` (output: 'export'): 成功, 產生 26 個靜態頁 (首頁、roadmap、_not-found、21 章 + slug 索引)。以 `NEXT_PUBLIC_BASE_PATH=/gpu-kernel-lab` build 亦成功。
- `out/` 檢查: 21 章頁面皆存在;asset 連結帶 `/gpu-kernel-lab` basePath 前綴;chapter 頁含互動元件 markup (Memory Coalescing Visualizer、平台程式碼切換、章末測驗);KaTeX 數學已 render。
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
- 皆有預設 props / 內建資料, 無 runtime 需求;鍵盤可操作 (radio、button、range、select、tablist role/aria)。

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
- 觀察到的合理現象 (供讀者對照, 非承諾數字): stride 上升時 bandwidth 遞減;float4 copy 明顯快於 scalar;saxpy unroll4 快於 unroll1;split-K 在瘦長矩陣快於 plain;online softmax 快於 three-pass。
- 命名規範: PascalCase (BenchResult / GemmBuffers)、camelCase (benchmarkKernel / warpReduceSum)、kCamelCase (kBlockSize / kTile)、snake_case (區域變數 / 參數)。註解全英文。
- MDX 與 kernels 的一致性: 章節中的程式碼為對應 kernel 的節錄, 節錄處以註解標明 (例如 GEMM step2 標注 "full kernel in kernels/03-gemm/...");核心邏輯與檔案一致。

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
- 內容: Track A (8)、Track B (9)、Track C (8)、Track D (7), 共 32 題, 混合 paper-and-pencil 與 programming。每題附完整解答。
- programming 參考解 kernels: `kernels/exercises/ex_a_saxpy`、`ex_b_block_reduce`, 已在 ROCm 實機 build + run, correctness check 全通過 (saxpy 誤差 1.19e-7;兩種 reduction 收尾皆通過)。其餘 programming 題複用既有 03-gemm / 05-advanced-scheduling / 06-pytorch-integration 目錄。
- 驗證: website tsc / lint / build (含 basePath) 再次全通過, 靜態輸出新增 `/exercises` 與 4 個 track 頁面 (solution 以收合 `<details>` render);簡體字掃描 0 命中、placeholder 掃描 0 命中。
- 修復記錄: `track-c.mdx` 內文的 `<1%` 被 MDX 當成 JSX 標籤 → 以 inline code 包成 `` `<1%` ``。

### 原 Final Audit summary

- 全站 tsc / lint / build 再次全通過 (含 basePath build)。
- PLAN.md 全項目已打勾且對應檔案存在 (21 章 + 7 元件 + 21 lab 目錄/檔案)。
- 全 repo 簡體字掃描 0 命中、placeholder 掃描 0 命中、內部 chapter 連結與 lab 路徑全部有效。
- 已知限制:
  1. CUDA 路徑未實機編譯 (環境無 nvcc);以 HIP 實機驗證 + portability header/CMake 結構支援。
  2. GEMM Step 5 (WMMA/MFMA) 與 library (cuBLAS/hipBLASLt) 比較不在預設 build, 提供實作指引於 kernels/03-gemm/README.md。
  3. PyTorch (06) 章節僅 py_compile 語法驗證, 未在 GPU + PyTorch 環境實跑。
  4. 網站 benchmark 預設資料為示意數據 (illustrative), 需以 scripts/bench_all.py 產生真實數據替換。
