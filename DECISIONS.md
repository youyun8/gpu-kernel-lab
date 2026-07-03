# DECISIONS

記錄開發過程中遇到的模稜兩可決策與理由 (依 prompt 要求)。

## 網站技術選型

- **@next/mdx 而非 contentlayer**: App Router + static export 下 `@next/mdx` 最直接、相依最少, 能在 MDX 內嵌 React 元件。 以一個 static import registry (`content/chapters/registry.ts`) 搭配 `generateStaticParams` 匯出全部 21 章。
- **rehype-pretty-code (shiki) 做語法高亮**: 符合 prompt「shiki 或 rehype-pretty-code」的要求, dark theme 預設 (`github-dark-dimmed`), 支援 CUDA C++ / HIP / Python。
- **Recharts 做互動圖表**: 比手寫 D3 快, SSR/static export 相容。 roofline 用 ComposedChart 疊 log-log 屋頂線與散點。
- **自寫 prose 樣式而非 @tailwindcss/typography**: 少一個相依, 對中英混排與 code block 的細節掌控更好。

## PlatformTabs 的 API

原本設計成 `cuda={...} hip={...}` 的 prop 形式, 但 MDX 無法在 JSX attribute 內解析 fenced code block。 改為 children 形式: `<PlatformTabs><Platform name="CUDA">```...```</Platform>...</PlatformTabs>`, 讓程式碼區塊維持標準 markdown。

## kernels 可攜性策略

- **單一 `.cpp` + portability header**: `common/gpu_portability.h` 把 `gpuMalloc` / event API 對映到 CUDA 或 HIP, 由 build 定義 `USE_CUDA` / `USE_HIP`。 多數 kernel (vector add、tiling、reduction、softmax) 因此一份原始碼兩平台通用。
- **CMake 用 first-class HIP language**: `check_language(HIP)` + `enable_language(HIP)`, 比手動指定 hipcc 穩健, configure 與 build 都通過。
- **Step 5 (WMMA/MFMA) 與 library 比較不進預設 build**: 這兩者高度依賴平台 arch 與 SDK, 放進跨平台預設 build 會在缺 SDK 的機器 configure/compile 失敗。 改以 `kernels/03-gemm/README.md` 提供完整實作指引, 維持「clone 就能 build」的體驗。
- **benchmark peak 用 illustrative 常數**: harness 的 % of peak 以 A100 等級的示意 peak 計算, 使用者應替換成自己硬體的數字; 真實 GFLOP/s 由實際執行量得。

## 數據誠實

網站預設圖表資料 (`content/data/benchmarks.json`) 明確標為示意數據, 並提供 `bench_all.py` 產生真實數據。 不在 repo 內放任何聲稱為真實量測的捏造數字。

## 驗證環境

建置環境含 ROCm (hipcc, gfx 目標), 不含 nvcc。 因此所有 kernel 以 HIP 實際 build + run 驗證; CUDA 路徑為 portability header 與 CMake 的結構性支援。 此限制在 `README.md` 與 `VERIFICATION_LOG.md` 明確標註, 符合 prompt「無法執行 GPU 程式時仍要確保結構正確並標註驗證狀態」的要求 (此處更進一步達成 HIP 的實機執行)。

## 2026-07-02 內容補完批次 (任務 1–7) 的決策

### 任務 3: On This Page anchors — 採用

`rehype-slug` 已在 MDX pipeline 中, 每個 heading 都有 id。 Sidebar 改為 client component 後, 以 `useEffect` 掃描 `article h2` 取得 anchors, 免去在 build 期為 32+ 章抽 TOC 的複雜度, 也保證與實際渲染結果一致。 成本低, 依預設加入。

### 任務 5: 平行化基礎章節位置 — Track A 之首 (`a0`)

選擇把 `a0-parallelization-and-data-races` 放在 Track A 最前面 (num 0, 顯示為 `00`), 而不是另立 Track 0。 理由: (1) 內容量是一章的量, 獨立 track 會顯得空; (2) Track A 的敘事本來就是「建立心智模型」, 平行化與 data race 是其自然前置; (3) 不動 track 顏色與 roadmap 版面。 CPU 範例放 `kernels/00-parallel-foundations/`, 編號接在 track 命名慣例之前。

### 任務 6: NCCL/RCCL 章節位置 — 新增 Track G「Multi-GPU / Collectives」

選擇新 Track G (slug `g33-nccl-rccl-collectives`) 而非併入 Track F。 理由: Track F 的主軸是單 GPU production tactics, multi-GPU collective 是不同的抽象層 (跨裝置通訊), 且未來可自然擴充 (NVSHMEM、multi-node、pipeline parallel 等)。 Track G 顏色用 `#2f81f7` (藍, 與現有六色不衝突)。

### 任務 6: NCCL/RCCL 可攜層 — 獨立 `common/ccl_portability.h`

不直接塞進 `gpu_portability.h`, 因為 NCCL/RCCL 是可選依賴 (系統可能只有 GPU toolchain 沒有 collective library); 獨立 header 讓既有 kernel 完全不受影響。 header 內以 `USE_CUDA`/`USE_HIP` 對映 `nccl.h`/`rccl/rccl.h`, API 名稱兩邊一致 (RCCL 提供 nccl 相容 API), 只需處理 include 與 error-check macro。

### 任務 7: 收合圖示範圍 — Quiz + Solution 一致化

Quiz 依需求加 ChevronDown (純 CSS `group-open:rotate-180`)。 `Solution` 既有 ▶ + `group-open:rotate-90` 已符合「圖示隨開合改變」的精神, 改為與 Quiz 相同的 chevron 樣式以求一致; `LabBox` / `FurtherReading` 不是 `<details>` (非收合元件), 不動。

### 任務 1: BenchmarkComparison 不改, 「錯 vs 對」對照用文字化輸出示例

`BenchmarkComparison` 與 GEMM benchmark JSON 深度耦合 (platform toggle、% of ref)。 為「錯 vs 對」kernel 對照另擴充會把示意數據與真實量測混在一起, 違反「數據誠實」原則。 改用: (1) MDX 內並列兩段 kernel code fence; (2) 附上 benchmark 程式的輸出格式範例 (標明 illustrative) 或直接讓讀者跑 lab; (3) 視覺化交給任務 4 的新 widgets (DataLayoutVisualizer / GatherScatterVisualizer)。

### 任務 5/6: CMake optional dependency 策略

`00-parallel-foundations` 為純 CPU target, 不經 `gklab_add_kernel`; OpenMP 版用 `find_package(OpenMP QUIET)`、pthread 版用 `find_package(Threads QUIET)`, 找不到就 `message(STATUS ...)` 後跳過該 target。 `07-multi-gpu-collectives` 在有 GPU platform 時以 `find_path`/`find_library` 找 nccl/rccl, 找不到同樣跳過。 兩者都不會讓整體 configure/build 失敗。 另外 `00-parallel-foundations` 移到 GPU platform 檢查之外, 讓無 GPU toolchain 的機器也能編 CPU 範例。

### 驗證環境限制 (本批次)

本容器無 GPU、無 nvcc/hipcc、無 nccl/rccl、無 cmake 系統套件 (以 pip 安裝 cmake 4.3.4)。 因此: CPU 範例 (00-parallel-foundations) 以 g++ 實際編譯執行驗證; GPU/NCCL target 驗證到「CMake configure 通過 + 語法審閱」為止, 於 VERIFICATION_LOG.md 標註未實測與缺少條件。

## 2026-07-03 收尾批次 (任務 8 補完 + 全面實機驗證) 的決策

### 環境升級: 本輪具備完整 GPU/RCCL toolchain

與前一批次不同, 本輪環境為 ROCm 7.2 + 64 張 AMD Instinct MI355X (gfx950) + RCCL + cmake 3.31.10 + g++。 Node/npm 不在 PATH, 以 registry.npmjs.org 的 npm tarball bootstrap 出 npm 10.9.2 搭配容器內既有的 node v22 執行網站 build。 因此本輪把前批次標為「未實測」的項目 (busbw、GPU kernel correctness) 全部補上實機執行結果, 記於 VERIFICATION_LOG.md。

### 任務 8: 補上專用 kernel `03-occupancy/tuning_levers_demo.cpp`

前批次把 Occupancy Tuning Levers 的重寫放在 `b8-occupancy-latency-hiding.mdx` (本 repo 的 occupancy 章即 b8, 非 prompt 猜測的 c13/c14), 但 LabBox 復用 `01-basics/occupancy_experiment`, 未依 prompt 建立專用 kernel。 本輪新增 `kernels/03-occupancy/tuning_levers_demo.cpp` (+ CMake, 納入上層 build), 一支程式覆蓋五個旋鈕: 用 runtime occupancy API 對 128/256/512 印理論 occupancy (Lever 1)、有無 `__launch_bounds__` 對照 (Lever 2)、scalar vs tiled 的 tile/ILP 對照 (Lever 3+4)、並計時高 occupancy vs 低 occupancy+高 ILP。 章節同步補上: 每個旋鈕的 before/after code fence、把 stall 診斷表展開成 decision-tree 文字 flowchart (每葉連回對應 Lever 的 code)、LabBox 改指 `kernels/03-occupancy`、Solution 貼上 MI355X 實跑數字。 未用 mermaid (MDX pipeline 未裝), 依 prompt 允許改用 text flowchart。

### portability header 新增 occupancy wrapper

為讓 `tuning_levers_demo` 維持「單一 .cpp 兩平台通用」, 在 `common/gpu_portability.h` 加入 `gpuOccupancyMaxActiveBlocksPerMultiprocessor` (template) 與 `gpuGetMaxThreadsPerMultiprocessor`, HIP/CUDA 兩分支對稱實作。

### 任務 5: `race_demo` 改用 `volatile` 讓 race 在 -O2 下也可觀察

前批次的 `race_demo` 用普通 `long long counter`, 在 Release (-O2) 下編譯器會把每個 thread 的迴圈收斂成一次 register 累加 + 單次 store, race window 消失 → 實測 5/5 trial 都剛好等於期望值, 但程式仍無條件印出「results are unstable」(不誠實)。 改為 `volatile long long counter` 強制每次 ++ 都走記憶體 (但不 atomic, race 仍在), 並把結論改成依實際結果 (統計 wrong_trials) 判斷。 修正後 -O2 下 5/5 trial 穩定損失 ~80% 更新, 對照 mutex/atomic 版全數正確 — 滿足硬性 gate「壞版不穩定、正確版穩定」。
