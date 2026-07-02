# DECISIONS

記錄開發過程中遇到的模稜兩可決策與理由 (依 prompt 要求)。

## 網站技術選型

- **@next/mdx 而非 contentlayer**: App Router + static export 下 `@next/mdx` 最直接、相依最少, 能在 MDX 內嵌 React 元件。以一個 static import registry (`content/chapters/registry.ts`) 搭配 `generateStaticParams` 匯出全部 21 章。
- **rehype-pretty-code (shiki) 做語法高亮**: 符合 prompt「shiki 或 rehype-pretty-code」的要求, dark theme 預設 (`github-dark-dimmed`), 支援 CUDA C++ / HIP / Python。
- **Recharts 做互動圖表**: 比手寫 D3 快, SSR/static export 相容。roofline 用 ComposedChart 疊 log-log 屋頂線與散點。
- **自寫 prose 樣式而非 @tailwindcss/typography**: 少一個相依, 對中英混排與 code block 的細節掌控更好。

## PlatformTabs 的 API

原本設計成 `cuda={...} hip={...}` 的 prop 形式, 但 MDX 無法在 JSX attribute 內解析 fenced code block。改為 children 形式: `<PlatformTabs><Platform name="CUDA">```...```</Platform>...</PlatformTabs>`, 讓程式碼區塊維持標準 markdown。

## kernels 可攜性策略

- **單一 `.cpp` + portability header**: `common/gpu_portability.h` 把 `gpuMalloc` / event API 對映到 CUDA 或 HIP, 由 build 定義 `USE_CUDA` / `USE_HIP`。多數 kernel (vector add、tiling、reduction、softmax) 因此一份原始碼兩平台通用。
- **CMake 用 first-class HIP language**: `check_language(HIP)` + `enable_language(HIP)`, 比手動指定 hipcc 穩健, configure 與 build 都通過。
- **Step 5 (WMMA/MFMA) 與 library 比較不進預設 build**: 這兩者高度依賴平台 arch 與 SDK, 放進跨平台預設 build 會在缺 SDK 的機器 configure/compile 失敗。改以 `kernels/03-gemm/README.md` 提供完整實作指引, 維持「clone 就能 build」的體驗。
- **benchmark peak 用 illustrative 常數**: harness 的 % of peak 以 A100 等級的示意 peak 計算, 使用者應替換成自己硬體的數字;真實 GFLOP/s 由實際執行量得。

## 數據誠實

網站預設圖表資料 (`content/data/benchmarks.json`) 明確標為示意數據, 並提供 `bench_all.py` 產生真實數據。不在 repo 內放任何聲稱為真實量測的捏造數字。

## 驗證環境

建置環境含 ROCm (hipcc, gfx 目標), 不含 nvcc。因此所有 kernel 以 HIP 實際 build + run 驗證;CUDA 路徑為 portability header 與 CMake 的結構性支援。此限制在 `README.md` 與 `VERIFICATION_LOG.md` 明確標註, 符合 prompt「無法執行 GPU 程式時仍要確保結構正確並標註驗證狀態」的要求 (此處更進一步達成 HIP 的實機執行)。
