# GPU Kernel Lab

一個以繁體中文撰寫的 **CUDA / ROCm kernel 優化學習實驗室**: 從第一個 vector add, 一路到接近 library 水準的 GEMM, 再把技巧帶回 PyTorch 實戰。 所有 technical terms (kernel、warp、shared memory、occupancy、Tensor Core、MFMA、GEMM...) 保留英文, 程式碼註解全英文。

內容分成一個 **互動學習網站** (可部署到 GitHub Pages) 與一套 **可執行的 kernels / benchmark / profiling 範例**, 兩者章節一一對應, 涵蓋 NVIDIA (CUDA) 與 AMD (ROCm/HIP) 兩個平台。

## Repo 結構

```
gpu-kernel-lab/
├── website/                 # Next.js 14 + TypeScript + Tailwind + MDX 學習網站
│   ├── app/                 # App Router 頁面 (首頁、roadmap、chapters/[slug])
│   ├── components/          # 互動元件 (7 個 widgets) 與版面
│   ├── content/chapters/    # 32 章 MDX 內容 (Track A–F)
│   ├── content/exercises/   # 5 組練習與完整解答 (Track A–E)
│   └── content/data/        # benchmark 示意數據 JSON
├── kernels/                 # 可執行 kernel, 對應網站章節
│   ├── common/              # portability header + benchmark harness
│   ├── 01-basics/           # vector add、occupancy 實驗
│   ├── 02-memory/           # coalescing、transpose、vectorized、roofline probe
│   ├── 03-gemm/             # SGEMM、quantized epilogue、autotuning examples
│   ├── 04-reductions-softmax/   # reduction、softmax、histogram atomics
│   ├── 05-advanced-scheduling/  # Split-K、CTA swizzle、async/persistent/stream pipeline
│   ├── exercises/           # programming 練習參考解 (可編譯執行)
│   └── 06-pytorch-integration/  # cpp_extension / load_inline / triton / custom_op
├── scripts/                 # bench_all.py、profile_ncu.sh、profile_rocprof.sh
├── docker/                  # CUDA 與 ROCm Dockerfile
└── .github/workflows/       # GitHub Pages 部署
```

## 環境需求

- 網站開發: Node.js ≥ 18、npm。
- kernels: CMake ≥ 3.24, 以及 **CUDA ≥ 12.x** 或 **ROCm ≥ 6.x** 其中之一。 無 GPU toolchain 時, CMake 會進入不建置 GPU target 的模式, 可改用 syntax-only 驗證 (見 `kernels/common/README.md`)。
- PyTorch 章節: Python ≥ 3.10 與對應 GPU 版本的 PyTorch; 第 20 章需要 Triton。

## 快速開始 (10 分鐘)

```bash
# 1. 啟動網站
cd website && npm install && npm run dev
# 開啟 http://localhost:3000

# 2. 編譯並執行第一個 kernel (自動偵測 CUDA / HIP)
cd ../kernels && cmake -B build -S . && cmake --build build -j
./build/01-basics/vector_add

# 3. 產生你自己硬體的 benchmark 數據
cd .. && python scripts/bench_all.py --build-dir kernels/build \
    --out website/content/data/benchmarks.local.json
```

## 本地開發網站

```bash
cd website
npm install
npm run dev        # 開發伺服器 (http://localhost:3000)
npm run typecheck  # tsc --noEmit
npm run lint       # next lint
npm run build      # static export 到 website/out/
```

## 建置與執行 kernels

```bash
cd kernels
cmake -B build -S .            # 自動偵測 CUDA 或 HIP
# 強制 HIP:  cmake -B build -S . -DGKLAB_FORCE_HIP=ON
cmake --build build -j

# 執行任一 target, 會先做 correctness check 再報告 GB/s / GFLOP/s / % of peak
./build/03-gemm/sgemm_step1_shared
./build/02-memory/transpose
```

每個 kernel 都: 用 CMake 建置 (CUDA/HIP 自動偵測)、對 CPU 或 library reference 做 correctness check、用共用 harness 報告 achieved bandwidth 與 throughput 及 % of theoretical peak。 C++ 命名遵守 PascalCase / camelCase / kCamelCase / snake_case, 註解全英文。

## 透過 GitHub Pages 部署

本專案已內建 `.github/workflows/deploy.yml`, 只要完成下列設定即可自動部署到 GitHub Pages。

### 啟用 GitHub Pages

1. 進入 repo 的 **Settings → Pages**。
2. 在 **Build and deployment → Source** 選擇 **GitHub Actions**。

### 自動部署流程

每次 push 到 `main` 分支時, CI 會依序執行:

1. `npm ci` 安裝依賴。
2. `npm run typecheck` 與 `npm run lint` 檢查程式品質。
3. `npm run build` 產生靜態 site (自動帶入 `basePath=/gpu-kernel-lab`)。
4. 透過 `actions/deploy-pages@v4` 部署到 GitHub Pages。

也可手動觸發: 到 **Actions → Deploy website to GitHub Pages → Run workflow**。

### 調整 basePath (若 fork 到不同名稱的 repo)

GitHub Pages project site 的網址格式為 `https://<user>.github.io/<repo>`。 若你的 repo 名稱不是 `gpu-kernel-lab`, 請同時修改兩處:

- `website/next.config.mjs` 中的 `basePath`。
- `.github/workflows/deploy.yml` 中的 `NEXT_PUBLIC_BASE_PATH` 環境變數。

例如 repo 名稱為 `my-gpu-lab`, 則兩處皆改為 `/my-gpu-lab`。

## 練習與解答

Track A–E 各有一組練習, 混合 paper-and-pencil (計算與推理) 與 programming (改寫 / 實作 kernel) 兩類, 共 42 題, 附完整解答:

- 網站: `/exercises` (每題可展開「顯示解答」), 內容在 `website/content/exercises/`。
- programming 參考解: `kernels/exercises/` (可用 CMake 直接編譯執行), 其餘題目複用既有 kernel 目錄。

網站上的預設 benchmark 圖表標示為 **示意數據 (illustrative)**, 不是任何真實硬體的量測結果。 用 `scripts/bench_all.py` 在你自己的 GPU 上跑, 把結果 JSON 合併進 `website/content/data/benchmarks.json` 即可換成真實數字。

## 驗證狀態

`kernels/01-basics` 至 `05-advanced-scheduling` 的 19 個 target 已在 ROCm (hipcc) 環境實際編譯並執行, 全部通過 correctness check; CUDA 路徑經 CMake 邏輯與 portability header 支援, 但本專案的建置環境未含 nvcc, 故 CUDA 為結構性支援。 PyTorch 章節以 `py_compile` 語法驗證。 詳見 `VERIFICATION_LOG.md`。

## License

MIT, 見 [LICENSE](LICENSE)。
