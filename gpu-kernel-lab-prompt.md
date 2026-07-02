# Claude Code 任務:gpu-kernel-lab 內容補完、連結修復、Sidebar/Quiz UI、平行化基礎與 NCCL/RCCL 專章

Repo：`https://github.com/youyun8/gpu-kernel-lab`（以 clone 到本機的工作副本為準）

---

## 專案慣例（先讀，全程遵守）

* 敘述文字用繁體中文，技術術語保留英文（coalescing、warp、data race、all-reduce…），沿用現有 MDX 風格。
* 所有 code comment 一律英文。
* C++ 命名慣例（CPU 與 GPU code 皆適用，與 repo 現況一致）：
  * 型別（class / struct）用 `PascalCase`。
  * 函式、method、以及指派給變數的 lambda 用 `camelCase`。
  * `constexpr` / `const` 變數用 `kCamelCase`（如 `kBlockSize`）。
  * 其餘變數、參數、namespace 用 `snake_case`。
  * lambda 的 capture clause 明確寫出，參數用 `snake_case`。
* Kernel 程式碼放 `kernels/`，一律 `.cpp` + `#include "gpu_portability.h"`，透過 `GPU_CHECK` / `GPU_LAUNCH` / `gpuMalloc` 等 macro 跨 CUDA/ROCm；benchmark 用 `kernels/common/benchmark.h` 的 `gklab::` 工具；每個子目錄要有自己的 `CMakeLists.txt` 並被上層 `kernels/CMakeLists.txt` 納入。
* 網站在 `website/`，Next.js 14 App Router + TypeScript + Tailwind + MDX，`next.config.mjs` 為 `output: 'export'`（GitHub Pages 靜態部署），資源路徑要走 `basePath`。
* 互動 widget 放 `website/components/widgets/`，並在 `website/mdx-components.tsx` 註冊後才能在 MDX 使用；client component 要 `'use client'`，需支援明暗主題（用現有 CSS 變數，勿寫死顏色）。
* 章節 registry 在 `website/content/chapters/registry.ts`，課綱結構在 `website/lib/curriculum.ts`。

---

## 前置門檻（動工前先做）

1. 先完整巡一遍 repo，把本任務會用到的既有慣例對齊清楚，不要重新發明既有 pattern。
2. 把「需決策」項目的選擇與理由寫進 `DECISIONS.md`，每項選一個合理預設值，之後照此自主執行，不需中途停下等我。
3. 完成後在 `VERIFICATION_LOG.md` 逐項記錄驗證結果。

---

## 任務 1：每個 CUDA/ROCm 說明都要有對應可執行程式碼

目標：讀者看到任一 CUDA/ROCm 概念時，都能同時看到「可讀的 code + 可跑的 kernel 連結」，而不是只有文字。

1. 全站稽核：掃過 `website/content/chapters/*.mdx`，找出「有講 CUDA/ROCm 概念但沒有對應 code」的段落。已知重點對象（請至少涵蓋，並自行補上其他遺漏）：
   * 完全沒有 code fence：`a1-what-is-a-gpu`、`a3-memory-hierarchy`、`a5-roofline-model`、`c15-kernel-library-ecosystem`、`c16-profiling-deep-dive`、`e25-inference-engines`。
   * code 過少（僅 1 段）：`c13`、`c14`、`e23`、`e24`、`f31`、`f32`。
2. 補 code 原則：
   * 每個新增概念要有「MDX 內的最小 code fence（說明用）」+「`kernels/` 內可編譯的完整範例」+「`LabBox` 連到該範例」三件一組。
   * 概念若牽涉「錯 vs 對」對照（例如 strided vs coalesced），給兩段 kernel 並用 `BenchmarkComparison` 呈現差異。
3. 重點展開 `b6-memory-coalescing` 的「進階存取型態」（目前 line 95–99 只有條列文字，讀者無法理解），每一項都要補真正的 kernel code 與圖/互動示意：
   * **AoSoA**：給 AoS / SoA / AoSoA 三種 layout 的 struct 定義與存取 kernel，示範 AoSoA 以 warp/wavefront 大小分塊時的 coalescing 行為；搭配任務 4 的 layout 視覺化 widget。
   * **Vectorized loads**：`float4` / `int4` 向量化存取 kernel + 尾端 scalar cleanup（可引用既有 `kernels/02-memory/vectorized_copy.cpp`，並把 tail handling 講清楚）。
   * **Cooperative loading**：一個 block 協力把不規則 global 資料搬進 shared memory 規則 tile 的 kernel，標出 `__syncthreads()` 邊界。
   * **Gather vs. scatter**：給 gather 與 scatter 兩個 kernel；scatter 版示範 atomic 前先做 warp/block aggregation。
   * **Row padding**：示範 2D tensor 的 leading dimension 補到 128-byte 倍數（pitched allocation 或手動 pad），對照補與不補的 misalignment。
   * **embedding lookup 範例**：實作 gather-based embedding lookup kernel，示範「同 batch 內 index 排序/分桶 → L2 hit rate 提升 → 輸出端用 inverse permutation 還原」的完整流程。
   * 上述 kernel 統一放 `kernels/02-memory/`（或新增 `kernels/02-memory/access-patterns/`），全部納入 CMake。

**驗收**：任務 1 涉及的章節，凡出現 CUDA/ROCm 概念處都有對應 code fence 與可連到的 kernel；`b6` 五種型態 + embedding 範例皆有可編譯 kernel。

---

## 任務 2：修正所有錯誤連結，並讓每個「對應程式碼」都指向真實可編譯的檔案

1. 修 base URL：把 `your-org` 全部改成 `youyun8`。已知兩處：
   * `website/lib/site.ts` 的 `repo` 欄位。
   * `website/lib/curriculum.ts:341` 的 `repoBase`。
   * 再全 repo `grep -rn "your-org"` 確認歸零（含 README、workflow、docker 等）。
2. 連結完整性稽核：抽出所有 `LabBox path=...` 與 `labUrl(...)` 使用的路徑，逐一確認對應目錄/檔案真實存在於 repo。缺的就補實作，不留死連結。
3. 補參考答案：全站目前 `<Solution>` 使用數為 0。針對每個 `LabBox` 指到的實驗，於 `kernels/` 內放可編譯的參考解，並在對應章節用既有 `<Solution>` widget 收合呈現關鍵片段（完整檔案仍以 repo 連結為主）。MDX 只放關鍵段落 + 連結，勿塞整份長 code。

**驗收**：全 repo 無 `your-org`；每個 `LabBox` / `labUrl` 路徑都能在 repo 找到對應檔案；每個實驗都有參考解。`VERIFICATION_LOG.md` 附上「路徑 → 是否存在」稽核表。

---

## 任務 3：Sidebar 改為可調寬、可收合（參考 cp-handbook 風格）

參考實作：`youyun8/cp-handbook` 的 `components/HandbookSidebar.tsx`。

1. 移植互動能力到 `website/components/ChapterSidebar.tsx`（改為 client component）：
   * 拖曳右緣調整寬度，範圍 `180–480px`，預設 `256px`。
   * 可收合切換鈕（用 `lucide-react` 的 `PanelLeft` / `PanelLeftClose`），收合後只留窄條。
   * 寬度以 `localStorage`（建議 key `chapter-sidebar-width`）持久化，重載後保留；SSR 安全（初始值判斷 `typeof window`）。
   * 拖曳時設定 `document.body` 的 `cursor: col-resize` 與 `userSelect: none`，放開還原。
   * drag handle 平時透明、hover 顯示。
2. 保留現有導覽語意：track 分組、目前章節高亮、章節編號補零顯示等現有行為不可退化。
3. 調整版面：`website/app/chapters/[slug]/page.tsx` 目前用寫死的 `lg:grid-cols-[220px_1fr]`，改成能吃 sidebar 動態寬度的排版（sidebar 自帶寬度、主內容 `flex-1 min-w-0`），確保調寬時內容區跟著縮放且不 overflow。
4. 需決策：是否一併加入 cp-handbook 的「本頁內容（On This Page）anchors」區塊。預設加入，若成本過高則記錄於 DECISIONS.md 並略過。
5. 手機/窄螢幕維持現有行為（隱藏或抽屜），不要因桌機改動破壞 RWD。

**驗收**：桌機可拖曳調寬且重載保留、可收合；窄螢幕不破版；`next build` 通過。

---

## 任務 4：用圖或互動 widget 讓概念更易懂

沿用既有 widget pattern（`MemoryCoalescingVisualizer`、`TilingAnimator`、`OccupancyCalculator`、`RooflineChart` 等）。新增下列 widget，註冊於 `mdx-components.tsx`，並在對應章節嵌入：

1. **Data layout 視覺化**（給任務 1 的 AoSoA）：切換 AoS / SoA / AoSoA，動畫呈現一個 warp 在同一 cycle 觸及的記憶體位址，標示是否 coalesced。
2. **Gather / Scatter 視覺化**：並列 gather（讀 coalesced、寫連續）與 scatter（寫發散、含 atomic aggregation）。
3. **Collective 動畫**（給任務 6-NCCL）：以 ring 拓樸逐步動畫呈現 all-reduce / all-gather / reduce-scatter 的資料流與每步傳輸量，可切換 GPU 數量；有餘力再加 ring vs tree 對照。
4. **Data race 時序視覺化**（給任務 5）：以兩條（或多條）thread 的時間軸，動畫呈現「交錯的 read-modify-write 造成 lost update」，並示範加上 lock / atomic 後的正確時序；可切換「有無同步」。
5. 通用要求：
   * 全部 `'use client'`、鍵盤可操作、`aria-label` 完整、支援明暗主題（用現有 CSS 變數）。
   * 效能與風格對齊既有 widget；勿引入與現有堆疊衝突的重量級依賴。
   * 每個 widget 在其章節都要有一段繁中說明搭配，不能只丟圖。
6. 對「概念適合用圖但目前純文字」的既有章節，酌情復用上述或既有 widget 補強，不需為每章硬造新元件。

**驗收**：四個新 widget 可正常渲染、可互動、主題正確；已在對應章節嵌入。

---

## 任務 5：新增「平行化設計與 data race」章節（C++/OpenMP/Pthread ↔ CUDA/ROCm 對照）

目標：讓讀者理解如何設計平行化，以及平行化時會踩到的問題（尤其 data race），並用 CPU（C++ std::thread / OpenMP / Pthread）與 GPU（CUDA/ROCm）同一組概念雙邊對照。每個概念都要給可執行 code。

1. 內容範圍（繁中敘述 + 英文術語）：
   * 如何設計平行化：decomposition（data parallelism vs task parallelism）、granularity 與 load balancing、把工作 map 到硬體（CPU thread/core vs GPU block/warp/thread）、為什麼 GPU 偏好 massive data parallelism。
   * data race 是什麼、如何發生：同一記憶體位址的並行未同步 read/modify/write；用 lost update 的具體例子講清楚。
   * 同步與正確性（雙邊對照，每項都給 CPU + GPU code）：
     * Mutual exclusion：C++ `std::mutex` / `std::lock_guard`、Pthread `pthread_mutex_*`、OpenMP `critical` ↔ GPU 為何很少用 lock、改用 atomic 或重新設計避免競爭。
     * Atomics：C++ `std::atomic` 與 `std::memory_order` ↔ CUDA/ROCm `atomicAdd` 等 device atomic 與 atomic scope。
     * Reduction：OpenMP `reduction(+:...)` ↔ GPU warp/block reduction（可連結既有 `04-reductions-softmax`）。
     * Barrier / 同步點：Pthread barrier、OpenMP `barrier` ↔ CUDA/ROCm `__syncthreads()`、cooperative groups、memory fence（`__threadfence`）。
   * 常見陷阱：race condition、deadlock、false sharing（CPU cache line）、非決定性結果、以及為何浮點 reduction 順序會影響數值。
   * 偵錯工具：CPU 用 ThreadSanitizer（`-fsanitize=thread`）；GPU 用 `compute-sanitizer --tool racecheck`（ROCm 對應工具亦一併提及）。
2. 可執行 code：新增 `kernels/00-parallel-foundations/`：
   * CPU 範例：`race_demo`（故意有 race，示範 lost update）、`race_fixed_mutex`、`race_fixed_atomic`、OpenMP `reduction` 版、Pthread 版；需要 OpenMP/pthread 的 target 在 CMake 用 `find_package(OpenMP)` / `Threads`，找不到就 skip 該 target（勿讓整體 build 失敗）。
   * GPU 範例：對應的 `atomic_histogram`、`block_reduce`、`threadfence` 示範，沿用 `gpu_portability.h`。
   * 每個「壞版 vs 正確版」都要能實際跑出「壞版結果不穩定 / 正確版穩定」的對照輸出。
3. 章節與課綱（需決策）：此為基礎主題，預設新增一章置於 Track A 早段（建議 slug `a0-parallelization-and-data-races`，並確保排序/prev-next 正確）；若你判斷放獨立的「Track 0 平行化基礎」更順，擇一並記錄於 DECISIONS.md。需同步更新 `registry.ts` 與 `curriculum.ts`。
4. 章內嵌入任務 4 的 Data race 時序視覺化 widget，以及至少一個 `Quiz` 章末測驗。

**驗收**：新章可在導覽出現；CPU/GPU 雙邊 code 皆存在且 CMake 在有/無 OpenMP/pthread/GPU toolchain 時都不讓整體 build 掛掉；壞版/正確版對照可實際觀察到差異。

---

## 任務 6：新增 NCCL / RCCL 專章（含實測 bus bandwidth 門檻）

目前 NCCL/RCCL 只在 `a1` / `d21` / `f28` 被順帶提到，無專章、無 code。

1. 內容範圍（繁中敘述 + 英文術語）：
   * 為何需要 collective：data/tensor/pipeline parallel 對通訊的需求。
   * 核心 primitive：all-reduce、all-gather、reduce-scatter、broadcast、all-to-all；點出 `all-reduce == reduce-scatter + all-gather`。
   * 演算法：ring vs tree，ring 為何 bandwidth-optimal、tree 為何 latency 較低；bus bandwidth 與 message size 的關係。
   * 拓樸與互連：NVLink/NVSwitch 對 xGMI/Infinity Fabric，intra-node vs inter-node。
   * 與 compute overlap：communication stream、用 event 建立 compute↔comm 依賴（呼應 f28）。
   * CUDA↔ROCm 對照：NCCL 與 RCCL API 相容性，以及編譯/連結差異。
2. 章節與課綱（需決策）：預設新增 Track G「Multi-GPU / Collectives」，slug `g33-nccl-rccl-collectives`；若併入 Track F 更順，改 `f33-...` 亦可，擇一並記錄於 DECISIONS.md。需同步更新 `registry.ts` 與 `curriculum.ts`（track 定義、顏色、編號、prev/next）。
3. 可執行 code：新增 `kernels/07-multi-gpu-collectives/`：
   * single-process、multi-device 的 ring all-reduce 範例（用 `ncclCommInitAll` 在單機多卡上跑），含正確性檢查。
   * 量測並輸出 bus bandwidth：遵循 nccl-tests 慣例，計算並印出 `algbw`（algorithm bandwidth）與 `busbw`（bus bandwidth），並在文件中寫出 busbw 的推導（all-reduce 的 `busbw = algbw * 2*(n-1)/n`）。輸出格式要清楚標示 size、time、algbw(GB/s)、busbw(GB/s)。
   * 擴充 `kernels/common/gpu_portability.h`，加入 NCCL/RCCL 可攜層（RCCL 提供 nccl 相容 API；處理標頭與 lib 命名差異），讓同一份 `.cpp` 在 CUDA 與 ROCm 皆可編。
   * `CMakeLists.txt` 用 `find_package` / `find_library` 找 nccl 或 rccl，找不到時 gracefully skip（勿讓整體 build 失敗）。
   * 用 `LabBox` 連到此目錄，並嵌入任務 4 的 Collective 動畫 widget。
4. 在 `a1` / `d21` / `f28` 既有提及處，補交叉連結指向本新章。

**驗收（硬性）**：

* 新章可在導覽出現。
* kernel 目錄存在；CMake 在有/無 nccl 環境都不讓整體 build 掛掉。
* 在多卡機實際執行 all-reduce 範例，`VERIFICATION_LOG.md` 必須貼上真實一輪的輸出，包含各 message size 對應的 algbw 與 busbw（GB/s）數字；若當前環境無多卡/無 nccl toolchain 無法執行，必須明確標註「未實測」與缺少的條件，不可用假數字充數。

---

## 任務 7：章末測驗加入可反映開合狀態的圖示

現況：`website/components/widgets/Quiz.tsx` 用原生 `<details>/<summary>`，預設收合，但只有原生 marker，沒有明確、會隨開合變化的圖示。

1. 在 `<summary>` 加入明確的收合圖示（例如倒三角 ▼ 或 `lucide-react` 的 `ChevronDown`），讓讀者一眼看出「預設是收合、可展開」。
2. 圖示要隨開合狀態改變樣子：收合時指向右/上（如 ▶），展開時指向下（如 ▼）。優先用純 CSS 依 `<details open>` 狀態旋轉（例如收合預設、`open:` 時 `rotate-180`），不需額外 JS，維持 SSR 安全與可存取性。
3. 隱藏原生 disclosure marker（避免和自訂圖示重複），但保留鍵盤可操作與 focus 樣式。
4. 保持預設收合（不要加 `open` 屬性）。
5. 需決策：是否讓 `Solution`、`LabBox`、`FurtherReading` 等同樣用 `<details>` 的 widget 一併套用一致的圖示規則。預設只改 Quiz（符合這次需求），若順手且風格一致再擴及其他，並記錄於 DECISIONS.md。

**驗收**：章末測驗預設收合且顯示收合圖示；展開後圖示改變；明暗主題與鍵盤操作正常。

---

## 任務 8：全面改寫 Occupancy 章節的 Tuning Levers 段落

現況：Occupancy 相關章節（`c13` 或 `c14`）中的 **Tuning Levers** 小節目前純條列，缺乏具體範例與脈絡，讀者難以理解各調參旋鈕背後的真實意義與取捨邏輯。問題如下：

> 目前文字（示例）：
> - Block size：128、256、512 threads 是常見候選。太小會浪費 scheduler/block slots，太大會降低 block-level scheduling 彈性。
> - Register cap：`__launch_bounds__` 或 compiler flag 可壓 register，但一旦出現 spill 就要退回。
> - Tile size：大 tile 提高 reuse 和 ILP，但吃 register/shared memory。小 tile occupancy 高，但可能 memory traffic 變多。
> - Independent accumulators：對 latency-bound loop，多個 accumulator 可用 ILP 替代部分 occupancy。
> - Grid size：achieved occupancy 低可能只是 blocks 太少。小矩陣用 Split-K、batching 或 persistent scheduler 比調 register 有效。
> - Practical loop：先用 profiler 看 stall reason。若是 long scoreboard，增加 active warps 或每 warp independent loads；若是 not selected 很高，warp 已經足夠但 issue pipe 競爭，該看 instruction mix；若是 math pipe throttle，occupancy 不是瓶頸，要調算術 pipeline 或使用 MMA。

請**全面改寫**此段落，要求：

1. **每個調參旋鈕都要有最小可執行範例**，以 code fence 展示「改動前 vs 改動後」，讓讀者看出如何實際調整（例如：如何改 block size、如何加 `__launch_bounds__`、如何展開 accumulator）。
2. **加入數字化說明**：用 occupancy calculator 的角度說明，例如給定 SM 的 register file 上限與 shared memory 大小，示範「register 用量如何限制 resident warps 數」，可搭配現有 `OccupancyCalculator` widget 或補充一個互動試算小例。
3. **stall reason → 對策 的決策樹**：把「practical loop」展開成一張清晰的 flowchart 文字版（或 mermaid diagram），讓讀者能依 profiler 輸出找到對應的行動項目，每個節點都連結對應 code 範例。
4. **改寫原則**：
   * 繁中敘述 + 英文術語（與專案慣例一致）。
   * 深度不降：仍涵蓋 block size / register cap / tile size / independent accumulators / grid size / stall reason 六條主軸。
   * 新增「何時 occupancy 不是瓶頸」的判斷說明，避免讀者盲目追高 occupancy。
   * 對應 `kernels/` 內補上可編譯的示範 kernel（如 `kernels/03-occupancy/tuning_levers_demo.cpp`），納入 CMake，並用 `LabBox` 連結。

**驗收**：改寫後的 Tuning Levers 段落每個旋鈕都有 code fence；stall reason 決策路徑清楚可循；`OccupancyCalculator`（或新 widget）已嵌入；對應 kernel 可編譯。

---

## 全域驗證門檻（每項都要在 VERIFICATION_LOG.md 記錄）

1. `cd website && npm ci && npm run build` 成功產生靜態輸出（`output: export` 不可壞，basePath 不可壞）。
2. `grep -rn "your-org" .` 結果為空。
3. 全部 `LabBox` / `labUrl` 路徑稽核表（路徑 → 存在與否），無死連結。
4. 新增/修改的 kernel：至少通過 CMake configure；無對應 toolchain 時記錄「已加入 target、待有 nvcc/hipcc/nccl/OpenMP 時可編」並確認語法無明顯錯誤。
5. NCCL/RCCL bus bandwidth 門檻（硬性）：多卡機實測輸出（algbw + busbw）已貼進 log；無法實測則明確標註未實測與缺少條件。
6. 任務 5 的壞版/正確版對照：實際跑出「壞版結果不穩定、正確版穩定」的輸出並記錄。
7. Sidebar 手動驗收：拖曳調寬→重載保留、收合/展開正常、窄螢幕不破版（附簡述或截圖）。
8. Quiz 手動驗收：預設收合、圖示隨開合改變、鍵盤可操作、明暗主題正確。
9. 四個新 widget 手動驗收：可渲染、可互動、明暗主題皆正確。
10. Occupancy Tuning Levers 手動驗收：每個旋鈕有 code fence、stall reason 決策路徑清楚、`OccupancyCalculator` widget 已嵌入。
11. `DECISIONS.md` 已記錄所有「需決策」項目的選擇與理由；`VERIFICATION_LOG.md` 已更新。

---

## Commit 策略

* 依任務切分成數個語意明確的 commit（例如：`fix: repo links your-org → youyun8`、`feat: resizable collapsible chapter sidebar`、`feat: parallelization and data-race chapter with CPU/GPU examples`、`feat: NCCL/RCCL collectives chapter with busbw benchmark`、`feat: quiz collapse indicator`、`docs: add code examples for access patterns`、`docs: rewrite occupancy tuning-levers with examples and decision flowchart`…）。
* 每個 commit 訊息用英文、動詞開頭、說明「為什麼」。
* 全程勿破壞既有 static export 與 GitHub Actions 部署流程。
