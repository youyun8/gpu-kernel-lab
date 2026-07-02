import Link from 'next/link';
import { LearningPathMap } from '@/components/LearningPathMap';
import { siteConfig } from '@/lib/site';
import { flatChapters } from '@/lib/curriculum';

export default function HomePage() {
  const totalChapters = flatChapters.length;
  return (
    <main className="mx-auto max-w-6xl px-4">
      <section className="relative overflow-hidden rounded-2xl border border-surface-border bg-gradient-to-br from-[#0d1117] via-[#111a13] to-[#0d1117] px-6 py-16 sm:px-12">
        <p className="mb-3 text-sm font-medium text-brand">CUDA · ROCm / HIP · Triton · PyTorch</p>
        <h1 className="max-w-3xl text-4xl font-bold leading-tight text-white sm:text-5xl">
          從第一個 kernel 到 library 級 GEMM, 一步步把 GPU 榨到極限
        </h1>
        <p className="mt-5 max-w-2xl text-lg text-slate-300">
          {siteConfig.name} 是一套漸進式的 kernel 優化課程: {totalChapters} 章結構化內容, 搭配可執行的 kernels、benchmark harness、profiling 腳本與互動元件, 涵蓋 NVIDIA (CUDA) 與 AMD (ROCm/HIP) 兩個平台。
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link href="/chapters/a1-what-is-a-gpu" className="rounded-lg bg-brand px-5 py-2.5 font-medium text-black transition hover:bg-brand-muted">
            開始學習 →
          </Link>
          <Link href="/roadmap" className="rounded-lg border border-surface-border px-5 py-2.5 font-medium text-slate-200 transition hover:border-brand">
            查看學習路線圖
          </Link>
          <Link href="/exercises" className="rounded-lg border border-surface-border px-5 py-2.5 font-medium text-slate-200 transition hover:border-brand">
            練習與解答
          </Link>
        </div>
      </section>

      <section className="mt-12 grid gap-4 sm:grid-cols-3">
        <InfoCard title="目標讀者" body="會寫 C/C++、對效能有好奇心的工程師與研究者。不需要 GPU 背景, 但假設你能讀基本的 C++。從 Track A 開始, 或依 profiler 找到的瓶頸直接跳章。" />
        <InfoCard title="環境需求" body={`${siteConfig.requirements}。需要 CMake ≥ 3.24、Python ≥ 3.10 (PyTorch 章節)。無 GPU 也能讀內容並做 syntax-level 編譯驗證。`} />
        <InfoCard title="數據誠實" body="網站上的預設圖表為示意數據 (illustrative)。用 scripts/bench_all.py 在你自己的硬體上跑出真實數字, 再替換 JSON。" />
      </section>

      <section className="mt-12">
        <h2 className="mb-2 text-2xl font-semibold text-white">10 分鐘快速開始</h2>
        <p className="mb-4 text-sm text-slate-400">把 repo 跑起來, 確認 toolchain 正常, 然後打開第一章。</p>
        <pre className="overflow-x-auto rounded-lg border border-surface-border bg-[#0b0f14] p-4 text-[13px] leading-6 text-slate-200">
{`# 1. clone 並啟動網站
git clone ${siteConfig.repo}.git
cd gpu-kernel-lab/website && npm install && npm run dev
# 打開 http://localhost:3000

# 2. 編譯並執行第一個 kernel (自動偵測 CUDA 或 HIP)
cd ../kernels && cmake -B build -S . && cmake --build build -j
./build/01-basics/vector_add

# 3. 產生你自己硬體的 benchmark 數據
cd .. && python scripts/bench_all.py --out website/content/data/benchmarks.json`}
        </pre>
      </section>

      <section className="mt-12 mb-16">
        <h2 className="mb-2 text-2xl font-semibold text-white">學習路線圖</h2>
        <p className="mb-6 text-sm text-slate-400">Track A → D。前三個 track 是能力階梯, Track D 貫穿各等級, 把技巧帶回 PyTorch 實戰。</p>
        <LearningPathMap />
      </section>
    </main>
  );
}

function InfoCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-surface-border bg-surface-raised/40 p-5">
      <h3 className="mb-2 text-base font-semibold text-white">{title}</h3>
      <p className="text-sm text-slate-400">{body}</p>
    </div>
  );
}
