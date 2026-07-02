import type { Metadata } from 'next';
import Link from 'next/link';
import { exerciseSets } from '@/lib/curriculum';

export const metadata: Metadata = {
  title: '練習與解答',
  description: 'GPU Kernel Lab 的 paper-and-pencil 與 programming 練習題,附完整解答。',
};

export default function ExercisesIndexPage() {
  const total = exerciseSets.reduce((sum, s) => sum + s.count, 0);
  return (
    <main className="mx-auto max-w-6xl px-4 py-12">
      <h1 className="text-3xl font-bold text-white">練習與解答</h1>
      <p className="mt-3 max-w-2xl text-slate-400">
        每個 track 都有一組練習,混合 paper-and-pencil(計算與推理)與 programming(改寫 / 實作 kernel)兩類,共 {total} 題。每題附完整解答(點開 <span className="text-brand">顯示解答</span> 展開);programming 題的參考解位於 <code>kernels/exercises/</code>,可直接編譯執行。
      </p>
      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {exerciseSets.map((set) => (
          <Link
            key={set.slug}
            href={`/exercises/${set.slug}`}
            className="rounded-lg border border-surface-border bg-surface-raised/40 p-5 transition hover:border-brand"
          >
            <div className="mb-2 flex items-center gap-2">
              <span aria-hidden className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: set.trackColor }} />
              <span className="text-sm font-medium" style={{ color: set.trackColor }}>
                {set.trackLabel}
              </span>
              <span className="ml-auto rounded-full border border-surface-border px-2 py-0.5 text-xs text-slate-400">
                {set.count} 題
              </span>
            </div>
            <h2 className="text-base font-semibold text-white">{set.title}</h2>
            <p className="mt-1 text-sm text-slate-500">{set.summary}</p>
          </Link>
        ))}
      </div>
    </main>
  );
}
