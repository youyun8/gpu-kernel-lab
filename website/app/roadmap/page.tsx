import type { Metadata } from 'next';
import Link from 'next/link';
import { LearningPathMap } from '@/components/LearningPathMap';
import { AppWidthContainer } from '@/components/AppWidthContainer';
import { kFlatChapters, kTracks } from '@/lib/curriculum';

export const metadata: Metadata = {
  title: '學習路線圖',
  description: 'GPU Kernel Lab 的完整學習路線圖, 從平行化基礎到 production kernel。',
};

export default function RoadmapPage() {
  const first_chapter = kFlatChapters[0];
  return (
    <AppWidthContainer as="main" className="px-4 py-12">
      <h1 className="text-3xl font-bold text-foreground">學習路線圖</h1>
      <p className="mt-3 max-w-2xl text-muted-foreground">
        建議依序完成 Track 1 → 6, 建立完整的 mental model 並深入 GEMM 優化; Track 7 → 10 會把技巧延伸到 PyTorch、library 與 production kernel。 每章都附互動元件、動手實驗與章末測驗。
      </p>
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <Link
          href={`/chapters/${first_chapter.slug}`}
          className="rounded-lg bg-primary px-5 py-2.5 font-medium text-primary-foreground transition hover:bg-primary/90"
        >
          從第一章開始 →
        </Link>
        <Link
          href="/exercises"
          className="rounded-lg border border-border px-5 py-2.5 font-medium text-foreground transition hover:border-primary"
        >
          練習與解答
        </Link>
        <span className="text-sm text-muted-foreground">
          共 {kTracks.length} 個 Track、{kFlatChapters.length} 章
        </span>
      </div>
      <div className="mt-8">
        <LearningPathMap />
      </div>
    </AppWidthContainer>
  );
}
