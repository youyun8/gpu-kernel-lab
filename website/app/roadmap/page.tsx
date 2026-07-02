import type { Metadata } from 'next';
import { LearningPathMap } from '@/components/LearningPathMap';
import { AppWidthContainer } from '@/components/AppWidthContainer';

export const metadata: Metadata = {
  title: '學習路線圖',
  description: 'GPU Kernel Lab 的完整 Track A → F 學習路線圖。',
};

export default function RoadmapPage() {
  return (
    <AppWidthContainer as="main" className="px-4 py-12">
      <h1 className="text-3xl font-bold text-foreground">學習路線圖</h1>
      <p className="mt-3 max-w-2xl text-muted-foreground">
        建議依序完成 Track A → B → C, 建立完整的 mental model; Track D → F 會把技巧延伸到 PyTorch、library 與 production kernel。 每章都附互動元件、動手實驗與章末測驗。
      </p>
      <div className="mt-8">
        <LearningPathMap />
      </div>
    </AppWidthContainer>
  );
}
