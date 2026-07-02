import { siteConfig } from '@/lib/site';

export function SiteFooter() {
  return (
    <footer className="border-t border-border bg-card/40">
      <div className="mx-auto max-w-6xl px-4 py-8 text-sm text-muted-foreground">
        <p className="text-muted-foreground">{siteConfig.name}</p>
        <p className="mt-1">
          內容以繁體中文撰寫, technical terms 保留英文。網站上的預設 benchmark 圖表為
          <strong className="text-foreground"> 示意數據 (illustrative)</strong>, 請用 <code>scripts/bench_all.py</code>{' '}
          在你自己的硬體上產生真實數據。
        </p>
        <p className="mt-3 text-xs text-muted-foreground">環境需求: {siteConfig.requirements}。以 MIT License 釋出。</p>
      </div>
    </footer>
  );
}
