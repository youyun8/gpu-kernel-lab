import { siteConfig } from '@/lib/site';

export function SiteFooter() {
  return (
    <footer className="border-t border-surface-border bg-surface-raised/40">
      <div className="mx-auto max-w-6xl px-4 py-8 text-sm text-slate-400">
        <p className="text-slate-300">{siteConfig.name}</p>
        <p className="mt-1">
          內容以繁體中文撰寫,technical terms 保留英文。網站上的預設 benchmark 圖表為
          <strong className="text-slate-200"> 示意數據 (illustrative)</strong>,請用 <code>scripts/bench_all.py</code>{' '}
          在你自己的硬體上產生真實數據。
        </p>
        <p className="mt-3 text-xs text-slate-500">環境需求:{siteConfig.requirements}。以 MIT License 釋出。</p>
      </div>
    </footer>
  );
}
