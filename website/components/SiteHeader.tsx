import Link from 'next/link';
import { siteConfig } from '@/lib/site';

const navItems = [
  { href: '/', label: '首頁' },
  { href: '/chapters/a1-what-is-a-gpu', label: '開始學習' },
  { href: '/roadmap', label: '學習路線圖' },
  { href: '/exercises', label: '練習與解答' },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-surface-border bg-surface/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-white">
          <span aria-hidden className="text-brand">▍</span>
          <span>{siteConfig.name}</span>
        </Link>
        <nav aria-label="主要導覽" className="flex items-center gap-1 text-sm">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded px-3 py-1.5 text-slate-300 transition hover:bg-surface-raised hover:text-white"
            >
              {item.label}
            </Link>
          ))}
          <a
            href={siteConfig.repo}
            className="ml-1 rounded border border-surface-border px-3 py-1.5 text-slate-300 transition hover:border-brand hover:text-white"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
