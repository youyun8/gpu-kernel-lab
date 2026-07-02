import Link from 'next/link';
import { siteConfig } from '@/lib/site';
import { ThemeToggle } from '@/components/ThemeToggle';
import { TextSizeToggle } from '@/components/TextSizeToggle';

const navItems = [
  { href: '/', label: '首頁' },
  { href: '/chapters/a1-what-is-a-gpu', label: '開始學習' },
  { href: '/roadmap', label: '學習路線圖' },
  { href: '/exercises', label: '練習與解答' },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-foreground">
          <span aria-hidden className="text-primary">▍</span>
          <span>{siteConfig.name}</span>
        </Link>
        <nav aria-label="主要導覽" className="flex items-center gap-1 text-sm">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded px-3 py-1.5 text-muted-foreground transition hover:bg-card hover:text-foreground"
            >
              {item.label}
            </Link>
          ))}
          <a
            href={siteConfig.repo}
            className="ml-1 rounded border border-border px-3 py-1.5 text-muted-foreground transition hover:border-primary hover:text-foreground"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
          <span className="mx-1 h-5 w-px bg-border" aria-hidden />
          <TextSizeToggle />
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
