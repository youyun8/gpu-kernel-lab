import Link from 'next/link';
import { siteConfig } from '@/lib/site';
import { AppWidthContainer } from '@/components/AppWidthContainer';
import { SettingsNavButton } from '@/components/SettingsModal';
import { ThemeToggle } from '@/components/ThemeToggle';
import { MobileNav } from '@/components/MobileNav';
import { SiteSearch } from '@/components/SiteSearch';

const navItems = [
  { href: '/', label: '首頁' },
  { href: '/roadmap', label: '學習路線圖' },
  { href: '/exercises', label: '練習與解答' },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
      <AppWidthContainer className="flex h-14 items-center gap-2 px-4 sm:gap-4">
        <Link href="/" className="flex shrink-0 items-center gap-2 font-semibold text-foreground">
          <span aria-hidden className="text-primary">▍</span>
          <span className="hidden sm:inline">{siteConfig.name}</span>
        </Link>
        <SiteSearch />
        <nav aria-label="主要導覽" className="ml-auto hidden items-center gap-1 text-sm md:flex">
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
          <SettingsNavButton />
          <ThemeToggle />
        </nav>
        <MobileNav navItems={navItems} repo={siteConfig.repo} />
      </AppWidthContainer>
    </header>
  );
}
