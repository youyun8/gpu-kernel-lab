'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Menu, X } from 'lucide-react';
import { SettingsNavButton } from '@/components/SettingsModal';
import { ThemeToggle } from '@/components/ThemeToggle';

interface NavItem {
  href: string;
  label: string;
}

/**
 * Compact header navigation for small screens. The desktop header lays every
 * link out in a single row, which overflows a phone viewport, so below `md`
 * we collapse the links, GitHub and 設定 into a slide-down panel behind a
 * hamburger while keeping the theme toggle always reachable.
 */
export function MobileNav({ navItems, repo }: { navItems: NavItem[]; repo: string }) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    document.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div className="flex items-center gap-1 md:hidden">
      <ThemeToggle />
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-label={open ? '關閉選單' : '開啟選單'}
        aria-expanded={open}
        className="rounded border border-border p-2 text-muted-foreground transition hover:border-primary hover:text-foreground"
      >
        {open ? <X className="h-5 w-5" aria-hidden /> : <Menu className="h-5 w-5" aria-hidden />}
      </button>

      {open && (
        <>
          <div
            className="fixed inset-x-0 bottom-0 top-14 z-40 bg-background/60 backdrop-blur-sm"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <nav
            aria-label="主要導覽"
            className="fixed inset-x-0 top-14 z-40 border-b border-border bg-background/95 px-4 py-3 shadow-lg backdrop-blur"
          >
            <ul className="flex flex-col gap-0.5">
              {navItems.map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    onClick={() => setOpen(false)}
                    className="block rounded px-3 py-2.5 text-sm text-foreground transition hover:bg-card"
                  >
                    {item.label}
                  </Link>
                </li>
              ))}
              <li>
                <a
                  href={repo}
                  target="_blank"
                  rel="noreferrer"
                  onClick={() => setOpen(false)}
                  className="block rounded px-3 py-2.5 text-sm text-foreground transition hover:bg-card"
                >
                  GitHub
                </a>
              </li>
            </ul>
            <div className="mt-3 border-t border-border pt-3" onClick={() => setOpen(false)}>
              <SettingsNavButton />
            </div>
          </nav>
        </>
      )}
    </div>
  );
}
