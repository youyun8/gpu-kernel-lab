'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';
import { Monitor, Moon, Sun } from 'lucide-react';

const baseClass =
  'inline-flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-sm text-muted-foreground transition hover:border-primary hover:text-foreground';

/**
 * Cycles system → light → dark → system, mirroring cp-handbook's ThemeToggle.
 * Renders a stable placeholder until mounted to avoid hydration mismatch.
 */
export function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const { theme, setTheme } = useTheme();

  useEffect(() => setMounted(true), []);

  if (!mounted) {
    return (
      <button type="button" className={baseClass} aria-label="切換色彩模式">
        <Monitor className="h-4 w-4" aria-hidden />
      </button>
    );
  }

  const current = theme ?? 'system';
  const next = current === 'system' ? 'light' : current === 'light' ? 'dark' : 'system';
  const label = current === 'system' ? '系統' : current === 'light' ? '淺色' : '深色';
  const Icon = current === 'system' ? Monitor : current === 'light' ? Sun : Moon;

  return (
    <button
      type="button"
      className={baseClass}
      aria-label={`色彩模式：${label}（點擊切換）`}
      title={`色彩模式：${label}`}
      onClick={() => setTheme(next)}
    >
      <Icon className="h-4 w-4" aria-hidden />
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}
