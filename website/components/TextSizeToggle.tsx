'use client';

import { useEffect, useState } from 'react';
import { ALargeSmall } from 'lucide-react';

type TextSize = 'small' | 'normal' | 'large';

const order: TextSize[] = ['small', 'normal', 'large'];
const labels: Record<TextSize, string> = { small: '小', normal: '中', large: '大' };
const STORAGE_KEY = 'gklab-text-size';

const baseClass =
  'inline-flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-sm text-muted-foreground transition hover:border-primary hover:text-foreground';

function apply(size: TextSize) {
  const root = document.documentElement;
  if (size === 'normal') {
    delete root.dataset.textSize;
  } else {
    root.dataset.textSize = size;
  }
}

/**
 * Cycles the document text size via the `data-text-size` hooks defined in
 * globals.css, matching cp-handbook's text-size setting. Persisted to
 * localStorage; a stable placeholder is rendered until mounted.
 */
export function TextSizeToggle() {
  const [mounted, setMounted] = useState(false);
  const [size, setSize] = useState<TextSize>('normal');

  useEffect(() => {
    const stored = (localStorage.getItem(STORAGE_KEY) as TextSize | null) ?? 'normal';
    setSize(stored);
    apply(stored);
    setMounted(true);
  }, []);

  function cycle() {
    const nextSize = order[(order.indexOf(size) + 1) % order.length];
    setSize(nextSize);
    apply(nextSize);
    localStorage.setItem(STORAGE_KEY, nextSize);
  }

  return (
    <button
      type="button"
      className={baseClass}
      aria-label={`文字大小：${labels[size]}（點擊切換）`}
      title={`文字大小：${labels[size]}`}
      onClick={mounted ? cycle : undefined}
    >
      <ALargeSmall className="h-4 w-4" aria-hidden />
      <span className="hidden sm:inline">{mounted ? labels[size] : '中'}</span>
    </button>
  );
}
