'use client';

import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { SettingsPanel } from '@/components/SettingsPanel';

export function SettingsNavButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        className="rounded border border-border px-3 py-1.5 text-sm text-muted-foreground transition hover:border-primary hover:text-foreground"
        onClick={() => setOpen(true)}
      >
        設定
      </button>
      <SettingsDialog open={open} onClose={() => setOpen(false)} />
    </>
  );
}

export function SettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose();
    }
    document.addEventListener('keydown', onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [open, onClose]);

  if (!open || !mounted) return null;

  return createPortal(
    <div role="dialog" aria-modal="true" aria-label="設定" className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      <div className="absolute inset-0 bg-background/60 backdrop-blur-md" onClick={onClose} aria-hidden />
      <div className="relative z-10 flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-3xl border border-border bg-card/95 shadow-2xl">
        <header className="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
          <div>
            <p className="text-sm font-semibold text-foreground">設定</p>
            <p className="mt-0.5 text-xs text-muted-foreground">調整網站外觀與閱讀偏好。這些設定會保存在目前瀏覽器。</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="關閉"
            className="rounded-full border border-border px-2.5 py-1 text-sm text-muted-foreground transition hover:bg-card hover:text-foreground"
          >
            ✕
          </button>
        </header>
        <div className="scrollbar-thin flex-1 overflow-y-auto px-5 py-4">
          <SettingsPanel />
        </div>
      </div>
    </div>,
    document.body,
  );
}
