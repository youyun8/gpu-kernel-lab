import type { ReactNode } from 'react';
import { labUrl } from '@/lib/curriculum';

export function LabBox({ path, children }: { path: string; children: ReactNode }) {
  return (
    <section
      aria-label="動手實驗"
      className="my-8 rounded-lg border border-brand-muted/50 bg-brand-muted/10 p-5"
    >
      <p className="mb-2 flex items-center gap-2 text-base font-semibold text-brand">
        <span aria-hidden>🧪</span>
        <span>動手實驗</span>
      </p>
      <div className="text-sm text-slate-300 [&>p]:my-2">{children}</div>
      <p className="mt-3 text-sm">
        對應程式碼:
        <a className="ml-1 font-mono text-accent hover:underline" href={labUrl(path)} target="_blank" rel="noreferrer">
          {path}
        </a>
      </p>
    </section>
  );
}
