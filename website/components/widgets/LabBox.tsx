import type { ReactNode } from 'react';
import { labUrl } from '@/lib/curriculum';

export function LabBox({ path, children }: { path: string; children: ReactNode }) {
  return (
    <section
      aria-label="動手實驗"
      className="my-8 rounded-lg border border-primary/50 bg-primary/10 p-5"
    >
      <p className="mb-2 flex items-center gap-2 text-base font-semibold text-primary">
        <span aria-hidden>🧪</span>
        <span>動手實驗</span>
      </p>
      <div className="text-sm text-muted-foreground [&>p]:my-2">{children}</div>
      <p className="mt-3 text-sm">
        對應程式碼:
        <a className="ml-1 font-mono text-primary hover:underline" href={labUrl(path)} target="_blank" rel="noreferrer">
          {path}
        </a>
      </p>
    </section>
  );
}
