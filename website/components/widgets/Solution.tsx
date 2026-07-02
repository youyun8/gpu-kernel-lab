import type { ReactNode } from 'react';

// A native <details> element keeps the solution hidden by default while staying
// fully keyboard-accessible and functional without JavaScript (static export).
export function Solution({ children }: { children: ReactNode }) {
  return (
    <details className="group mt-3 rounded-md border border-surface-border bg-surface">
      <summary className="cursor-pointer list-none px-4 py-2 text-sm font-medium text-brand marker:content-none hover:bg-surface-raised">
        <span aria-hidden className="mr-2 inline-block transition-transform group-open:rotate-90">
          ▶
        </span>
        顯示解答 (Solution)
      </summary>
      <div className="border-t border-surface-border px-4 py-3 text-sm text-slate-300 [&>p]:my-2">
        {children}
      </div>
    </details>
  );
}
