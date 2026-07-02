import type { ReactNode } from 'react';

// A native <details> element keeps the solution hidden by default while staying
// fully keyboard-accessible and functional without JavaScript (static export).
export function Solution({ children }: { children: ReactNode }) {
  return (
    <details className="group mt-3 rounded-md border border-border bg-background">
      <summary className="cursor-pointer list-none px-4 py-2 text-sm font-medium text-primary marker:content-none hover:bg-card">
        <span aria-hidden className="mr-2 inline-block transition-transform group-open:rotate-90">
          ▶
        </span>
        顯示解答 (Solution)
      </summary>
      <div className="border-t border-border px-4 py-3 text-sm text-muted-foreground [&>p]:my-2">
        {children}
      </div>
    </details>
  );
}
