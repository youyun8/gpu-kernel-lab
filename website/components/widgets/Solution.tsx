import type { ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';

// A native <details> element keeps the solution hidden by default while staying
// fully keyboard-accessible and functional without JavaScript (static export).
export function Solution({ children }: { children: ReactNode }) {
  return (
    <details className="group mt-3 rounded-md border border-border bg-background">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded px-4 py-2 text-sm font-medium text-primary marker:content-none hover:bg-card focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary [&::-webkit-details-marker]:hidden">
        <span>顯示解答 (Solution)</span>
        <ChevronDown
          aria-hidden
          className="ml-auto h-4 w-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180"
        />
      </summary>
      <div className="border-t border-border px-4 py-3 text-sm text-muted-foreground [&>p]:my-2">
        {children}
      </div>
    </details>
  );
}
