import type { ReactNode } from 'react';

type Variant = 'info' | 'warn' | 'insight';

const kStyles: Record<Variant, { border: string; label: string; icon: string }> = {
  info: { border: 'border-accent', label: '說明', icon: 'ℹ' },
  warn: { border: 'border-[#ffa657]', label: '注意', icon: '⚠' },
  insight: { border: 'border-primary', label: '直覺', icon: '★' },
};

export function Callout({
  variant = 'info',
  title,
  children,
}: {
  variant?: Variant;
  title?: string;
  children: ReactNode;
}) {
  const style = kStyles[variant];
  return (
    <div className={`my-6 rounded-lg border-l-4 ${style.border} bg-card/60 p-4`}>
      <p className="mb-1 flex items-center gap-2 text-sm font-semibold text-foreground">
        <span aria-hidden>{style.icon}</span>
        <span>{title ?? style.label}</span>
      </p>
      <div className="text-sm text-muted-foreground [&>p]:my-2">{children}</div>
    </div>
  );
}
