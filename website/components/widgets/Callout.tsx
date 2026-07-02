import type { ReactNode } from 'react';

type Variant = 'info' | 'warn' | 'insight';

const styles: Record<Variant, { border: string; label: string; icon: string }> = {
  info: { border: 'border-accent', label: '說明', icon: 'ℹ' },
  warn: { border: 'border-[#ffa657]', label: '注意', icon: '⚠' },
  insight: { border: 'border-brand', label: '直覺', icon: '★' },
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
  const style = styles[variant];
  return (
    <div className={`my-6 rounded-lg border-l-4 ${style.border} bg-surface-raised/60 p-4`}>
      <p className="mb-1 flex items-center gap-2 text-sm font-semibold text-white">
        <span aria-hidden>{style.icon}</span>
        <span>{title ?? style.label}</span>
      </p>
      <div className="text-sm text-slate-300 [&>p]:my-2">{children}</div>
    </div>
  );
}
