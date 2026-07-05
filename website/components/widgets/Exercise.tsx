import type { ReactNode } from 'react';

type Kind = 'paper' | 'code';
type Level = '入門' | '進階' | '挑戰';

const kKindMeta: Record<Kind, { label: string; icon: string }> = {
  paper: { label: 'paper-and-pencil', icon: '✎' },
  code: { label: 'programming', icon: '⌨' },
};

const kLevelColor: Record<Level, string> = {
  入門: '#39d353',
  進階: '#58a6ff',
  挑戰: '#f778ba',
};

export function Exercise({
  id,
  title,
  kind = 'paper',
  level = '入門',
  children,
}: {
  id: string;
  title: string;
  kind?: Kind;
  level?: Level;
  children: ReactNode;
}) {
  const meta = kKindMeta[kind];
  return (
    <section
      id={id}
      aria-label={`練習 ${id}`}
      className="my-6 rounded-lg border border-border bg-card/40 p-5 scroll-mt-24"
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm text-muted-foreground">{id}</span>
        <h3 className="text-base font-semibold text-foreground">{title}</h3>
        <span
          className="rounded-full border px-2 py-0.5 text-xs"
          style={{ borderColor: kLevelColor[level], color: kLevelColor[level] }}
        >
          {level}
        </span>
        <span className="inline-flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground">
          <span aria-hidden>{meta.icon}</span>
          {meta.label}
        </span>
      </div>
      <div className="text-sm text-muted-foreground [&>p]:my-2">{children}</div>
    </section>
  );
}
