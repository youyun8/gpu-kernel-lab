'use client';

import { Children, isValidElement, useState, type ReactElement, type ReactNode } from 'react';

/**
 * Marker component for one platform panel. Rendering is handled by the parent
 * PlatformTabs; this just carries a `name` label and its markdown children.
 */
export function Platform({ children }: { name: string; children: ReactNode }) {
  return <>{children}</>;
}

interface PanelData {
  name: string;
  content: ReactNode;
}

export function PlatformTabs({ children }: { children: ReactNode }) {
  const panels: PanelData[] = Children.toArray(children)
    .filter((child): child is ReactElement<{ name?: string; children?: ReactNode }> => isValidElement(child))
    .map((child) => ({ name: child.props.name ?? 'Code', content: child.props.children }));

  const [active, setActive] = useState(0);
  if (panels.length === 0) return null;
  const current = panels[Math.min(active, panels.length - 1)];

  return (
    <div className="my-6 overflow-hidden rounded-lg border border-surface-border">
      <div role="tablist" aria-label="平台程式碼切換" className="flex border-b border-surface-border bg-surface-raised">
        {panels.map((panel, i) => (
          <button
            key={panel.name}
            role="tab"
            aria-selected={active === i}
            id={`tab-${panel.name}`}
            className={`px-4 py-2 text-sm font-medium transition ${
              active === i ? 'bg-surface text-brand' : 'text-slate-400 hover:text-white'
            }`}
            onClick={() => setActive(i)}
          >
            {panel.name}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        aria-labelledby={`tab-${current.name}`}
        className="bg-surface px-4 [&_pre]:my-0 [&_pre]:rounded-none [&_pre]:border-0"
      >
        {current.content}
      </div>
    </div>
  );
}
