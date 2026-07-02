'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { PanelLeft, PanelLeftClose } from 'lucide-react';
import { tracks } from '@/lib/curriculum';

const kSidebarWidthKey = 'chapter-sidebar-width';
const kMinWidth = 180;
const kMaxWidth = 480;
const kDefaultWidth = 256;

interface PageAnchor {
  id: string;
  label: string;
}

export function ChapterSidebar({ activeSlug }: { activeSlug: string }) {
  const [collapsed, setCollapsed] = useState(false);
  const [width, setWidth] = useState<number>(() => {
    if (typeof window === 'undefined') return kDefaultWidth;
    const saved = window.localStorage.getItem(kSidebarWidthKey);
    const parsed = saved ? parseInt(saved, 10) : NaN;
    return !Number.isNaN(parsed) && parsed >= kMinWidth && parsed <= kMaxWidth ? parsed : kDefaultWidth;
  });
  const [anchors, setAnchors] = useState<PageAnchor[]>([]);

  const dragging = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

  // Collect "On This Page" anchors from the rendered chapter body. rehype-slug
  // gives every heading an id, so scanning the DOM after mount stays in sync
  // with the actual content without a build-time TOC extraction step.
  useEffect(() => {
    const headings = document.querySelectorAll<HTMLHeadingElement>('article.prose-doc h2[id]');
    setAnchors(
      Array.from(headings).map((heading) => ({
        id: heading.id,
        label: heading.textContent ?? heading.id,
      })),
    );
  }, [activeSlug]);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current) return;
    const delta = e.clientX - startX.current;
    const next = Math.min(kMaxWidth, Math.max(kMinWidth, startWidth.current + delta));
    setWidth(next);
  }, []);

  const onMouseUp = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    setWidth((w) => {
      window.localStorage.setItem(kSidebarWidthKey, String(w));
      return w;
    });
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  const onResizeMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    startX.current = e.clientX;
    startWidth.current = width;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };

  return (
    <aside
      suppressHydrationWarning
      className={`relative hidden shrink-0 transition-[width] duration-100 lg:block ${collapsed ? 'w-10' : ''}`}
      style={collapsed ? undefined : { width }}
    >
      <button
        type="button"
        onClick={() => setCollapsed(!collapsed)}
        className="mb-2 flex w-full items-center justify-between rounded-lg border border-border bg-card/60 px-3 py-2 text-xs font-semibold text-muted-foreground transition hover:bg-accent hover:text-foreground"
        aria-label={collapsed ? '展開側欄' : '收合側欄'}
        aria-expanded={!collapsed}
      >
        {!collapsed && <span>章節目錄</span>}
        {collapsed ? <PanelLeft className="h-4 w-4" aria-hidden /> : <PanelLeftClose className="h-4 w-4" aria-hidden />}
      </button>

      {!collapsed && (
        <div className="sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto pr-2">
          <nav aria-label="章節目錄" className="space-y-5 text-sm">
            {tracks.map((track) => (
              <div key={track.id}>
                <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  <span aria-hidden className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: track.color }} />
                  {track.label}
                </p>
                <ul className="space-y-0.5">
                  {track.chapters.map((chapter) => {
                    const active = chapter.slug === activeSlug;
                    return (
                      <li key={chapter.slug}>
                        <Link
                          href={`/chapters/${chapter.slug}`}
                          aria-current={active ? 'page' : undefined}
                          className={`block rounded px-2 py-1.5 transition ${
                            active ? 'bg-primary/15 font-medium text-primary' : 'text-muted-foreground hover:bg-card hover:text-foreground'
                          }`}
                        >
                          <span className="font-mono text-xs opacity-60">{String(chapter.num).padStart(2, '0')}</span> {chapter.title}
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </nav>

          {anchors.length > 0 && (
            <div className="mt-5 border-t border-border pt-3">
              <p className="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">本頁內容</p>
              <nav aria-label="本頁內容" className="space-y-0.5">
                {anchors.map((anchor) => (
                  <a
                    key={anchor.id}
                    href={`#${anchor.id}`}
                    className="block truncate rounded px-2 py-1 text-xs text-muted-foreground transition hover:text-foreground"
                  >
                    {anchor.label}
                  </a>
                ))}
              </nav>
            </div>
          )}
        </div>
      )}

      {!collapsed && (
        <div
          onMouseDown={onResizeMouseDown}
          title="拖曳調整側欄寬度"
          aria-hidden
          className="absolute inset-y-0 right-0 z-10 flex w-2 cursor-col-resize items-center justify-center opacity-0 transition-opacity hover:opacity-100"
        >
          <div className="h-12 w-0.5 rounded-full bg-border" />
        </div>
      )}
    </aside>
  );
}
