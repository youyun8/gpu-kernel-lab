'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { BookText, ChevronRight, PanelLeft, PanelLeftClose, X } from 'lucide-react';
import { tracks } from '@/lib/curriculum';

/** Which track owns a given chapter slug — used to auto-expand the active track. */
function trackIdForSlug(slug: string): string | undefined {
  return tracks.find((track) => track.chapters.some((chapter) => chapter.slug === slug))?.id;
}

const kSidebarWidthKey = 'chapter-sidebar-width';
const kMinWidth = 180;
const kMaxWidth = 480;
const kDefaultWidth = 256;

interface PageAnchor {
  id: string;
  label: string;
}

/** Shared track list + "On This Page" markup, reused by the desktop rail and
 * the mobile drawer. `onNavigate` lets the drawer close itself on selection.
 *
 * The track list is a collapsible tree: each track is a disclosure header that
 * expands to reveal its nested chapters. The track owning the active chapter is
 * expanded by default (and re-expanded whenever navigation changes it), so the
 * reader always sees where they are without hunting through a long flat list. */
function SidebarLists({
  activeSlug,
  anchors,
  onNavigate,
}: {
  activeSlug: string;
  anchors: PageAnchor[];
  onNavigate?: () => void;
}) {
  const activeTrackId = trackIdForSlug(activeSlug);
  const [expanded, setExpanded] = useState<Record<string, boolean>>(() =>
    activeTrackId ? { [activeTrackId]: true } : {},
  );

  // Keep the active track open as the reader navigates between chapters. Other
  // tracks retain whatever open/closed state the reader last chose.
  useEffect(() => {
    if (activeTrackId) setExpanded((prev) => (prev[activeTrackId] ? prev : { ...prev, [activeTrackId]: true }));
  }, [activeTrackId]);

  const toggle = (id: string) => setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));

  return (
    <>
      <nav aria-label="章節目錄" className="space-y-1 text-sm">
        {tracks.map((track) => {
          const open = expanded[track.id] ?? false;
          const hasActive = track.id === activeTrackId;
          const panelId = `track-panel-${track.id}`;
          return (
            <div key={track.id}>
              <button
                type="button"
                onClick={() => toggle(track.id)}
                aria-expanded={open}
                aria-controls={panelId}
                className={`group flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs font-semibold uppercase tracking-wide transition hover:bg-card ${
                  hasActive ? 'text-foreground' : 'text-muted-foreground'
                }`}
              >
                <ChevronRight
                  aria-hidden
                  className={`h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform duration-150 ${open ? 'rotate-90' : ''}`}
                />
                <span aria-hidden className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: track.color }} />
                <span className="min-w-0 flex-1 truncate normal-case">{track.label}</span>
                <span className="shrink-0 rounded-full border border-border px-1.5 text-[10px] font-medium tabular-nums text-muted-foreground">
                  {track.chapters.length}
                </span>
              </button>

              {open && (
                <ul
                  id={panelId}
                  className="mb-1 ml-[15px] space-y-0.5 border-l border-border pl-2"
                  style={{ borderColor: hasActive ? track.color : undefined }}
                >
                  {track.chapters.map((chapter) => {
                    const active = chapter.slug === activeSlug;
                    return (
                      <li key={chapter.slug}>
                        <Link
                          href={`/chapters/${chapter.slug}`}
                          aria-current={active ? 'page' : undefined}
                          onClick={onNavigate}
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
              )}
            </div>
          );
        })}
      </nav>

      {anchors.length > 0 && (
        <div className="mt-5 border-t border-border pt-3">
          <p className="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">本頁內容</p>
          <nav aria-label="本頁內容" className="space-y-0.5">
            {anchors.map((anchor) => (
              <a
                key={anchor.id}
                href={`#${anchor.id}`}
                onClick={onNavigate}
                className="block truncate rounded px-2 py-1 text-xs text-muted-foreground transition hover:text-foreground"
              >
                {anchor.label}
              </a>
            ))}
          </nav>
        </div>
      )}
    </>
  );
}

export function ChapterSidebar({ activeSlug }: { activeSlug: string }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
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

  // Lock body scroll and wire Escape while the mobile drawer is open.
  useEffect(() => {
    if (!mobileOpen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setMobileOpen(false);
    }
    document.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', onKey);
    };
  }, [mobileOpen]);

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
    <>
      {/* Mobile trigger — the desktop rail is hidden below lg, so phones/tablets
          reach the chapter list through this button and the drawer below. */}
      <button
        type="button"
        onClick={() => setMobileOpen(true)}
        className="mb-4 flex w-full items-center gap-2 rounded-lg border border-border bg-card/60 px-3 py-2.5 text-sm font-semibold text-muted-foreground transition hover:bg-accent hover:text-foreground lg:hidden"
        aria-label="開啟章節目錄"
      >
        <BookText className="h-4 w-4" aria-hidden />
        章節目錄
      </button>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-background/60 backdrop-blur-sm" onClick={() => setMobileOpen(false)} aria-hidden />
          <div className="absolute inset-y-0 left-0 flex w-[85%] max-w-sm flex-col border-r border-border bg-background shadow-2xl">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <span className="text-sm font-semibold text-foreground">章節目錄</span>
              <button
                type="button"
                onClick={() => setMobileOpen(false)}
                aria-label="關閉章節目錄"
                className="rounded-full border border-border p-1.5 text-muted-foreground transition hover:bg-card hover:text-foreground"
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </div>
            <div className="scrollbar-thin flex-1 overflow-y-auto px-4 py-4">
              <SidebarLists activeSlug={activeSlug} anchors={anchors} onNavigate={() => setMobileOpen(false)} />
            </div>
          </div>
        </div>
      )}

      {/* Desktop rail — the aside itself is the sticky, self-contained flex
          column so the chapter list scrolls internally. Keeping `sticky` on the
          aside (rather than an inner box) means it never runs past its own
          height and parks, which previously left a large empty gap below the
          list once the article was scrolled. */}
      <aside
        suppressHydrationWarning
        className={`sticky top-20 hidden max-h-[calc(100vh-6rem)] shrink-0 flex-col transition-[width] duration-100 lg:flex ${collapsed ? 'w-10' : ''}`}
        style={collapsed ? undefined : { width }}
      >
        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          className="mb-2 flex w-full shrink-0 items-center justify-between rounded-lg border border-border bg-card/60 px-3 py-2 text-xs font-semibold text-muted-foreground transition hover:bg-accent hover:text-foreground"
          aria-label={collapsed ? '展開側欄' : '收合側欄'}
          aria-expanded={!collapsed}
        >
          {!collapsed && <span>章節目錄</span>}
          {collapsed ? <PanelLeft className="h-4 w-4" aria-hidden /> : <PanelLeftClose className="h-4 w-4" aria-hidden />}
        </button>

        {!collapsed && (
          <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto pr-2">
            <SidebarLists activeSlug={activeSlug} anchors={anchors} />
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
    </>
  );
}
