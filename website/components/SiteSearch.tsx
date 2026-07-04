'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useRouter } from 'next/navigation';
import { Search, X } from 'lucide-react';
import { asset } from '@/lib/site';
import { search, splitHighlight, type SearchDoc, type SearchHit } from '@/lib/search';

function Highlighted({ text, query }: { text: string; query: string }) {
  return (
    <>
      {splitHighlight(text, query).map((part, i) =>
        part.hit ? (
          <mark key={i} className="rounded-sm bg-primary/25 text-foreground">
            {part.text}
          </mark>
        ) : (
          <span key={i}>{part.text}</span>
        ),
      )}
    </>
  );
}

export function SiteSearch() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const [index, setIndex] = useState<SearchDoc[] | null>(null);
  const [indexError, setIndexError] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultRefs = useRef<(HTMLAnchorElement | null)[]>([]);

  useEffect(() => setMounted(true), []);

  // Lazily fetch the pre-built search index the first time the dialog opens.
  useEffect(() => {
    if (!open || index || indexError) return;
    let cancelled = false;
    fetch(asset('/search-index.json'))
      .then((res) => {
        if (!res.ok) throw new Error(`search index ${res.status}`);
        return res.json();
      })
      .then((data: SearchDoc[]) => {
        if (!cancelled) setIndex(data);
      })
      .catch(() => {
        if (!cancelled) setIndexError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [open, index, indexError]);

  const hits = useMemo<SearchHit[]>(() => (index ? search(index, query) : []), [index, query]);

  useEffect(() => setActiveIndex(0), [query]);

  // Global "/" or Ctrl/Cmd+K opens the palette from anywhere on the site.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
      if ((event.key === '/' && !typing) || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k')) {
        event.preventDefault();
        setOpen(true);
      }
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const focusTimer = setTimeout(() => inputRef.current?.focus(), 0);
    return () => {
      document.body.style.overflow = previousOverflow;
      clearTimeout(focusTimer);
    };
  }, [open]);

  function close() {
    setOpen(false);
    setQuery('');
  }

  function go(url: string) {
    close();
    router.push(url);
  }

  function onInputKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Escape') {
      close();
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, hits.length - 1));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const hit = hits[activeIndex];
      if (hit) go(hit.doc.url);
    }
  }

  useEffect(() => {
    resultRefs.current[activeIndex]?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex]);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex w-full max-w-xs items-center gap-2 rounded-md border border-border bg-card/40 px-3 py-1.5 text-left text-sm text-muted-foreground transition hover:border-primary hover:text-foreground sm:max-w-sm"
      >
        <Search className="h-4 w-4 shrink-0" aria-hidden />
        <span className="hidden truncate sm:inline">搜尋章節、練習、關鍵字…</span>
        <span className="truncate sm:hidden">搜尋…</span>
        <kbd className="ml-auto hidden shrink-0 rounded border border-border bg-background px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground md:inline">
          /
        </kbd>
      </button>

      {open &&
        mounted &&
        createPortal(
          <div role="dialog" aria-modal="true" aria-label="搜尋" className="fixed inset-0 z-50 flex items-start justify-center p-4 pt-[12vh] sm:p-6 sm:pt-[14vh]">
            <div className="absolute inset-0 bg-background/60 backdrop-blur-md" onClick={close} aria-hidden />
            <div className="relative z-10 flex max-h-[70vh] w-full max-w-xl flex-col overflow-hidden rounded-2xl border border-border bg-card/95 shadow-2xl">
              <div className="flex items-center gap-2 border-b border-border px-4 py-3">
                <Search className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
                <input
                  ref={inputRef}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={onInputKeyDown}
                  type="text"
                  placeholder="搜尋章節標題、內文、練習…"
                  aria-label="搜尋內容"
                  className="w-full bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
                />
                <button
                  type="button"
                  onClick={close}
                  aria-label="關閉搜尋"
                  className="rounded p-1 text-muted-foreground transition hover:bg-background hover:text-foreground"
                >
                  <X className="h-4 w-4" aria-hidden />
                </button>
              </div>

              <div className="scrollbar-thin flex-1 overflow-y-auto p-2">
                {!query.trim() ? (
                  <p className="px-3 py-6 text-center text-sm text-muted-foreground">
                    輸入關鍵字搜尋所有章節與練習的標題、段落標題與內文。
                  </p>
                ) : !index && !indexError ? (
                  <p className="px-3 py-6 text-center text-sm text-muted-foreground">載入搜尋索引中…</p>
                ) : indexError ? (
                  <p className="px-3 py-6 text-center text-sm text-muted-foreground">搜尋索引載入失敗, 請重新整理頁面再試一次。</p>
                ) : hits.length === 0 ? (
                  <p className="px-3 py-6 text-center text-sm text-muted-foreground">
                    找不到符合「<span className="text-foreground">{query}</span>」的結果。
                  </p>
                ) : (
                  <ul className="space-y-1">
                    {hits.map((hit, i) => (
                      <li key={hit.doc.id}>
                        <a
                          ref={(el) => {
                            resultRefs.current[i] = el;
                          }}
                          href={asset(hit.doc.url)}
                          onMouseEnter={() => setActiveIndex(i)}
                          onClick={(e) => {
                            e.preventDefault();
                            go(hit.doc.url);
                          }}
                          className={`block rounded-lg px-3 py-2.5 transition ${i === activeIndex ? 'bg-primary/10' : 'hover:bg-background'}`}
                        >
                          <div className="flex items-center gap-2">
                            <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: hit.doc.trackColor }} />
                            <span className="truncate text-sm font-medium text-foreground">
                              <Highlighted text={hit.doc.title} query={query} />
                            </span>
                            <span className="ml-auto shrink-0 rounded-full border border-border px-2 py-0.5 text-[10px] text-muted-foreground">
                              {hit.doc.kind === 'chapter' ? '章節' : '練習'}
                            </span>
                          </div>
                          <p className="mt-0.5 truncate text-xs text-muted-foreground">{hit.doc.section}</p>
                          <p className="mt-1 line-clamp-2 text-xs leading-5 text-muted-foreground">
                            <Highlighted text={hit.snippet} query={query} />
                          </p>
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="hidden items-center gap-3 border-t border-border px-4 py-2 text-[11px] text-muted-foreground sm:flex">
                <span>↑↓ 選擇</span>
                <span>Enter 開啟</span>
                <span>Esc 關閉</span>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
