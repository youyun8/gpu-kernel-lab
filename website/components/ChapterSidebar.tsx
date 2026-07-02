import Link from 'next/link';
import { tracks } from '@/lib/curriculum';

export function ChapterSidebar({ activeSlug }: { activeSlug: string }) {
  return (
    <nav aria-label="章節目錄" className="space-y-5 text-sm">
      {tracks.map((track) => (
        <div key={track.id}>
          <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            <span aria-hidden className="h-2 w-2 rounded-full" style={{ backgroundColor: track.color }} />
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
                      active ? 'bg-brand/15 font-medium text-brand' : 'text-slate-400 hover:bg-surface-raised hover:text-white'
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
  );
}
