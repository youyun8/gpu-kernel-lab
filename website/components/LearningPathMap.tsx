import Link from 'next/link';
import { tracks } from '@/lib/curriculum';

export function LearningPathMap() {
  return (
    <div className="space-y-6">
      {tracks.map((track) => (
        <div key={track.id} className="rounded-lg border border-surface-border bg-surface-raised/40 p-5">
          <div className="mb-3 flex items-center gap-3">
            <span aria-hidden className="h-3 w-3 rounded-full" style={{ backgroundColor: track.color }} />
            <h3 className="text-lg font-semibold text-white">{track.label}</h3>
            <span className="rounded-full border border-surface-border px-2 py-0.5 text-xs text-slate-400">{track.level}</span>
          </div>
          <p className="mb-4 text-sm text-slate-400">{track.description}</p>
          <ol className="grid gap-2 sm:grid-cols-2">
            {track.chapters.map((chapter) => (
              <li key={chapter.slug} className="list-none">
                <Link
                  href={`/chapters/${chapter.slug}`}
                  className="flex h-full items-start gap-3 rounded-md border border-surface-border bg-surface p-3 transition hover:border-brand"
                >
                  <span className="mt-0.5 font-mono text-sm" style={{ color: track.color }}>
                    {String(chapter.num).padStart(2, '0')}
                  </span>
                  <span>
                    <span className="block text-sm font-medium text-slate-100">{chapter.title}</span>
                    <span className="mt-0.5 block text-xs text-slate-500">{chapter.summary}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ol>
        </div>
      ))}
    </div>
  );
}
