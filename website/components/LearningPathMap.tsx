import Link from 'next/link';
import { kTracks } from '@/lib/curriculum';

export function LearningPathMap() {
  return (
    <div className="space-y-6">
      {kTracks.map((track) => (
        <div key={track.id} className="rounded-lg border border-border bg-card/40 p-5">
          <div className="mb-3 flex items-center gap-3">
            <span aria-hidden className="h-3 w-3 rounded-full" style={{ backgroundColor: track.color }} />
            <h3 className="text-lg font-semibold text-foreground">{track.label}</h3>
            <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground">{track.level}</span>
          </div>
          <p className="mb-4 text-sm text-muted-foreground">{track.description}</p>
          <ol className="grid gap-2 sm:grid-cols-2">
            {track.chapters.map((chapter) => (
              <li key={chapter.slug} className="list-none">
                <Link
                  href={`/chapters/${chapter.slug}`}
                  className="flex h-full items-start gap-3 rounded-md border border-border bg-background p-3 transition hover:border-primary"
                >
                  <span className="mt-0.5 font-mono text-sm" style={{ color: track.color }}>
                    {String(chapter.num).padStart(2, '0')}
                  </span>
                  <span>
                    <span className="block text-sm font-medium text-foreground">{chapter.title}</span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">{chapter.summary}</span>
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
