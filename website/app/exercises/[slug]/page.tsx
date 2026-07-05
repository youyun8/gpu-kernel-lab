import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { kExerciseComponents } from '@/content/exercises/registry';
import { kExerciseSets, getExerciseSet } from '@/lib/curriculum';

export function generateStaticParams() {
  return kExerciseSets.map((set) => ({ slug: set.slug }));
}

export function generateMetadata({ params }: { params: { slug: string } }): Metadata {
  const set = getExerciseSet(params.slug);
  if (!set) return {};
  return { title: set.title, description: set.summary };
}

export default function ExerciseSetPage({ params }: { params: { slug: string } }) {
  const MDX = kExerciseComponents[params.slug];
  const set = getExerciseSet(params.slug);
  if (!MDX || !set) notFound();

  const index = kExerciseSets.findIndex((s) => s.slug === params.slug);
  const prev = index > 0 ? kExerciseSets[index - 1] : undefined;
  const next = index < kExerciseSets.length - 1 ? kExerciseSets[index + 1] : undefined;

  return (
    <main className="mx-auto max-w-3xl px-4 py-8">
      <p className="mb-2 text-sm font-medium" style={{ color: set.track_color }}>
        {set.track_label} · 練習
      </p>
      <article className="prose-doc">
        <MDX />
      </article>

      <nav aria-label="練習導覽" className="mt-12 grid gap-3 border-t border-border pt-6 sm:grid-cols-2">
        {prev ? (
          <Link href={`/exercises/${prev.slug}`} className="rounded-lg border border-border p-4 transition hover:border-primary">
            <span className="text-xs text-muted-foreground">← 上一組</span>
            <span className="mt-1 block text-sm font-medium text-foreground">{prev.track_label}</span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/exercises/${next.slug}`} className="rounded-lg border border-border p-4 text-right transition hover:border-primary">
            <span className="text-xs text-muted-foreground">下一組 →</span>
            <span className="mt-1 block text-sm font-medium text-foreground">{next.track_label}</span>
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </main>
  );
}
