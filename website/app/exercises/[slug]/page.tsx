import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { exerciseComponents } from '@/content/exercises/registry';
import { exerciseSets, getExerciseSet } from '@/lib/curriculum';

export function generateStaticParams() {
  return exerciseSets.map((set) => ({ slug: set.slug }));
}

export function generateMetadata({ params }: { params: { slug: string } }): Metadata {
  const set = getExerciseSet(params.slug);
  if (!set) return {};
  return { title: set.title, description: set.summary };
}

export default function ExerciseSetPage({ params }: { params: { slug: string } }) {
  const MDX = exerciseComponents[params.slug];
  const set = getExerciseSet(params.slug);
  if (!MDX || !set) notFound();

  const index = exerciseSets.findIndex((s) => s.slug === params.slug);
  const prev = index > 0 ? exerciseSets[index - 1] : undefined;
  const next = index < exerciseSets.length - 1 ? exerciseSets[index + 1] : undefined;

  return (
    <main className="mx-auto max-w-3xl px-4 py-8">
      <p className="mb-2 text-sm font-medium" style={{ color: set.trackColor }}>
        {set.trackLabel} · 練習
      </p>
      <article className="prose-doc">
        <MDX />
      </article>

      <nav aria-label="練習導覽" className="mt-12 grid gap-3 border-t border-border pt-6 sm:grid-cols-2">
        {prev ? (
          <Link href={`/exercises/${prev.slug}`} className="rounded-lg border border-border p-4 transition hover:border-primary">
            <span className="text-xs text-muted-foreground">← 上一組</span>
            <span className="mt-1 block text-sm font-medium text-foreground">{prev.trackLabel}</span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/exercises/${next.slug}`} className="rounded-lg border border-border p-4 text-right transition hover:border-primary">
            <span className="text-xs text-muted-foreground">下一組 →</span>
            <span className="mt-1 block text-sm font-medium text-foreground">{next.trackLabel}</span>
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </main>
  );
}
