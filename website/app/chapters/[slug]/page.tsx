import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ChapterSidebar } from '@/components/ChapterSidebar';
import { AppWidthContainer } from '@/components/AppWidthContainer';
import { kChapterComponents } from '@/content/chapters/registry';
import { kFlatChapters, getChapterNav } from '@/lib/curriculum';

export function generateStaticParams() {
  return kFlatChapters.map((chapter) => ({ slug: chapter.slug }));
}

export function generateMetadata({ params }: { params: { slug: string } }): Metadata {
  const chapter = kFlatChapters.find((c) => c.slug === params.slug);
  if (!chapter) return {};
  return { title: chapter.title, description: chapter.summary };
}

export default function ChapterPage({ params }: { params: { slug: string } }) {
  const MDX = kChapterComponents[params.slug];
  const { current, prev, next } = getChapterNav(params.slug);
  if (!MDX || !current) notFound();

  return (
    <AppWidthContainer className="px-4 py-8 lg:flex lg:items-start lg:gap-10">
      <ChapterSidebar active_slug={params.slug} />

      <main className="min-w-0 lg:flex-1">
        <p className="mb-2 text-sm font-medium" style={{ color: current.track_color }}>
          {current.track_label} · Chapter {current.num}
        </p>
        <article className="prose-doc">
          <MDX />
        </article>

        <nav aria-label="章節導覽" className="mt-12 grid gap-3 border-t border-border pt-6 sm:grid-cols-2">
          {prev ? (
            <Link href={`/chapters/${prev.slug}`} className="rounded-lg border border-border p-4 transition hover:border-primary">
              <span className="text-xs text-muted-foreground">← 上一章</span>
              <span className="mt-1 block text-sm font-medium text-foreground">{prev.title}</span>
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link href={`/chapters/${next.slug}`} className="rounded-lg border border-border p-4 text-right transition hover:border-primary">
              <span className="text-xs text-muted-foreground">下一章 →</span>
              <span className="mt-1 block text-sm font-medium text-foreground">{next.title}</span>
            </Link>
          ) : (
            <span />
          )}
        </nav>
      </main>
    </AppWidthContainer>
  );
}
