import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ChapterSidebar } from '@/components/ChapterSidebar';
import { chapterComponents } from '@/content/chapters/registry';
import { flatChapters, getChapterNav } from '@/lib/curriculum';

export function generateStaticParams() {
  return flatChapters.map((chapter) => ({ slug: chapter.slug }));
}

export function generateMetadata({ params }: { params: { slug: string } }): Metadata {
  const chapter = flatChapters.find((c) => c.slug === params.slug);
  if (!chapter) return {};
  return { title: chapter.title, description: chapter.summary };
}

export default function ChapterPage({ params }: { params: { slug: string } }) {
  const MDX = chapterComponents[params.slug];
  const { current, prev, next } = getChapterNav(params.slug);
  if (!MDX || !current) notFound();

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 lg:grid lg:grid-cols-[220px_1fr] lg:gap-10">
      <aside className="mb-8 hidden lg:block">
        <div className="sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto pr-2">
          <ChapterSidebar activeSlug={params.slug} />
        </div>
      </aside>

      <main className="min-w-0">
        <p className="mb-2 text-sm font-medium" style={{ color: current.trackColor }}>
          {current.trackLabel} · Chapter {current.num}
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
    </div>
  );
}
