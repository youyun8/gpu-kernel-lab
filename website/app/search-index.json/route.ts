import { NextResponse } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';
import { kFlatChapters, kExerciseSets } from '@/lib/curriculum';

// Static export: this route has no request-dependent behavior, so Next emits
// it as a plain `search-index.json` file at build time (same as any other
// static asset) instead of running it per-request.
export const dynamic = 'force-static';

export interface SearchDoc {
  id: string;
  kind: 'chapter' | 'exercise';
  url: string;
  title: string;
  section: string;
  track_color: string;
  summary: string;
  headings: string[];
  text: string;
}

/** Strip MDX/JSX syntax down to roughly-readable prose for full-text search. */
function stripMdx(source: string): string {
  return source
    .replace(/```[\s\S]*?```/g, ' ') // fenced code blocks
    .replace(/`[^`]*`/g, ' ') // inline code
    .replace(/<[^>]+>/g, ' ') // JSX/HTML tags, incl. self-closing widgets like <Quiz .../>
    .replace(/\{[^{}]*\}/g, ' ') // leftover JSX expression braces
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1') // markdown links -> keep the label text
    .replace(/^#{1,6}\s*/gm, '') // heading markers
    .replace(/^\s*\|.*\|\s*$/gm, ' ') // markdown table rows
    .replace(/[*_>~`|]/g, ' ') // remaining markdown punctuation
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{2,}/g, '\n')
    .trim();
}

/** Pull out `##`/`###` headings so section titles are searchable on their own. */
function extractHeadings(source: string): string[] {
  const headings: string[] = [];
  for (const line of source.split('\n')) {
    const match = /^(#{2,3})\s+(.*)$/.exec(line.trim());
    if (match) headings.push(match[2].replace(/[`*]/g, '').trim());
  }
  return headings;
}

function readSource(dir: string, slug: string): string {
  try {
    return fs.readFileSync(path.join(dir, `${slug}.mdx`), 'utf8');
  } catch {
    return '';
  }
}

export function GET() {
  const chapters_dir = path.join(process.cwd(), 'content', 'chapters');
  const exercises_dir = path.join(process.cwd(), 'content', 'exercises');

  const docs: SearchDoc[] = [];

  for (const chapter of kFlatChapters) {
    const raw = readSource(chapters_dir, chapter.slug);
    docs.push({
      id: `chapter:${chapter.slug}`,
      kind: 'chapter',
      url: `/chapters/${chapter.slug}`,
      title: chapter.title,
      section: `${chapter.track_label} · Chapter ${chapter.num}`,
      track_color: chapter.track_color,
      summary: chapter.summary,
      headings: extractHeadings(raw).slice(0, 40),
      text: stripMdx(raw).slice(0, 6000),
    });
  }

  for (const set of kExerciseSets) {
    const raw = readSource(exercises_dir, set.slug);
    docs.push({
      id: `exercise:${set.slug}`,
      kind: 'exercise',
      url: `/exercises/${set.slug}`,
      title: set.title,
      section: set.track_label,
      track_color: set.track_color,
      summary: set.summary,
      headings: extractHeadings(raw).slice(0, 40),
      text: stripMdx(raw).slice(0, 6000),
    });
  }

  return NextResponse.json(docs);
}
