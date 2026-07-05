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

export interface SearchHit {
  doc: SearchDoc;
  score: number;
  /** Which field the preview snippet was pulled from. */
  snippet_source: 'heading' | 'summary' | 'text' | null;
  snippet: string;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Build a short excerpt around the first match of `query` inside `field`. */
function excerpt(field: string, query: string, radius = 70): string {
  const idx = field.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return field.slice(0, radius * 2).trim();
  const start = Math.max(0, idx - radius);
  const end = Math.min(field.length, idx + query.length + radius);
  const prefix = start > 0 ? '…' : '';
  const suffix = end < field.length ? '…' : '';
  return `${prefix}${field.slice(start, end).trim()}${suffix}`;
}

/**
 * Score a document against a (whitespace-tokenized) query. Title / heading
 * matches count far more than body-text matches; an exact phrase match beats
 * the same words scattered across a field. Pure string matching (no
 * tokenizer/stemmer) so CJK substrings match naturally without segmentation.
 */
function scoreDoc(doc: SearchDoc, query: string, tokens: string[]): { score: number; snippet_source: SearchHit['snippet_source'] } {
  let score = 0;
  let snippet_source: SearchHit['snippet_source'] = null;

  const title = doc.title.toLowerCase();
  const headings_joined = doc.headings.join(' • ');
  const headings_lower = headings_joined.toLowerCase();
  const summary = doc.summary.toLowerCase();
  const text = doc.text.toLowerCase();

  if (title.includes(query)) {
    score += 100;
    snippet_source = snippet_source ?? null; // title itself is shown as the result title already
  }
  for (const token of tokens) {
    if (title.includes(token)) score += 30;
  }

  if (headings_lower.includes(query)) {
    score += 45;
    snippet_source = snippet_source ?? 'heading';
  }
  for (const token of tokens) {
    if (headings_lower.includes(token)) score += 12;
  }

  if (summary.includes(query)) {
    score += 25;
    snippet_source = snippet_source ?? 'summary';
  }
  for (const token of tokens) {
    if (summary.includes(token)) score += 6;
  }

  if (text.includes(query)) {
    score += 14;
    snippet_source = snippet_source ?? 'text';
  }
  for (const token of tokens) {
    if (text.includes(token)) score += 2;
  }

  return { score, snippet_source };
}

/** Rank `docs` against `query`, returning the top `limit` hits with a preview snippet. */
export function search(docs: SearchDoc[], query: string, limit = 8): SearchHit[] {
  const trimmed = query.trim().toLowerCase();
  if (!trimmed) return [];
  const tokens = trimmed.split(/\s+/).filter(Boolean);

  const hits: SearchHit[] = [];
  for (const doc of docs) {
    const { score, snippet_source } = scoreDoc(doc, trimmed, tokens);
    if (score <= 0) continue;
    const field = snippet_source === 'heading' ? doc.headings.join(' • ') : snippet_source === 'summary' ? doc.summary : snippet_source === 'text' ? doc.text : doc.summary;
    hits.push({ doc, score, snippet_source, snippet: excerpt(field, trimmed) || doc.summary });
  }

  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, limit);
}

/** Split `text` into plain/matched pieces for `<mark>`-style highlighting. */
export function splitHighlight(text: string, query: string): { text: string; hit: boolean }[] {
  const trimmed = query.trim();
  if (!trimmed) return [{ text, hit: false }];
  const re = new RegExp(`(${escapeRegExp(trimmed)})`, 'ig');
  return text.split(re).map((part) => ({ text: part, hit: part.toLowerCase() === trimmed.toLowerCase() }));
}
