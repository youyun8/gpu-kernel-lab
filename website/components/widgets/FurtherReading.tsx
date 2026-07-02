interface ReadingLink {
  // IEEE-style citation text, e.g. author/organization, "Title," source, year.
  citation: string;
  href: string;
}

export function FurtherReading({ links }: { links: ReadingLink[] }) {
  return (
    <section aria-label="延伸閱讀 (References)" className="my-8 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-3 text-base font-semibold text-foreground">延伸閱讀 (References)</p>
      <ol className="space-y-2 text-sm">
        {links.map((link, index) => (
          <li key={link.href} className="grid grid-cols-[2.25rem_1fr] gap-1">
            <span className="font-mono text-muted-foreground">[{index + 1}]</span>
            <span className="text-muted-foreground">
              {link.citation} [Online]. Available:{' '}
              <a className="break-all text-primary hover:underline" href={link.href} target="_blank" rel="noreferrer">
                {link.href}
              </a>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
