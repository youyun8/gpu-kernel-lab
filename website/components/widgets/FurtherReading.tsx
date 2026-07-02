interface ReadingLink {
  label: string;
  href: string;
}

export function FurtherReading({ links }: { links: ReadingLink[] }) {
  return (
    <section aria-label="延伸閱讀" className="my-8 rounded-lg border border-surface-border bg-surface-raised/40 p-5">
      <p className="mb-2 text-base font-semibold text-white">延伸閱讀</p>
      <ul className="list-disc space-y-1 pl-6 text-sm">
        {links.map((link) => (
          <li key={link.href}>
            <a className="text-accent hover:underline" href={link.href} target="_blank" rel="noreferrer">
              {link.label}
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}
