#!/usr/bin/env python3
"""Fix Chinese/English prose typography. Skip fenced code and inline code."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_BEFORE_OPEN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9_%\*`」』])([（(])"
)
TEXT_AFTER_CLOSE = re.compile(
    r"([）)])([\u4e00-\u9fffA-Za-z0-9_%\*`「『])"
)

# Normalize spacing inside inline code / <code> blocks (not prose typography).
CODE_PAREN_SPACING = [
    (re.compile(r"(\w)\s+\("), r"\1("),
    (re.compile(r"\(\s+"), "("),
    (re.compile(r"\s+\)"), ")"),
]


def normalize_code_parens(text: str) -> str:
    for pattern, repl in CODE_PAREN_SPACING:
        text = pattern.sub(repl, text)
    return text


def fix_prose_parens(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = TEXT_BEFORE_OPEN.sub(r"\1 \2", text)
        text = TEXT_AFTER_CLOSE.sub(r"\1 \2", text)
    return text


# Punctuation followed by text (CJK, Latin, digits, quotes, etc.).
TEXT_AFTER_PUNCT = re.compile(
    r"([,，;；!?！？。]|(?<!:)[：:])(?=[\u4e00-\u9fffA-Za-z0-9「『(（*])"
)

# URL scheme, C++ scope, known host:port patterns.
SKIP_COLON = re.compile(r":(?=/|$)|::|(?:localhost|127\.0\.0\.1):\d{2,5}")

# Thousands separator in numbers (e.g. 1,000).
SKIP_COMMA = re.compile(r"(?<=\d),(?=\d)")

URL = re.compile(r"https?://[^\s'\"<>)}]+")


def inside_url(text: str, pos: int) -> bool:
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    rel = pos - line_start
    return any(match.start() <= rel < match.end() for match in URL.finditer(line))


def fix_bold_colon_spacing(text: str) -> str:
    return re.sub(r"\*\*([^*\n]+?:)\s+\*\*", r"**\1**", text)


def fix_prose_punctuation(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        punct = match.group(1)
        pos = match.start()
        if inside_url(text, pos):
            return punct
        if punct in ",，" and SKIP_COMMA.match(text, match.start()):
            return punct
        if punct in ":：":
            snippet = text[max(0, pos - 16) : min(len(text), pos + 6)]
            if SKIP_COLON.search(snippet):
                return punct
        return f"{punct} "

    prev = None
    while prev != text:
        prev = text
        text = TEXT_AFTER_PUNCT.sub(repl, text)
    return text


def _fence_at_line_start(content: str, index: int) -> bool:
    line_start = content.rfind("\n", 0, index) + 1
    return content[line_start:index].strip() == ""


def split_fences(content: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    i = 0
    n = len(content)

    while i < n:
        if content.startswith("```", i) and _fence_at_line_start(content, i):
            end = content.find("```", i + 3)
            if end == -1:
                segments.append(("fence", content[i:]))
                break
            end += 3
            segments.append(("fence", content[i:end]))
            i = end
            continue

        next_fence = n
        pos = content.find("```", i)
        while pos != -1:
            if _fence_at_line_start(content, pos):
                next_fence = pos
                break
            pos = content.find("```", pos + 3)

        segments.append(("text", content[i:next_fence]))
        i = next_fence

    return segments


def split_inline_math(content: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    i = 0
    n = len(content)

    while i < n:
        if content.startswith("$$", i):
            end = content.find("$$", i + 2)
            if end == -1:
                segments.append(("math", content[i:]))
                break
            end += 2
            segments.append(("math", content[i:end]))
            i = end
            continue

        if content[i] == "$" and (i + 1 >= n or content[i + 1] != "$"):
            end = content.find("$", i + 1)
            if end == -1:
                segments.append(("math", content[i:]))
                break
            end += 1
            segments.append(("math", content[i:end]))
            i = end
            continue

        if content[i] == "`":
            end = content.find("`", i + 1)
            if end == -1:
                segments.append(("inline", content[i:]))
                break
            end += 1
            segments.append(("inline", content[i:end]))
            i = end
            continue

        if content.startswith("<code>", i):
            end = content.find("</code>", i + 6)
            if end == -1:
                segments.append(("inline", content[i:]))
                break
            end += len("</code>")
            segments.append(("inline", content[i:end]))
            i = end
            continue

        next_special = n
        for marker in ("$$", "`", "<code>"):
            pos = content.find(marker, i)
            if pos != -1:
                next_special = min(next_special, pos)
        if next_special == n:
            pos = content.find("$", i)
            if pos != -1:
                next_special = pos
        segments.append(("prose", content[i:next_special]))
        i = next_special

    return segments


def fix_backtick_adjacency(content: str) -> str:
    """Add prose space between inline code and adjacent punctuation."""
    content = re.sub(
        r"([\u4e00-\u9fffA-Za-z0-9_%])`([（(])([^`]+)`([）)])([\u4e00-\u9fffA-Za-z0-9_%])",
        r"\1 (`\3`) \5",
        content,
    )
    content = re.sub(
        r"`([^`]+)`([（(])",
        r"`\1` \2",
        content,
    )
    content = re.sub(
        r"([,，:：])(`)",
        r"\1 \2",
        content,
    )
    return content


def process_text_chunk(content: str) -> str:
    content = fix_backtick_adjacency(content)
    content = fix_bold_colon_spacing(content)
    out: list[str] = []
    for kind, chunk in split_inline_math(content):
        if kind == "prose":
            chunk = fix_prose_parens(chunk)
            out.append(fix_prose_punctuation(chunk))
        elif kind == "inline":
            if chunk.startswith("<code>") and chunk.endswith("</code>"):
                out.append(
                    "<code>"
                    + normalize_code_parens(chunk[6:-7])
                    + "</code>"
                )
            elif chunk.startswith("`") and chunk.endswith("`"):
                out.append("`" + normalize_code_parens(chunk[1:-1]) + "`")
            else:
                out.append(chunk)
        else:
            out.append(chunk)
    return "".join(out)


def process_content(content: str) -> str:
    out: list[str] = []
    for kind, chunk in split_fences(content):
        if kind == "fence":
            out.append(chunk)
        else:
            out.append(process_text_chunk(chunk))
    return "".join(out)


def process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = process_content(original)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> int:
    patterns = [
        "website/content/**/*.mdx",
        "website/content/**/*.md",
        "*.md",
        "kernels/**/*.md",
    ]
    changed: list[Path] = []
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if process_file(path):
                changed.append(path)

    for path in sorted(changed):
        print(path.relative_to(ROOT))
    print(f"\nUpdated {len(changed)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
