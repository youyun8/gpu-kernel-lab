#!/usr/bin/env python3
"""Fix Chinese/English prose typography. Skip fenced code and inline code."""

from __future__ import annotations

import re
import sys
from pathlib import Path

kRoot = Path(__file__).resolve().parents[1]

kTextBeforeOpen = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9_%\*`」』])([（(])"
)
kTextAfterClose = re.compile(
    r"([）)])([\u4e00-\u9fffA-Za-z0-9_%\*`「『])"
)

# Normalize spacing inside inline code / <code> blocks (not prose typography).
kCodeParenSpacing = [
    (re.compile(r"(\w)\s+\("), r"\1("),
    (re.compile(r"\(\s+"), "("),
    (re.compile(r"\s+\)"), ")"),
]


def normalizeCodeParens(text: str) -> str:
    for pattern, repl in kCodeParenSpacing:
        text = pattern.sub(repl, text)
    return text


def fixProseParens(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = kTextBeforeOpen.sub(r"\1 \2", text)
        text = kTextAfterClose.sub(r"\1 \2", text)
    return text


# Punctuation followed by text (CJK, Latin, digits, quotes, etc.).
kTextAfterPunct = re.compile(
    r"([,，;；!?！？。]|(?<!:)[：:])(?=[\u4e00-\u9fffA-Za-z0-9「『(（*])"
)

# URL scheme, C++ scope, known host:port patterns.
kSkipColon = re.compile(r":(?=/|$)|::|(?:localhost|127\.0\.0\.1):\d{2,5}")

# Thousands separator in numbers (e.g. 1,000).
kSkipComma = re.compile(r"(?<=\d),(?=\d)")

kUrl = re.compile(r"https?://[^\s'\"<>)}]+")


def insideUrl(text: str, pos: int) -> bool:
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    rel = pos - line_start
    return any(match.start() <= rel < match.end() for match in kUrl.finditer(line))


def fixBoldColonSpacing(text: str) -> str:
    return re.sub(r"\*\*([^*\n]+?:)\s+\*\*", r"**\1**", text)


def fixProsePunctuation(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        punct = match.group(1)
        pos = match.start()
        if insideUrl(text, pos):
            return punct
        if punct in ",，" and kSkipComma.match(text, match.start()):
            return punct
        if punct in ":：":
            snippet = text[max(0, pos - 16) : min(len(text), pos + 6)]
            if kSkipColon.search(snippet):
                return punct
        return f"{punct} "

    prev = None
    while prev != text:
        prev = text
        text = kTextAfterPunct.sub(repl, text)
    return text


def fenceAtLineStart(content: str, index: int) -> bool:
    line_start = content.rfind("\n", 0, index) + 1
    return content[line_start:index].strip() == ""


def splitFences(content: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    i = 0
    n = len(content)

    while i < n:
        if content.startswith("```", i) and fenceAtLineStart(content, i):
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
            if fenceAtLineStart(content, pos):
                next_fence = pos
                break
            pos = content.find("```", pos + 3)

        segments.append(("text", content[i:next_fence]))
        i = next_fence

    return segments


def splitInlineMath(content: str) -> list[tuple[str, str]]:
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


def fixBacktickAdjacency(content: str) -> str:
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


def processTextChunk(content: str) -> str:
    content = fixBacktickAdjacency(content)
    content = fixBoldColonSpacing(content)
    out: list[str] = []
    for kind, chunk in splitInlineMath(content):
        if kind == "prose":
            chunk = fixProseParens(chunk)
            out.append(fixProsePunctuation(chunk))
        elif kind == "inline":
            if chunk.startswith("<code>") and chunk.endswith("</code>"):
                out.append(
                    "<code>"
                    + normalizeCodeParens(chunk[6:-7])
                    + "</code>"
                )
            elif chunk.startswith("`") and chunk.endswith("`"):
                out.append("`" + normalizeCodeParens(chunk[1:-1]) + "`")
            else:
                out.append(chunk)
        else:
            out.append(chunk)
    return "".join(out)


def processContent(content: str) -> str:
    out: list[str] = []
    for kind, chunk in splitFences(content):
        if kind == "fence":
            out.append(chunk)
        else:
            out.append(processTextChunk(chunk))
    return "".join(out)


def processFile(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = processContent(original)
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
        for path in kRoot.glob(pattern):
            if processFile(path):
                changed.append(path)

    for path in sorted(changed):
        print(path.relative_to(kRoot))
    print(f"\nUpdated {len(changed)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
