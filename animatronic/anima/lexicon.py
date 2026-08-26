"""Building a closed vocabulary from a corpus.

The point of this module is that the constraint becomes real. A prompt asking a
model to restrict its vocabulary will drift — the model has no mechanism to
check itself against a word list mid-sentence. A lexicon compiled here, applied
at the sampler, makes the restriction a property of the machine: the model
cannot emit a word outside the corpus, because those tokens are never available
to it.

That enforcement is only possible with a local model. It is the strongest
argument for running one.

    python -m anima.lexicon build corpus/*.txt -o lexicon/blake.json
    python -m anima.lexicon stats lexicon/blake.json

A note on what a closed vocabulary does to a model: it will fight you. Modern
models are trained toward modern usage, and forcing them through a nineteenth
century lexicon produces stranger, stiffer, more archaic output than the same
model unconstrained. That is presumably the point — but it means the persona
prompt should ask for the register the lexicon will impose anyway, so the model
is pushing in the same direction as the constraint rather than against it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Words are kept as they appear, minus surrounding punctuation. Internal
# apostrophes and hyphens are part of the word: "thou'rt" and "self-annihilation"
# are single lexical items, and splitting them would silently admit fragments
# that never appear in the corpus on their own.
_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")

# Project Gutenberg wraps texts in a licence header and footer. Including them
# would quietly admit words like "copyright" and "ebook" into the lexicon.
_PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
_PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)


def strip_boilerplate(text: str) -> str:
    """Remove a Project Gutenberg header and footer if present."""
    start = _PG_START.search(text)
    if start:
        text = text[start.end():]
    end = _PG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def extract_words(text: str) -> Counter[str]:
    """Count word forms in a text, lowercased.

    Forms, not lemmas: if the corpus has "burning" but never "burn", then
    "burn" is not in the lexicon. That is the honest reading of "words found
    in the oeuvre", and it is what makes the constraint bite.
    """
    return Counter(match.group(0).lower() for match in _WORD.finditer(text))


def build(
    paths: list[Path],
    *,
    min_count: int = 1,
    strip_pg: bool = True,
) -> dict:
    """Compile a lexicon from one or more corpus files.

    Args:
        paths: text files making up the corpus.
        min_count: drop forms appearing fewer than this many times. Raising it
            to 2 removes most scanning errors and typographical debris at the
            cost of losing genuine hapax legomena — and in a poet, the rare
            word is often the one you wanted. Default keeps everything.
        strip_pg: remove Project Gutenberg boilerplate.
    """
    counts: Counter[str] = Counter()
    sources: list[dict] = []

    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if strip_pg:
            text = strip_boilerplate(text)
        found = extract_words(text)
        counts.update(found)
        sources.append(
            {"path": str(path), "words": sum(found.values()), "distinct": len(found)}
        )

    if min_count > 1:
        counts = Counter({w: c for w, c in counts.items() if c >= min_count})

    return {
        "words": sorted(counts),
        "counts": dict(counts.most_common()),
        "sources": sources,
        "total_words": sum(counts.values()),
        "distinct_words": len(counts),
        "min_count": min_count,
    }


def load(path: str | Path) -> set[str]:
    """Load a compiled lexicon as a set of allowed word forms."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(data["words"])


def coverage(lexicon: set[str], text: str) -> tuple[float, list[str]]:
    """What fraction of ``text`` the lexicon admits, and what it rejects.

    Use this to sanity-check a persona file before wiring the constraint in:
    if the persona itself is full of words the lexicon forbids, the model is
    being shown a register it is then forbidden from producing.
    """
    words = [m.group(0).lower() for m in _WORD.finditer(text)]
    if not words:
        return 1.0, []
    missing = [w for w in words if w not in lexicon]
    return 1.0 - len(missing) / len(words), sorted(set(missing))


def _cmd_build(args) -> int:
    paths = [Path(p) for p in args.corpus]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"Not a file: {p}", file=sys.stderr)
        return 2

    data = build(paths, min_count=args.min_count, strip_pg=not args.keep_boilerplate)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"{data['distinct_words']:,} distinct forms from {data['total_words']:,} words")
    for source in data["sources"]:
        print(f"  {source['distinct']:>7,} distinct  {source['path']}")
    print(f"\nWrote {out}")
    return 0


def _cmd_stats(args) -> int:
    data = json.loads(Path(args.lexicon).read_text(encoding="utf-8"))
    counts = data.get("counts", {})
    print(f"{data['distinct_words']:,} distinct forms, {data['total_words']:,} total")
    print(f"\nMost common:")
    for word, count in list(counts.items())[:20]:
        print(f"  {count:>6,}  {word}")
    hapax = [w for w, c in counts.items() if c == 1]
    print(f"\n{len(hapax):,} forms appear exactly once ({len(hapax)/max(1,len(counts)):.0%})")
    return 0


def _cmd_check(args) -> int:
    lexicon = load(args.lexicon)
    text = Path(args.text).read_text(encoding="utf-8")
    fraction, missing = coverage(lexicon, text)
    print(f"{fraction:.1%} of {args.text} is inside the lexicon")
    if missing:
        print(f"\n{len(missing)} forms not in the lexicon:")
        for word in missing[:60]:
            print(f"  {word}")
        if len(missing) > 60:
            print(f"  ... and {len(missing) - 60} more")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anima.lexicon", description="Compile a closed vocabulary from a corpus."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="compile a lexicon from text files")
    build_cmd.add_argument("corpus", nargs="+", help="corpus text files")
    build_cmd.add_argument("-o", "--output", required=True, help="output .json path")
    build_cmd.add_argument(
        "--min-count", type=int, default=1,
        help="drop forms rarer than this (default 1: keep everything)",
    )
    build_cmd.add_argument(
        "--keep-boilerplate", action="store_true",
        help="do not strip Project Gutenberg headers and footers",
    )
    build_cmd.set_defaults(func=_cmd_build)

    stats_cmd = sub.add_parser("stats", help="summarise a compiled lexicon")
    stats_cmd.add_argument("lexicon")
    stats_cmd.set_defaults(func=_cmd_stats)

    check_cmd = sub.add_parser("check", help="check a text against a lexicon")
    check_cmd.add_argument("lexicon")
    check_cmd.add_argument("text")
    check_cmd.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # Piping into `head` closes stdout early. Without this the traceback
        # lands on top of otherwise-correct output, and Python raises a second
        # error flushing stdout at exit -- so point the fd at devnull first.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
