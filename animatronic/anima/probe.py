"""Running a fixed battery of questions and measuring what comes back.

Judging a character from one conversation is worthless: you remember the good
answer and forget the four bad ones. A fixed probe set makes comparison real --
same questions, every configuration, side by side.

Three things get measured, and each answers an open question:

**Lexicon compliance.** What fraction of each reply is inside the closed
vocabulary. With the grammar loaded this must be 100%; anything less is a bug
in the grammar and worth catching loudly. With the grammar *off*, this number
is how far prompting alone gets you -- and the gap between the two runs is
exactly what the sampler constraint is worth. That gap is the argument for
running a local model at all, so it deserves to be a number rather than a
belief.

**Time to first sentence.** Not total generation time. The visitor experiences
the pause before the sculpture starts speaking, and sentence streaming means
that is the first sentence, not the last. This is the number that decides
whether a given machine is fast enough.

**Reply length.** Long answers lose people who are standing up in a room. If
the model routinely runs past the sentence cap, the prompt is losing to the
model's habits.

    python -m anima.probe --config config.yaml -o runs/grammar-on.json
    python -m anima.probe --compare runs/grammar-off.json runs/grammar-on.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .brain import build_brain
from .brain.prompt import build_system_prompt, load_persona
from .config import Config
from .lexicon import coverage, load as load_lexicon
from .types import Conversation


@dataclass
class Probe:
    id: str
    category: str
    turns: list[str]


@dataclass
class Result:
    id: str
    category: str
    question: str
    reply: str
    sentences: int
    words: int
    compliance: float | None
    forbidden: list[str] = field(default_factory=list)
    first_sentence_s: float = 0.0
    total_s: float = 0.0


def load_probes(path: str | Path) -> list[Probe]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    probes = data.get("probes") or []
    if not probes:
        raise ValueError(f"No probes found in {path}")
    return [
        Probe(
            id=str(p["id"]),
            category=str(p.get("category", "uncategorised")),
            turns=[str(t) for t in p["turns"]],
        )
        for p in probes
    ]


def run_probe(brain, probe: Probe, lexicon: set[str] | None, max_sentences: int) -> Result:
    """Run one probe. Only the final turn's reply is scored."""
    conversation = Conversation()
    sentences: list[str] = []
    first_at = 0.0
    started = 0.0

    for index, turn in enumerate(probe.turns):
        conversation.add("visitor", turn)
        is_last = index == len(probe.turns) - 1

        sentences = []
        started = time.perf_counter()
        first_at = 0.0

        replies = brain.respond(conversation.tail(12))
        try:
            for sentence in replies:
                if not first_at:
                    first_at = time.perf_counter() - started
                sentences.append(sentence)
                if len(sentences) >= max_sentences:
                    break
        except Exception as exc:
            sentences = [f"<ERROR: {exc}>"]
        finally:
            close = getattr(replies, "close", None)
            if callable(close):
                close()

        total = time.perf_counter() - started
        reply = " ".join(sentences)
        if not is_last:
            conversation.add("character", reply)

    compliance: float | None = None
    forbidden: list[str] = []
    if lexicon is not None:
        compliance, forbidden = coverage(lexicon, reply)

    return Result(
        id=probe.id,
        category=probe.category,
        question=probe.turns[-1],
        reply=reply,
        sentences=len(sentences),
        words=len(reply.split()),
        compliance=compliance,
        forbidden=forbidden,
        first_sentence_s=round(first_at, 3),
        total_s=round(total, 3),
    )


def summarise(results: list[Result]) -> dict:
    compliances = [r.compliance for r in results if r.compliance is not None]
    # Include zeros: a brain fast enough to round to 0.000s still has timing
    # worth reporting, and filtering falsy values silently drops it.
    firsts = [r.first_sentence_s for r in results]
    return {
        "probes": len(results),
        "mean_compliance": round(statistics.mean(compliances), 4) if compliances else None,
        "fully_compliant": sum(1 for c in compliances if c >= 1.0),
        "mean_words": round(statistics.mean([r.words for r in results]), 1),
        "mean_first_sentence_s": round(statistics.mean(firsts), 3) if firsts else None,
        "max_first_sentence_s": round(max(firsts), 3) if firsts else None,
    }


def report(results: list[Result], meta: dict) -> str:
    lines = [
        f"# Probe run: {meta.get('label', 'unlabelled')}",
        "",
        f"backend `{meta.get('backend')}`  model `{meta.get('model') or '-'}`  "
        f"grammar `{meta.get('grammar') or 'NONE'}`",
        "",
    ]
    summary = summarise(results)
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    category = None
    for result in results:
        if result.category != category:
            category = result.category
            lines += ["", f"## {category}", ""]
        flag = ""
        if result.compliance is not None and result.compliance < 1.0:
            flag = f"  ⚠ outside lexicon: {', '.join(result.forbidden)}"
        lines.append(f"**{result.question}**")
        lines.append(f"> {result.reply or '(nothing)'}")
        lines.append(
            f"`{result.words}w  {result.sentences}s  "
            f"first={result.first_sentence_s}s`{flag}"
        )
        lines.append("")
    return "\n".join(lines)


def compare(path_a: str, path_b: str) -> str:
    """Diff two runs question by question. Blind by design: no scoring here.

    Which reply is better is a judgement about the character, and no metric
    settles it. What this shows is the two answers next to each other, plus the
    numbers that *are* objective.
    """
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    by_id_b = {r["id"]: r for r in b["results"]}

    lines = [
        "# Comparison",
        "",
        f"**A** — {a['meta'].get('label')}  (grammar `{a['meta'].get('grammar') or 'NONE'}`)",
        f"**B** — {b['meta'].get('label')}  (grammar `{b['meta'].get('grammar') or 'NONE'}`)",
        "",
    ]

    for key in ("mean_compliance", "fully_compliant", "mean_words", "mean_first_sentence_s"):
        lines.append(f"- **{key}**: A={a['summary'].get(key)}  B={b['summary'].get(key)}")
    lines.append("")

    for result in a["results"]:
        other = by_id_b.get(result["id"])
        if other is None:
            continue
        lines += [
            f"### {result['question']}",
            f"- **A**: {result['reply'] or '(nothing)'}",
            f"- **B**: {other['reply'] or '(nothing)'}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anima.probe",
        description="Run a fixed battery of questions against the character.",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--probes", default="probes/watchman.yaml")
    parser.add_argument("--lexicon", default="", help="score replies against this lexicon")
    parser.add_argument("-o", "--output", default="", help="write JSON results here")
    parser.add_argument("--markdown", default="", help="also write a markdown report here")
    parser.add_argument("--label", default="", help="name this run in reports")
    parser.add_argument(
        "--compare", nargs=2, metavar=("A.json", "B.json"),
        help="diff two previous runs and exit",
    )
    args = parser.parse_args(argv)

    if args.compare:
        print(compare(*args.compare))
        return 0

    config = Config.load(args.config)
    probes = load_probes(args.probes)

    lexicon_path = args.lexicon or config.brain.llamacpp_grammar.replace(".gbnf", ".json")
    lexicon = None
    if lexicon_path and Path(lexicon_path).is_file():
        lexicon = load_lexicon(lexicon_path)
        print(f"scoring against {lexicon_path} ({len(lexicon):,} forms)", file=sys.stderr)

    persona = load_persona(config.resolve(config.character.persona_file))
    system_prompt = build_system_prompt(
        persona,
        max_sentences=config.character.max_reply_sentences,
        style=config.character.style_rules,
        lexicon_source=config.character.lexicon_source,
    )
    brain = build_brain(config.brain, system_prompt)

    results: list[Result] = []
    for index, probe in enumerate(probes, 1):
        print(f"  [{index}/{len(probes)}] {probe.id}", file=sys.stderr)
        results.append(
            run_probe(brain, probe, lexicon, config.character.max_reply_sentences)
        )

    # Report the model and grammar that were actually in play. Falling back to
    # another backend's settings makes a report claim a configuration that did
    # not run, which is worse than reporting nothing.
    backend = config.brain.backend.lower()
    model = {
        "llamacpp": config.brain.llamacpp_model,
        "ollama": config.brain.ollama_model,
        "claude": config.brain.claude_model,
    }.get(backend, "")
    meta = {
        "label": args.label or Path(args.config).stem,
        "backend": config.brain.backend,
        "model": model,
        "grammar": config.brain.llamacpp_grammar if backend == "llamacpp" else "",
        "temperature": config.brain.temperature,
        "persona": config.character.persona_file,
    }
    payload = {
        "meta": meta,
        "summary": summarise(results),
        "results": [asdict(r) for r in results],
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)

    text = report(results, meta)
    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(text, encoding="utf-8")
        print(f"Wrote {md}", file=sys.stderr)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
