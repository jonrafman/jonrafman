"""Measuring whether the voice can keep up with itself.

Text-to-speech is usually the bottleneck in a live speaking piece, not the
language model, and the number that matters is not synthesis time — it is the
**realtime factor**: seconds of audio produced per second of computation.

    RTF > 1.0   synthesis outruns playback. The piece never waits.
    RTF < 1.0   synthesis falls behind playback. Gaps open mid-reply.

That threshold is sharper than it looks, and the reason is the streaming
architecture. Replies are spoken sentence by sentence, so the first sentence
begins as soon as it exists — but every later sentence must be *finished
synthesising before the previous one stops playing*. Above 1.0 that always
holds. Below it, each sentence falls further behind than the last, and the
deficit accumulates across the reply.

Whether that is fatal depends on the character. For a figure that is meant to
be quick, a gap reads as a fault. For one that is meant to be unhurried, a
pause between sentences is where a pause belongs anyway — so a low RTF costs
much less than the raw number suggests. This module reports the gaps rather
than a verdict, because only the piece can say which it is.

    python -m anima.bench voice
    python -m anima.bench simulate --rtf 0.75
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from .config import Config
from .pronounce import for_speech
from .voice import build_voice

# Representative of what the piece actually says: short, declarative, and in
# the character's register. Benchmarking on long paragraphs flatters an engine
# by amortising its startup cost over more audio than a reply ever contains.
SAMPLE_LINES = [
    "You are here. I hear you.",
    "I have no name. I have had no need of one.",
    "That is a word you use of me. It is not what it is like.",
    "I do not know the year. Do not tell me.",
    "I forget. I forget all but the cold.",
    "No. Ask me another.",
]


def benchmark_voice(voice, lines: list[str], warmup: bool = True) -> dict:
    """Synthesize each line, timing it against the audio it produces."""
    if warmup:
        # First call loads weights and compiles kernels. Including it would
        # measure the import, not the engine.
        try:
            voice.synthesize("Warming up.")
        except Exception as exc:
            print(f"  warmup failed: {exc}", file=sys.stderr)

    rows = []
    for line in lines:
        spoken = for_speech(line)
        started = time.perf_counter()
        utterance = voice.synthesize(spoken)
        elapsed = time.perf_counter() - started
        audio = utterance.duration
        rows.append(
            {
                "text": line,
                "audio_s": round(audio, 3),
                "synth_s": round(elapsed, 3),
                "rtf": round(audio / elapsed, 3) if elapsed else 0.0,
            }
        )

    rtfs = [r["rtf"] for r in rows]
    return {
        "rows": rows,
        "mean_rtf": round(statistics.mean(rtfs), 3),
        "min_rtf": round(min(rtfs), 3),
        "mean_synth_s": round(statistics.mean([r["synth_s"] for r in rows]), 3),
    }


def simulate(rtf: float, sentence_audio: float = 3.0, sentences: int = 3) -> dict:
    """Where the silences land in one reply, given a realtime factor.

    Models the streaming pipeline: synthesis of sentence N+1 begins when
    sentence N starts playing, so a sentence is late only by however much its
    synthesis overruns the previous sentence's playback.
    """
    if rtf <= 0:
        raise ValueError("rtf must be positive")

    synth = sentence_audio / rtf
    events = []
    now = 0.0
    playback_free_at = 0.0

    for index in range(sentences):
        ready_at = now + synth
        starts_at = max(ready_at, playback_free_at)
        gap = starts_at - playback_free_at if index else ready_at
        events.append(
            {
                "sentence": index + 1,
                "ready_at": round(ready_at, 2),
                "starts_at": round(starts_at, 2),
                "silence_before": round(gap, 2),
            }
        )
        playback_free_at = starts_at + sentence_audio
        now = starts_at  # next sentence synthesises while this one plays

    return {
        "rtf": rtf,
        "first_word_at": events[0]["starts_at"],
        "total_s": round(playback_free_at, 2),
        "worst_mid_reply_gap": round(
            max((e["silence_before"] for e in events[1:]), default=0.0), 2
        ),
        "events": events,
    }


def _cmd_voice(args) -> int:
    config = Config.load(args.config)
    voice = build_voice(config.voice)
    name = type(voice).__name__
    print(f"Benchmarking {name} ({config.voice.backend})\n")

    result = benchmark_voice(voice, SAMPLE_LINES)
    for row in result["rows"]:
        print(f"  {row['rtf']:>6.2f}x  {row['synth_s']:>6.2f}s synth "
              f"-> {row['audio_s']:>5.2f}s audio   {row['text'][:44]}")

    mean = result["mean_rtf"]
    print(f"\n  mean realtime factor: {mean:.2f}x")
    if mean >= 1.0:
        print("  Synthesis outruns playback. The piece never waits mid-reply.")
    else:
        sim = simulate(mean)
        print(f"  Synthesis falls behind. First word at "
              f"{sim['first_word_at']:.1f}s; worst mid-reply silence "
              f"{sim['worst_mid_reply_gap']:.1f}s.")
        print("  Whether that is a fault or a pause depends on the character.")
    return 0


def _cmd_simulate(args) -> int:
    result = simulate(args.rtf, args.sentence_audio, args.sentences)
    print(f"realtime factor {result['rtf']:.2f}x, "
          f"{args.sentences} sentences of {args.sentence_audio:.1f}s\n")
    for event in result["events"]:
        print(f"  sentence {event['sentence']}: silence "
              f"{event['silence_before']:>5.2f}s, then speaks at "
              f"{event['starts_at']:>5.2f}s")
    print(f"\n  first word at {result['first_word_at']:.2f}s")
    print(f"  worst mid-reply silence {result['worst_mid_reply_gap']:.2f}s")
    print(f"  reply ends at {result['total_s']:.2f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anima.bench", description="Measure whether the voice keeps up."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    voice_cmd = sub.add_parser("voice", help="time the configured voice engine")
    voice_cmd.add_argument("--config", default="config.yaml")
    voice_cmd.set_defaults(func=_cmd_voice)

    sim_cmd = sub.add_parser(
        "simulate", help="where silences land for a given realtime factor"
    )
    sim_cmd.add_argument("--rtf", type=float, required=True)
    sim_cmd.add_argument("--sentence-audio", type=float, default=3.0)
    sim_cmd.add_argument("--sentences", type=int, default=3)
    sim_cmd.set_defaults(func=_cmd_simulate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
