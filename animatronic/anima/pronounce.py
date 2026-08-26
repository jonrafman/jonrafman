"""Spelling the character's words so a voice engine says them correctly.

Blake's orthography is two centuries older than any speech model's training
data. Left alone, a TTS engine reads `thro` as "throw", `shewd` as "shooed",
`o'er` as something between "oh-er" and silence, and `answerd` as "an-swerd".
The writing survives the constraint and then dies at the speaker.

The fix relies on a distinction that is easy to miss: **the lexicon governs
what the model may emit; this module governs what the voice engine receives.**
Those are different stages of the pipeline. Rewriting `vanish'd` to `vanished`
on the way to the speaker does not loosen the constraint by one word — the
sculpture still cannot say anything Blake did not write. It only spells what
it is already saying in a way the voice can pronounce.

The original spelling is what gets logged and displayed. Only the audio path
sees the normalised form.
"""

from __future__ import annotations

import re

# Elisions Blake writes with an apostrophe. Mapped explicitly rather than by
# rule, because the rule would also catch pronoun contractions -- "he'd" would
# become "heed".
ELISIONS = {
    "astonish'd": "astonished",
    "bruis'd": "bruised",
    "call'd": "called",
    "cherish'd": "cherished",
    "emerg'd": "emerged",
    "enter'd": "entered",
    "fix'd": "fixed",
    "impress'd": "impressed",
    "learn'd": "learned",
    "link'd": "linked",
    "liv'd": "lived",
    "pluck'd": "plucked",
    "rais'd": "raised",
    "stain'd": "stained",
    "vanish'd": "vanished",
    "view'd": "viewed",
    "wip'd": "wiped",
    "threat'ning": "threatening",
    "ta'en": "taken",
    "can'st": "canst",
    "know'st": "knowest",
    "whate'er": "whatever",
    "where'er": "wherever",
    "ne'er": "never",
    # o'er is one syllable in verse and two as "over". Spoken dialogue is not
    # metrical, and a mispronounced word costs more than a lost syllable.
    "o'er": "over",
    "o're": "over",
}

# Past tenses Blake spells without the 'e'. Every one of these is a real word
# to a reader and gibberish to a phonemiser.
BARE_PAST = {
    "answerd": "answered",
    "bowd": "bowed",
    "ceasd": "ceased",
    "complaind": "complained",
    "exhald": "exhaled",
    "reclind": "reclined",
    "saild": "sailed",
    "shewd": "showed",
    "smild": "smiled",
    "stord": "stored",
    "unhinderd": "unhindered",
}

# Early modern contractions. thou/thee/thy/hath/doth are deliberately absent --
# a speech model pronounces those correctly, and they are the character.
CONTRACTIONS = {
    "thro": "through",
    "tis": "it is",
    "twas": "it was",
}

# Early modern GRAMMAR, as distinct from early modern spelling.
#
# The maps above change how a word is written. This one changes which word it
# is, so applying it is a real decision about the piece rather than a
# housekeeping fix. Kept separate for exactly that reason.
#
# What is lost: "thou" is the intimate second person, and English no longer
# has one. Collapsing thou/thee/ye into "you" removes a distinction the
# language cannot otherwise make. What is gained: the sculpture stops sounding
# like a costume drama, and a visitor hears an intelligence rather than a
# pastiche. Only one of those is recoverable later, so choose deliberately.
ARCHAIC_GRAMMAR = {
    "thou": "you", "thee": "you", "ye": "you",
    "thy": "your", "thine": "yours",
    "art": "are", "wast": "were",
    "doth": "does", "dost": "do", "didst": "did",
    "hath": "has", "hast": "have",
    "shalt": "shall", "wilt": "will",
    "canst": "can", "can'st": "can",
    "knowest": "know", "know'st": "know",
    "seest": "see", "sittest": "sit", "fearest": "fear",
    "countest": "count", "complainest": "complain",
    "walketh": "walks", "seeketh": "seeks",
    "accepteth": "accepts", "answereth": "answers", "ariseth": "arises",
}

# Pronoun contractions that the generic "'d -> ed" rule would ruin.
_KEEP = {"he'd", "she'd", "we'd", "i'd", "you'd", "they'd", "it'd", "who'd", "that'd"}

SUBSTITUTIONS = {**ELISIONS, **BARE_PAST, **CONTRACTIONS}

_TOKEN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def _match_case(original: str, replacement: str) -> str:
    """Carry the original's capitalisation onto the replacement."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _substitute(token: str) -> str:
    lowered = token.replace("’", "'").lower()

    if lowered in _KEEP:
        return token

    replacement = SUBSTITUTIONS.get(lowered)
    if replacement is not None:
        return _match_case(token, replacement)

    # Generic fallback for elisions not in the table -- a lexicon built from a
    # different Blake edition, or a different poet entirely, will have its own.
    if lowered.endswith("'d") and len(lowered) > 3:
        return _match_case(token, lowered[:-2] + "ed")

    return token


def for_speech(text: str) -> str:
    """Rewrite archaic spellings into forms a speech engine can pronounce."""
    return _TOKEN.sub(lambda m: _substitute(m.group(0)), text)


def unmapped(words: set[str]) -> list[str]:
    """Forms in a lexicon that look archaic but have no mapping.

    Run this when swapping in a new corpus: a lexicon from a different edition
    or a different poet will carry its own spellings, and the failure mode is
    silent -- the piece simply mispronounces a word every time it says it.
    """
    suspicious = []
    for word in sorted(words):
        lowered = word.lower()
        if lowered in SUBSTITUTIONS or lowered in _KEEP:
            continue
        if "'" in lowered and not lowered.endswith("'s"):
            suspicious.append(word)
        elif re.search(r"[bcdfghjklmnpqrstvwxz]d$", lowered) and not lowered.endswith("ed"):
            suspicious.append(word)
    return suspicious


def main(argv: list[str] | None = None) -> int:
    import argparse
    import textwrap

    from .lexicon import load

    parser = argparse.ArgumentParser(
        prog="anima.pronounce",
        description="Check a lexicon for spellings a voice engine will mangle.",
    )
    parser.add_argument("lexicon", help="compiled lexicon .json")
    parser.add_argument("--say", metavar="TEXT", help="show how one line is normalised")
    args = parser.parse_args(argv)

    if args.say:
        print(f"written: {args.say}")
        print(f"spoken : {for_speech(args.say)}")
        return 0

    words = load(args.lexicon)
    flagged = unmapped(words)
    print(f"{len(SUBSTITUTIONS)} mappings; {len(flagged)} unmapped forms look archaic\n")
    if flagged:
        print(textwrap.fill(", ".join(flagged), 100))
        print(
            "\nMost of these are ordinary words ending in -d (hand, world). Add real "
            "archaisms to BARE_PAST or ELISIONS in anima/pronounce.py."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
