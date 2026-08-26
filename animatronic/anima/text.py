"""Turning a token stream into speakable sentences.

This is the single most important latency trick in the whole project. A model
streams tokens; text-to-speech wants whole phrases. If you wait for the full
reply before speaking, the piece sits dead for several seconds and the
illusion dies. If you speak every token, the speech is choppy garbage.

So: accumulate tokens, emit as soon as a sentence boundary lands, and force an
emit at a clause boundary if the buffer grows long enough that waiting would be
worse than an slightly awkward break.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

# Abbreviations whose trailing period is not a sentence end. Deliberately short:
# this is spoken language, not a legal brief.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
}

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s|$)")
_CLAUSE_BREAK = re.compile(r"[,;:—–-]+(?=\s)")


def _is_abbreviation(buffer: str, end_index: int) -> bool:
    """True if the period at ``end_index`` closes a known abbreviation."""
    word = re.search(r"([A-Za-z.]+)\.$", buffer[:end_index])
    if not word:
        return False
    return word.group(1).lower().rstrip(".") in _ABBREVIATIONS


def _split_once(buffer: str, soft_limit: int) -> tuple[str, str] | None:
    """Find the best place to cut ``buffer``, or None if we should keep buffering.

    Prefers a real sentence ending. Falls back to a clause break only once the
    buffer is long enough that the audience would notice the silence.
    """
    for match in _SENTENCE_END.finditer(buffer):
        end = match.end()
        if _is_abbreviation(buffer, end):
            continue
        return buffer[:end], buffer[end:]

    if len(buffer) >= soft_limit:
        # No sentence in sight and we've waited long enough. Break at the last
        # clause boundary so the pause at least lands somewhere natural.
        breaks = list(_CLAUSE_BREAK.finditer(buffer))
        if breaks:
            end = breaks[-1].end()
            return buffer[:end], buffer[end:]

    return None


def chunk_sentences(
    tokens: Iterable[str],
    soft_limit: int = 180,
    min_length: int = 2,
) -> Iterator[str]:
    """Yield speakable sentences from a stream of text fragments.

    Args:
        tokens: fragments as they arrive from the model. May be single
            characters, whole words, or whole paragraphs -- it does not matter.
        soft_limit: buffer length past which we accept a clause break instead
            of holding out for a sentence ending.
        min_length: fragments shorter than this are merged forward rather than
            spoken alone, so a stray "Oh." does not become its own utterance
            with its own audio-device startup cost.
    """
    buffer = ""
    for token in tokens:
        if not token:
            continue
        buffer += token
        while True:
            split = _split_once(buffer, soft_limit)
            if split is None:
                break
            head, buffer = split
            head = head.strip()
            if len(head) >= min_length:
                yield head
            else:
                # Too short to be worth an utterance of its own; put it back on
                # the front of the buffer and let it ride with the next one.
                buffer = f"{head} {buffer.lstrip()}"
                break

    tail = buffer.strip()
    if tail:
        yield tail


def strip_stage_directions(text: str) -> str:
    """Remove *asides*, (parentheticals), and [brackets] from model output.

    Models love to narrate. Text-to-speech will happily read "asterisk leans
    forward asterisk" out loud, which is funny exactly once.
    """
    text = re.sub(r"\*[^*]{0,120}\*", " ", text)
    text = re.sub(r"\[[^\]]{0,120}\]", " ", text)
    text = re.sub(r"\([^)]{0,120}\)", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def limit_sentences(text: str, max_sentences: int) -> str:
    """Hard-cap a reply. Long monologues are the death of an installation."""
    if max_sentences <= 0:
        return text
    sentences = list(chunk_sentences([text]))
    return " ".join(sentences[:max_sentences])
