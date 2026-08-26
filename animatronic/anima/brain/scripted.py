"""A brain with no model behind it.

Three jobs, all of them real:

1. Develop the whole pipeline -- ears, voice, jaw, servos, timing -- without a
   model, an API key, or a network. Most of the hard engineering in this project
   has nothing to do with the language model.
2. Be the fallback when the network is down mid-exhibition. A puppet that says
   something evasive and in character beats a puppet that says nothing.
3. Be a deterministic fixture for tests.
"""

from __future__ import annotations

import itertools
import random
import re
from typing import Iterator, Sequence

from ..types import Turn

# Deliberately vague and deflecting: these are the lines a character can say to
# almost anything without breaking. Override them in config for your character.
DEFAULT_LINES = [
    "Mm. Say that again, slower.",
    "I heard you. I am deciding whether to answer.",
    "That was asked of me before. I forget by whom.",
    "The cold makes a liar of everyone. Ask me something else.",
    "You are standing very close.",
    "I have been thinking about that for a long time. I have not finished.",
]


class ScriptedBrain:
    """Replies from a fixed list, optionally keyed by what the visitor said.

    ``keyed`` maps a lowercase substring to a reply, checked before falling back
    to the rotating line pool -- enough to fake responsiveness in a demo, and
    enough to handle the two or three questions every visitor actually asks.
    """

    def __init__(
        self,
        lines: Sequence[str] | None = None,
        keyed: dict[str, str] | None = None,
        *,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        pool = list(lines) if lines else list(DEFAULT_LINES)
        if not pool:
            raise ValueError("ScriptedBrain needs at least one line")
        if shuffle:
            random.Random(seed).shuffle(pool)
        self._pool = pool
        self._cycle = itertools.cycle(pool)
        self._keyed = {k.lower(): v for k, v in (keyed or {}).items()}

    def respond(self, history: Sequence[Turn]) -> Iterator[str]:
        last = next(
            (t.text for t in reversed(history) if t.role == "visitor"),
            "",
        ).lower()

        for key, reply in self._keyed.items():
            if re.search(rf"\b{re.escape(key)}\b", last):
                yield reply
                return

        yield next(self._cycle)
