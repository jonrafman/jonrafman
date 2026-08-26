"""Core data types shared across the pipeline.

Kept dependency-free on purpose: every module imports from here, and nothing
here should ever pull in audio, GPIO, or network libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Protocol, Sequence

Role = Literal["visitor", "character"]


@dataclass(frozen=True)
class Turn:
    """One utterance in a conversation."""

    role: Role
    text: str


@dataclass
class Utterance:
    """Synthesized speech, ready to be played and rendered as light.

    ``samples`` is mono float32 in [-1, 1]. We keep raw samples rather than a
    file path because the light is driven from the waveform's amplitude
    envelope, and re-reading a temp file per sentence is wasted latency.
    """

    samples: Sequence[float]
    sample_rate: int
    text: str = ""

    @property
    def duration(self) -> float:
        if not self.sample_rate:
            return 0.0
        return len(self.samples) / self.sample_rate


class Brain(Protocol):
    """Turns conversation history into speech, one sentence at a time.

    Implementations MUST yield complete sentences as soon as they are known
    rather than returning a finished paragraph. The whole installation's
    perceived latency depends on the first sentence arriving fast: the puppet
    starts talking while the rest of the reply is still being generated.
    """

    def respond(self, history: Sequence[Turn]) -> Iterator[str]: ...


class Voice(Protocol):
    """Text to audio."""

    def synthesize(self, text: str) -> Utterance: ...


class Light(Protocol):
    """The sculpture's only physical output.

    ``set_level`` takes a *perceptual* level in [0, 1]; implementations apply
    their own gamma correction on the way to the hardware.
    """

    def set_level(self, level: float) -> None: ...

    def rest(self) -> None: ...


class Ears(Protocol):
    """Captures what a visitor said. Returns None if they said nothing."""

    def listen(self, timeout: float) -> str | None: ...


class Trigger(Protocol):
    """Blocks until a visitor is present."""

    def wait(self) -> None: ...


@dataclass
class Conversation:
    """A single visitor's exchange. Discarded when they walk away."""

    turns: list[Turn] = field(default_factory=list)

    def add(self, role: Role, text: str) -> None:
        self.turns.append(Turn(role, text))

    def tail(self, max_turns: int) -> list[Turn]:
        """Most recent ``max_turns`` turns, oldest first."""
        if max_turns <= 0:
            return list(self.turns)
        return self.turns[-max_turns:]
