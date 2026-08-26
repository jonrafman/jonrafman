"""Brains: the swappable thing that decides what the character says.

Three implementations, one interface:

  scripted  no model at all -- canned lines. For development, for tests, and
            as the fallback when the network dies mid-exhibition.
  ollama    a local model on the box. Offline, free, private, simpler.
  claude    the Claude API. Best conversation, needs network, costs pennies.

Swap between them with one line of config. Build against `scripted`, decide
the brain later.
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

from ..types import Brain, Turn

log = logging.getLogger(__name__)


class BrainUnavailable(RuntimeError):
    """The brain cannot answer right now: no network, no server, no model.

    Distinct from a bug. Callers are expected to catch this and degrade to a
    fallback rather than crash -- an installation must survive its own wifi.
    """


class FallbackBrain:
    """Wraps a primary brain and drops to a backup when it fails.

    The subtlety is *when* the failure lands. A brain streams, so it may raise
    after the piece has already said two sentences out loud. Substituting a
    canned line at that point produces a non-sequitur, so we only fall back if
    nothing has been spoken yet; a mid-reply failure just ends the reply, which
    reads as the character trailing off.
    """

    def __init__(self, primary: Brain, backup: Brain) -> None:
        self.primary = primary
        self.backup = backup

    def respond(self, history: Sequence[Turn]) -> Iterator[str]:
        spoke = False
        try:
            for sentence in self.primary.respond(history):
                spoke = True
                yield sentence
        except BrainUnavailable as exc:
            if spoke:
                log.warning("Brain failed mid-reply, trailing off: %s", exc)
                return
            log.warning("Brain unavailable, using fallback: %s", exc)
            yield from self.backup.respond(history)


def build_brain(config, system_prompt: str) -> Brain:
    """Construct the configured brain, wrapped in a scripted fallback.

    ``config`` is the ``brain`` section of the loaded config object.
    """
    from .scripted import ScriptedBrain

    fallback = ScriptedBrain(
        lines=config.fallback_lines or None,
        keyed=config.scripted_keyed,
    )

    backend = config.backend.lower()
    if backend == "scripted":
        return ScriptedBrain(
            lines=config.scripted_lines or None,
            keyed=config.scripted_keyed,
        )

    if backend == "ollama":
        from .ollama import OllamaBrain

        primary = OllamaBrain(
            host=config.ollama_host,
            model=config.ollama_model,
            system_prompt=system_prompt,
            temperature=config.temperature,
            num_predict=config.max_tokens,
        )
    elif backend == "claude":
        from .claude import ClaudeBrain

        primary = ClaudeBrain(
            model=config.claude_model,
            system_prompt=system_prompt,
            max_tokens=config.max_tokens,
            refusal_reply=(config.fallback_lines or ["Ask me something else."])[0],
        )
    elif backend == "llamacpp":
        from .llamacpp import LlamaCppBrain

        primary = LlamaCppBrain(
            model_path=config.llamacpp_model,
            system_prompt=system_prompt,
            grammar_path=config.llamacpp_grammar,
            n_ctx=config.llamacpp_n_ctx,
            n_gpu_layers=config.llamacpp_n_gpu_layers,
            temperature=config.temperature,
            repeat_penalty=config.repeat_penalty,
            max_tokens=config.max_tokens,
        )
    else:
        raise ValueError(
            f"Unknown brain backend '{config.backend}'. "
            "Expected one of: scripted, ollama, claude, llamacpp."
        )

    return FallbackBrain(primary, fallback)


__all__ = ["Brain", "BrainUnavailable", "FallbackBrain", "build_brain"]
