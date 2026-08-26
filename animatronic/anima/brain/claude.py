"""Claude as the character's brain.

Needs network, costs pennies per visitor, and is a dramatically better
conversationalist than anything that fits on a Raspberry Pi. Pair it with a
scripted fallback so a dropped connection degrades into an evasive presence
rather than a dead one.

Model choice for this medium:
  claude-opus-5    - the default. A naturalistic puppet must answer instantly
                     or it reads as broken, but an unhurried intelligence is
                     *supposed* to deliberate, and the thinking light makes the
                     wait legible. That buys the budget for the better model.
  claude-sonnet-5  - roughly half the cost and noticeably faster. Choose it if
                     the piece wants a quicker, less considered voice.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from ..text import chunk_sentences, strip_stage_directions
from ..types import Turn
from . import BrainUnavailable

_ROLE_MAP = {"visitor": "user", "character": "assistant"}

# Opus 5 misbehaves with thinking switched off -- it can leak reasoning tags
# into the spoken text. Low effort gets the same latency win safely.
_NO_DISABLE_THINKING = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")


class ClaudeBrain:
    """Streams a reply from the Claude API.

    Args:
        model: model id, e.g. ``claude-sonnet-5``.
        system_prompt: assembled persona + stage rules.
        api_key: falls back to ANTHROPIC_API_KEY / an ``ant auth login`` profile
            when omitted.
        max_tokens: replies are one to three sentences, so this is a backstop,
            not a target.
        refusal_reply: spoken when a safety classifier declines the request.
            Visitors in a gallery will absolutely try to make the piece say
            something vile; a deflection in character beats an error.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 300,
        cache_persona: bool = True,
        refusal_reply: str = "No. Ask me something else.",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrainUnavailable(
                "The 'anthropic' package is required for the claude brain. "
                "pip install anthropic"
            ) from exc

        # A bare constructor also picks up an `ant auth login` profile, so do
        # not demand an explicit key.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._anthropic = anthropic
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.cache_persona = cache_persona
        self.refusal_reply = refusal_reply

    def _system_blocks(self) -> list[dict]:
        block: dict = {"type": "text", "text": self.system_prompt}
        if self.cache_persona:
            # The persona is byte-identical on every turn, so it caches cleanly.
            # Note the ~1024-token minimum: a short persona simply will not
            # cache, which is harmless but explains a zero hit rate.
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _latency_params(self) -> dict:
        if self.model.startswith(_NO_DISABLE_THINKING):
            return {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low"},
            }
        return {"thinking": {"type": "disabled"}}

    def _tokens(self, history: Sequence[Turn]) -> Iterator[str]:
        messages = [
            {"role": _ROLE_MAP[turn.role], "content": turn.text} for turn in history
        ]
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system_blocks(),
                messages=messages,
                **self._latency_params(),
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except self._anthropic.APIStatusError as exc:
            raise BrainUnavailable(f"Claude API error {exc.status_code}: {exc}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise BrainUnavailable(f"Claude API unreachable: {exc}") from exc

        if final.stop_reason == "refusal":
            yield self.refusal_reply

    def respond(self, history: Sequence[Turn]) -> Iterator[str]:
        for sentence in chunk_sentences(self._tokens(history)):
            cleaned = strip_stage_directions(sentence)
            if cleaned:
                yield cleaned

    def health_check(self) -> str:
        """Cheapest possible round trip that proves auth and model access."""
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
        except self._anthropic.AuthenticationError as exc:
            raise BrainUnavailable(
                "Claude API rejected the credentials. Set ANTHROPIC_API_KEY "
                "or run 'ant auth login'."
            ) from exc
        except self._anthropic.APIError as exc:
            raise BrainUnavailable(f"Claude API check failed: {exc}") from exc
        return self.model
