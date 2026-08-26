"""Local Llama (or any Ollama model) as the character's brain.

Fully offline once the model is pulled: no network, no API key, no per-visitor
cost, and nothing leaves the box. The tradeoff is capability -- a model small
enough to run on a Raspberry Pi is a noticeably simpler conversationalist than a
frontier model, and slower. See the README for what to expect per board.
"""

from __future__ import annotations

import json
from typing import Iterator, Sequence
from urllib.parse import urlparse

from ..text import chunk_sentences, strip_stage_directions
from ..types import Turn
from . import BrainUnavailable

_ROLE_MAP = {"visitor": "user", "character": "assistant"}


class OllamaBrain:
    """Streams a reply from a local Ollama server.

    Args:
        host: base URL of the Ollama server, e.g. ``http://127.0.0.1:11434``.
        model: model tag as pulled, e.g. ``llama3.2:3b``.
        system_prompt: the assembled persona + stage rules.
        temperature: higher is stranger. Characters usually want 0.7-0.9.
        num_predict: hard token ceiling on a reply, as a backstop to the
            "keep it short" instruction in the prompt.
        timeout: seconds to wait for the *first* token. Generation itself
            streams, so this bounds the stall, not the whole reply.
    """

    def __init__(
        self,
        host: str,
        model: str,
        system_prompt: str,
        *,
        temperature: float = 0.8,
        num_predict: int = 160,
        timeout: float = 30.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        self._session = self._make_session()

    def _make_session(self):
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrainUnavailable(
                "The 'requests' package is required for the ollama brain. "
                "pip install requests"
            ) from exc

        session = requests.Session()
        # A proxy configured for general egress must not swallow loopback
        # traffic. This bites on managed/corporate machines and is invisible
        # until you wonder why the puppet cannot reach a server on its own box.
        hostname = urlparse(self.host).hostname or ""
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            session.trust_env = False
        return session

    def _messages(self, history: Sequence[Turn]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [
            {"role": _ROLE_MAP[turn.role], "content": turn.text} for turn in history
        ]
        return messages

    def _tokens(self, history: Sequence[Turn]) -> Iterator[str]:
        import requests

        payload = {
            "model": self.model,
            "messages": self._messages(history),
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        try:
            response = self._session.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BrainUnavailable(f"Ollama unreachable at {self.host}: {exc}") from exc

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise BrainUnavailable(f"Ollama error: {event['error']}")
            piece = event.get("message", {}).get("content", "")
            if piece:
                yield piece
            if event.get("done"):
                break

    def respond(self, history: Sequence[Turn]) -> Iterator[str]:
        for sentence in chunk_sentences(self._tokens(history)):
            cleaned = strip_stage_directions(sentence)
            if cleaned:
                yield cleaned

    def health_check(self) -> str:
        """Confirm the server is up and the model is present. Raises if not."""
        import requests

        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BrainUnavailable(
                f"Ollama is not running at {self.host}. Start it with 'ollama serve'."
            ) from exc

        tags = [m.get("name", "") for m in response.json().get("models", [])]
        # Ollama reports 'llama3.2:3b'; users routinely configure bare 'llama3.2'.
        if not any(t == self.model or t.startswith(f"{self.model}:") for t in tags):
            raise BrainUnavailable(
                f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}\n"
                f"Available: {', '.join(tags) or '(none)'}"
            )
        return self.model
