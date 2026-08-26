"""A local model with a hard vocabulary constraint.

This is the backend the piece is actually built for. llama.cpp runs the model
in-process, which gives access to the sampler — and the sampler is where the
constraint stops being a request and becomes a fact. With a grammar loaded,
tokens that cannot continue a legal string are masked out before sampling.
The model does not decline to use a word Blake never wrote; that word is not
among the options.

Ollama, by contrast, exposes JSON-schema structured output but not GBNF
grammars, so it cannot do this. That is the whole reason this module exists
rather than reusing the Ollama backend.

The model is loaded eagerly at construction. Loading a multi-gigabyte model
takes seconds to minutes, and an installation should pay that at startup
rather than making the first visitor of the day wait through it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Sequence

from ..text import chunk_sentences, strip_stage_directions
from ..types import Turn
from . import BrainUnavailable

log = logging.getLogger(__name__)

_ROLE_MAP = {"visitor": "user", "character": "assistant"}


class LlamaCppBrain:
    """Streams a reply from a local GGUF model, optionally grammar-constrained.

    Args:
        model_path: path to a ``.gguf`` model file.
        system_prompt: assembled persona, stage rules, style and lexicon note.
        grammar_path: a ``.gbnf`` file from ``anima.grammar``. Omit to run
            unconstrained — useful for hearing what the model says *before*
            the lexicon is imposed, which is worth doing at least once.
        n_ctx: context window. The whole conversation plus the persona must
            fit; 4096 is ample for short exchanges and keeps loading quick.
        n_gpu_layers: -1 offloads everything to the GPU. On Apple Silicon this
            means Metal, and it is the difference between usable and not.
        temperature: with a grammar clamping the vocabulary, higher values are
            safer than usual — the model cannot wander outside the lexicon, so
            the risk of raising it is strangeness rather than nonsense.
        repeat_penalty: matters more here than in unconstrained generation. A
            small vocabulary makes loops easy to fall into.
    """

    def __init__(
        self,
        model_path: str,
        system_prompt: str,
        *,
        grammar_path: str = "",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.85,
        top_p: float = 0.95,
        repeat_penalty: float = 1.15,
        max_tokens: int = 160,
        chat_format: str | None = None,
        seed: int | None = None,
    ) -> None:
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrainUnavailable(
                "llama-cpp-python is required for the llamacpp brain.\n"
                "  pip install llama-cpp-python\n"
                "On Apple Silicon, Metal support is built in by default."
            ) from exc

        if not model_path:
            raise BrainUnavailable(
                "No model configured. Set brain.llamacpp.model in config.yaml "
                "to the path of a .gguf file."
            )
        model = Path(model_path)
        if not model.is_file():
            raise BrainUnavailable(
                f"Model not found: {model}\n"
                "Download a GGUF and point brain.llamacpp.model at it."
            )

        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.max_tokens = max_tokens
        self.model_path = model

        self.grammar = None
        if grammar_path:
            grammar_file = Path(grammar_path)
            if not grammar_file.is_file():
                raise BrainUnavailable(
                    f"Grammar not found: {grammar_file}\n"
                    "Build one with: python -m anima.grammar <lexicon> -o <out.gbnf>"
                )
            try:
                self.grammar = LlamaGrammar.from_file(str(grammar_file), verbose=False)
            except Exception as exc:
                raise BrainUnavailable(
                    f"Grammar {grammar_file} failed to parse: {exc}"
                ) from exc
            log.info("Vocabulary constrained by %s", grammar_file)
        else:
            log.warning("No grammar loaded -- the vocabulary is NOT constrained.")

        log.info("Loading %s ...", model.name)
        kwargs = dict(
            model_path=str(model),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        if chat_format:
            kwargs["chat_format"] = chat_format
        if seed is not None:
            kwargs["seed"] = seed

        try:
            self._llm = Llama(**kwargs)
        except Exception as exc:
            raise BrainUnavailable(f"Could not load {model}: {exc}") from exc
        log.info("Loaded %s", model.name)

    def _messages(self, history: Sequence[Turn]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [
            {"role": _ROLE_MAP[turn.role], "content": turn.text} for turn in history
        ]
        return messages

    def _tokens(self, history: Sequence[Turn]) -> Iterator[str]:
        try:
            stream = self._llm.create_chat_completion(
                messages=self._messages(history),
                stream=True,
                grammar=self.grammar,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
            )
        except Exception as exc:
            raise BrainUnavailable(f"Generation failed: {exc}") from exc

        for chunk in stream:
            try:
                delta = chunk["choices"][0]["delta"]
            except (KeyError, IndexError):
                continue
            piece = delta.get("content")
            if piece:
                yield piece

    def respond(self, history: Sequence[Turn]) -> Iterator[str]:
        for sentence in chunk_sentences(self._tokens(history)):
            # With a grammar loaded this is belt-and-braces: asterisks and
            # brackets are not in the lexicon so they cannot be emitted. It
            # still matters when running unconstrained.
            cleaned = strip_stage_directions(sentence)
            if cleaned:
                yield cleaned

    def health_check(self) -> str:
        """Generate one short reply, proving the model and grammar work together.

        Cheap in absolute terms and the only check that actually proves the
        pairing: a grammar can parse fine and still be unsatisfiable by a
        given model's tokenizer, which shows up only when generating.
        """
        probe = [Turn("visitor", "Say one short sentence.")]
        text = " ".join(self.respond(probe))
        if not text.strip():
            raise BrainUnavailable(
                "The model generated nothing. If a grammar is loaded, it may be "
                "unsatisfiable — try again with the grammar disabled."
            )
        constrained = "constrained" if self.grammar else "UNCONSTRAINED"
        return f"{self.model_path.name}, {constrained}: {text.strip()[:60]!r}"
