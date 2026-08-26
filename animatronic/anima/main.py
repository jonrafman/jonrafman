"""The show loop.

    idle ──▶ trigger ──▶ listen ──▶ think ──▶ speak (+ jaw) ──▶ listen ...
      ▲                                                            │
      └──────────────── visitor stops answering ◀──────────────────┘

Run it:

    python -m anima.main --config config.yaml          # run the show
    python -m anima.main --check                       # verify the setup
    python -m anima.main --say "test one two"          # bench-test voice + jaw
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import islice
from pathlib import Path

from .brain import build_brain
from .brain.prompt import build_system_prompt, load_persona
from .config import Config
from .ears import build_ears
from .idle import build_idle
from .jaw import build_jaw
from .text import limit_sentences
from .trigger import build_trigger
from .types import Conversation
from .voice import build_voice

log = logging.getLogger("anima")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


class Show:
    """Owns the hardware, the brain, and the loop that runs the installation."""

    def __init__(self, config: Config) -> None:
        self.config = config

        persona = load_persona(config.resolve(config.character.persona_file))
        system_prompt = build_system_prompt(
            persona, max_sentences=config.character.max_reply_sentences
        )

        self.brain = build_brain(config.brain, system_prompt)
        self.ears = build_ears(config.ears)
        self.voice = build_voice(config.voice)
        self.jaw = build_jaw(config.jaw)
        self.trigger = build_trigger(config.trigger)
        self.idle = build_idle(config.idle, self.jaw)

    def say(self, text: str) -> None:
        """Speak one sentence and move the mouth with it."""
        text = text.strip()
        if not text:
            return
        log.info("%s: %s", self.config.character.name, text)
        try:
            utterance = self.voice.synthesize(text)
        except Exception as exc:
            log.error("Synthesis failed for %r: %s", text, exc)
            return
        try:
            self.jaw.perform(utterance)
        except Exception as exc:
            log.error("Playback failed: %s", exc)

    def converse(self) -> None:
        """One visitor, from arrival to departure."""
        conversation = Conversation()
        cfg = self.config

        if cfg.character.greeting:
            self.say(cfg.character.greeting)

        for _ in range(cfg.conversation.max_turns):
            heard = self.ears.listen(cfg.conversation.listen_timeout)
            if heard is None:
                log.info("Visitor left.")
                return

            log.info("visitor: %s", heard)
            conversation.add("visitor", heard)

            spoken = self._reply(conversation)
            if spoken:
                conversation.add("character", " ".join(spoken))

        log.info("Conversation hit its turn limit; resetting.")

    def _reply(self, conversation: Conversation) -> list[str]:
        """Stream a reply, speaking each sentence the moment it is complete."""
        cfg = self.config
        history = conversation.tail(cfg.conversation.max_history_turns)
        spoken: list[str] = []

        replies = self.brain.respond(history)
        try:
            for sentence in islice(replies, cfg.character.max_reply_sentences):
                # Second line of defence: the prompt asks for brevity, this
                # enforces it even when the model ignores the request.
                sentence = limit_sentences(sentence, cfg.character.max_reply_sentences)
                self.say(sentence)
                spoken.append(sentence)
        except Exception as exc:
            log.error("Brain failed: %s", exc)
        finally:
            # islice can abandon the generator mid-stream; close it explicitly
            # so the underlying HTTP connection is not left hanging open.
            close = getattr(replies, "close", None)
            if callable(close):
                close()

        return spoken

    def run(self) -> None:
        """Wait for visitors forever."""
        log.info(
            "%s is awake. brain=%s voice=%s ears=%s trigger=%s jaw=%s",
            self.config.character.name,
            self.config.brain.backend,
            self.config.voice.backend,
            self.config.ears.backend,
            self.config.trigger.backend,
            "servo" if self.config.jaw.enabled else "off",
        )
        while True:
            self.idle.start()
            try:
                self.trigger.wait()
            except (KeyboardInterrupt, EOFError):
                return
            finally:
                self.idle.stop()

            try:
                self.converse()
            except (KeyboardInterrupt, EOFError):
                return
            except Exception:
                # One bad conversation must never close the exhibition.
                log.exception("Conversation failed; resetting for the next visitor.")

    def close(self) -> None:
        self.idle.stop()
        for component in (self.jaw,):
            closer = getattr(component, "close", None)
            if callable(closer):
                closer()


def check(config: Config) -> int:
    """Verify the setup without running the show. Returns an exit code."""
    problems: list[str] = []

    try:
        persona = load_persona(config.resolve(config.character.persona_file))
        print(f"  persona   ok ({len(persona.split())} words)")
    except Exception as exc:
        problems.append(f"persona: {exc}")
        print(f"  persona   FAILED: {exc}")
        persona = ""

    if persona:
        prompt = build_system_prompt(persona, config.character.max_reply_sentences)
        try:
            brain = build_brain(config.brain, prompt)
            target = getattr(brain, "primary", brain)
            health = getattr(target, "health_check", None)
            if callable(health):
                print(f"  brain     ok ({config.brain.backend}: {health()})")
            else:
                print(f"  brain     ok ({config.brain.backend})")
        except Exception as exc:
            problems.append(f"brain: {exc}")
            print(f"  brain     FAILED: {exc}")

    try:
        voice = build_voice(config.voice)
        utterance = voice.synthesize("Testing, one, two.")
        print(
            f"  voice     ok ({type(voice).__name__}, "
            f"{utterance.duration:.1f}s @ {utterance.sample_rate} Hz)"
        )
    except Exception as exc:
        problems.append(f"voice: {exc}")
        print(f"  voice     FAILED: {exc}")

    for name, builder, section in (
        ("ears", build_ears, config.ears),
        ("trigger", build_trigger, config.trigger),
    ):
        try:
            built = builder(section)
            print(f"  {name:9s} ok ({type(built).__name__})")
        except Exception as exc:
            problems.append(f"{name}: {exc}")
            print(f"  {name:9s} FAILED: {exc}")

    try:
        jaw = build_jaw(config.jaw)
        print(f"  jaw       ok ({type(jaw).__name__})")
    except Exception as exc:
        problems.append(f"jaw: {exc}")
        print(f"  jaw       FAILED: {exc}")

    if problems:
        print(f"\n{len(problems)} problem(s) found.")
        return 1
    print("\nAll good.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anima", description="Run a conversational animatronic figure."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml")
    parser.add_argument("--check", action="store_true", help="verify setup and exit")
    parser.add_argument("--say", metavar="TEXT", help="speak one line and exit")
    parser.add_argument("--once", action="store_true", help="run one conversation and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = Config.load(args.config)
    except Exception as exc:
        print(f"Could not load config: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print(f"Checking {args.config}\n")
        return check(config)

    show = Show(config)
    try:
        if args.say:
            show.say(args.say)
        elif args.once:
            show.converse()
        else:
            show.run()
    except KeyboardInterrupt:
        print()
    finally:
        show.close()
        log.info("Figure at rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
