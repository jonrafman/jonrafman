"""The states of an intelligence, expressed only in light.

With no face and no motion, the light has to do the work a body would do — it
has to show that the thing is awake, that it noticed you, that it is
considering, and that it is speaking. Four states:

    DORMANT    nobody is here. A slow, deep, almost-imperceptible breath.
               Never fully dark: a light that reaches zero reads as switched
               off, and a piece that reads as switched off is not a piece.

    ATTENDING  it has noticed. Brighter, steadier, with a faint tremor so it
               does not look like a bulb someone left on.

    THINKING   the model is generating. This state is the reason the whole
               design works: the seconds a language model spends thinking are
               the biggest technical liability in a talking sculpture, and
               giving them their own visible behaviour converts that dead air
               into the most legible thing the piece does. The visitor does not
               experience latency, they experience deliberation.

    SPEAKING   driven by the amplitude envelope of the voice, modulated
               between a baseline and full rather than between off and full.

Transitions ease rather than jump — the animator eases the actual level toward
whatever the current state wants, so switching states is a fade, always.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time

import numpy as np

from .light import play_audio, speech_envelope
from .types import Utterance

log = logging.getLogger(__name__)

DORMANT = "dormant"
ATTENDING = "attending"
THINKING = "thinking"


class Presence:
    """Animates a light through the states of attention.

    Args:
        light: anything with ``set_level`` / ``rest``.
        dormant_level / dormant_swing / dormant_period: the resting breath.
        attending_level: brightness once a visitor is detected.
        thinking_level / thinking_swing / thinking_rate: the deliberation
            pulse. Faster and wider than the breath, so the change of state is
            unmistakable without being theatrical.
        speaking_baseline: the floor the voice modulates upward from. Higher
            reads as a steady presence that brightens with speech; near zero
            reads as a VU meter, which is the thing to avoid.
        ease: how quickly the actual level chases the target, per update.
            Lower is more languid.
        update_hz: animation rate. Light has no inertia, so this can be high.
    """

    def __init__(
        self,
        light,
        *,
        dormant_level: float = 0.08,
        dormant_swing: float = 0.05,
        dormant_period: float = 9.0,
        attending_level: float = 0.35,
        thinking_level: float = 0.5,
        thinking_swing: float = 0.22,
        thinking_rate: float = 1.6,
        speaking_baseline: float = 0.28,
        ease: float = 0.12,
        update_hz: float = 60.0,
        hop_ms: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.light = light
        self.dormant_level = dormant_level
        self.dormant_swing = dormant_swing
        self.dormant_period = max(0.5, dormant_period)
        self.attending_level = attending_level
        self.thinking_level = thinking_level
        self.thinking_swing = thinking_swing
        self.thinking_rate = max(0.1, thinking_rate)
        self.speaking_baseline = float(np.clip(speaking_baseline, 0.0, 0.95))
        self.ease = float(np.clip(ease, 0.01, 1.0))
        self.interval = 1.0 / update_hz
        self.hop_ms = hop_ms

        self._random = random.Random(seed)
        self._state = DORMANT
        self._level = 0.0
        self._speaking = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._animate, name="presence", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            log.warning("Presence thread did not stop cleanly.")
        self._thread = None
        try:
            self.light.rest()
        except Exception as exc:
            log.debug("Could not rest light: %s", exc)

    def close(self) -> None:
        self.stop()
        closer = getattr(self.light, "close", None)
        if callable(closer):
            closer()

    # -- states ---------------------------------------------------------------

    def enter(self, state: str) -> None:
        """Switch ambient state. The transition eases; it never jumps."""
        if state not in (DORMANT, ATTENDING, THINKING):
            raise ValueError(f"Unknown presence state: {state}")
        self._state = state

    @property
    def state(self) -> str:
        return self._state

    def _target(self, now: float) -> float:
        if self._state == DORMANT:
            phase = now / self.dormant_period
            return self.dormant_level + self.dormant_swing * (
                0.5 + 0.5 * math.sin(2 * math.pi * phase)
            )

        if self._state == ATTENDING:
            # A small irregular tremor: enough that it is not a dead bulb,
            # small enough that nobody consciously sees it move.
            return self.attending_level + self._random.uniform(-0.012, 0.012)

        # THINKING
        phase = now * self.thinking_rate
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * phase)
        jitter = self._random.uniform(-0.05, 0.05)
        return self.thinking_level + self.thinking_swing * pulse + jitter

    def _animate(self) -> None:
        started = time.monotonic()
        while not self._stop.is_set():
            if self._speaking.is_set():
                # speak() owns the light for its duration.
                self._stop.wait(self.interval)
                continue

            target = self._target(time.monotonic() - started)
            self._level += self.ease * (target - self._level)
            self._write(self._level)
            self._stop.wait(self.interval)

        self._write(0.0)

    def _write(self, level: float) -> None:
        try:
            self.light.set_level(float(np.clip(level, 0.0, 1.0)))
        except Exception as exc:
            log.debug("Light write failed: %s", exc)

    # -- speaking -------------------------------------------------------------

    def speak(self, utterance: Utterance) -> None:
        """Play an utterance, driving the light from its amplitude envelope.

        Sync comes from the wall clock rather than a frame counter: a slow
        write to the hardware then costs one dropped frame instead of
        accumulating into the light lagging behind the voice.
        """
        samples = np.asarray(utterance.samples, dtype=np.float32)
        envelope = speech_envelope(samples, utterance.sample_rate, hop_ms=self.hop_ms)
        hop_seconds = self.hop_ms / 1000.0
        span = 1.0 - self.speaking_baseline

        self._speaking.set()
        playback = play_audio(samples, utterance.sample_rate)
        started = time.perf_counter()

        try:
            duration = len(envelope) * hop_seconds
            while True:
                elapsed = time.perf_counter() - started
                if elapsed >= duration:
                    break
                index = min(int(elapsed / hop_seconds), len(envelope) - 1)
                level = self.speaking_baseline + span * float(envelope[index])
                self._write(level)
                self._level = level
                time.sleep(hop_seconds / 4.0)
        finally:
            if playback is not None:
                try:
                    playback()
                except Exception as exc:
                    log.debug("Waiting on playback failed: %s", exc)
            self._speaking.clear()


class NullPresence:
    """Presence disabled. The light stays dark and nothing animates."""

    state = DORMANT

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def enter(self, state: str) -> None: ...

    def speak(self, utterance: Utterance) -> None:
        playback = play_audio(
            np.asarray(utterance.samples, dtype=np.float32), utterance.sample_rate
        )
        if playback is not None:
            playback()
        else:
            time.sleep(utterance.duration)


def build_presence(config, light):
    if not config.enabled:
        return NullPresence()
    return Presence(
        light,
        dormant_level=config.dormant_level,
        dormant_swing=config.dormant_swing,
        dormant_period=config.dormant_period,
        attending_level=config.attending_level,
        thinking_level=config.thinking_level,
        thinking_swing=config.thinking_swing,
        thinking_rate=config.thinking_rate,
        speaking_baseline=config.speaking_baseline,
        ease=config.ease,
    )
