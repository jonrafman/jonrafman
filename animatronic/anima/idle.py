"""Keeping the figure alive between conversations.

A puppet that is perfectly still until spoken to reads as a prop. A puppet that
moves very slightly, all the time, reads as something waiting. This is the
cheapest large improvement available in the whole project: a few degrees of
jaw motion on a slow cycle, plus the occasional swallow.

Keep it subtle. If a visitor can consciously see the idle loop, it is too big.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time

from .types import Jaw

log = logging.getLogger(__name__)


class IdleMotion:
    """Background breathing and swallowing on the jaw servo.

    Runs on its own thread and must be stopped before the figure speaks --
    ``stop()`` blocks until the thread has actually exited, so the idle loop
    can never fight the lip-sync for control of the servo.

    Args:
        breath_amplitude: peak jaw opening during a breath, in [0, 1]. A few
            percent is plenty.
        breath_period: seconds per breath cycle. Slow reads as calm or old;
            fast reads as anxious.
        swallow_every: average seconds between swallows. Set to 0 to disable.
        update_hz: servo update rate.
    """

    def __init__(
        self,
        jaw: Jaw,
        *,
        breath_amplitude: float = 0.05,
        breath_period: float = 5.0,
        swallow_every: float = 25.0,
        update_hz: float = 25.0,
        seed: int | None = None,
    ) -> None:
        self.jaw = jaw
        self.breath_amplitude = breath_amplitude
        self.breath_period = max(0.5, breath_period)
        self.swallow_every = swallow_every
        self.interval = 1.0 / update_hz
        self._random = random.Random(seed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="idle-motion", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop and wait. Blocking is deliberate -- see the class docstring."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            log.warning("Idle thread did not stop cleanly.")
        self._thread = None
        try:
            self.jaw.rest()
        except Exception as exc:
            log.debug("Could not rest jaw after idle: %s", exc)

    def _next_swallow(self) -> float:
        if self.swallow_every <= 0:
            return math.inf
        # Jittered so the rhythm never becomes predictable.
        return time.monotonic() + self._random.uniform(
            self.swallow_every * 0.6, self.swallow_every * 1.4
        )

    def _run(self) -> None:
        started = time.monotonic()
        swallow_at = self._next_swallow()

        while not self._stop.is_set():
            now = time.monotonic()

            if now >= swallow_at:
                self._swallow()
                swallow_at = self._next_swallow()
                continue

            phase = (now - started) / self.breath_period
            # Offset sine so the jaw rests closed and only ever opens.
            opening = self.breath_amplitude * (0.5 + 0.5 * math.sin(2 * math.pi * phase))
            self._set(opening)
            self._stop.wait(self.interval)

        self._set(0.0)

    def _swallow(self) -> None:
        """A quick open-and-close. Reads as a living body doing something."""
        for opening in (0.18, 0.22, 0.1, 0.0):
            if self._stop.is_set():
                return
            self._set(opening)
            self._stop.wait(0.09)

    def _set(self, opening: float) -> None:
        try:
            self.jaw.set_opening(opening)
        except Exception as exc:
            log.debug("Idle jaw write failed: %s", exc)


class NullIdle:
    """Idle motion disabled."""

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


def build_idle(config, jaw: Jaw):
    if not config.enabled:
        return NullIdle()
    return IdleMotion(
        jaw,
        breath_amplitude=config.breath_amplitude,
        breath_period=config.breath_period,
        swallow_every=config.swallow_every,
    )
