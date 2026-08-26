"""Deciding that someone is there.

Wake words are the obvious choice and usually the wrong one for a gallery: they
misfire on ambient conversation, they fail in a loud room, and they force the
visitor to know a magic phrase before anything happens.

A physical trigger is better on all three counts. A button the visitor presses,
a handset they lift, a chair they sit in, a motion sensor at the threshold --
each is reliable, and each is also a piece of staging that tells the visitor
what to do without a sign explaining it.
"""

from __future__ import annotations

import logging
import sys
import time

log = logging.getLogger(__name__)


class KeyboardTrigger:
    """Press enter. For development."""

    def __init__(self, message: str = "\n[press enter to approach the piece]") -> None:
        self.message = message

    def wait(self) -> None:
        try:
            print(self.message)
            sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            raise


class AlwaysTrigger:
    """Never waits -- the piece is always listening.

    Only sensible with a close-talking microphone in a quiet space, or during
    testing. In a gallery this transcribes the room all day.
    """

    def wait(self) -> None:
        return


class ButtonTrigger:
    """A momentary push button on a GPIO pin.

    Wire one side to the pin and the other to ground; the internal pull-up
    handles the rest, so no resistor is needed.
    """

    def __init__(self, pin: int, *, bounce_time: float = 0.1) -> None:
        from gpiozero import Button

        self._button = Button(pin, pull_up=True, bounce_time=bounce_time)

    def wait(self) -> None:
        self._button.wait_for_press()
        self._button.wait_for_release()


class MotionTrigger:
    """A PIR sensor. Fires when someone enters the piece's space.

    ``settle_seconds`` waits after motion before starting, so the piece does
    not begin talking to a visitor's back as they walk past. ``cooldown``
    prevents one lingering person from re-triggering forever.
    """

    def __init__(
        self,
        pin: int,
        *,
        settle_seconds: float = 0.8,
        cooldown: float = 5.0,
    ) -> None:
        from gpiozero import MotionSensor

        self._sensor = MotionSensor(pin)
        self.settle_seconds = settle_seconds
        self.cooldown = cooldown
        self._last_fired = 0.0

    def wait(self) -> None:
        while True:
            self._sensor.wait_for_motion()
            now = time.monotonic()
            if now - self._last_fired < self.cooldown:
                self._sensor.wait_for_no_motion()
                continue
            time.sleep(self.settle_seconds)
            self._last_fired = time.monotonic()
            return


def build_trigger(config):
    """Construct the configured trigger, degrading to keyboard off a Pi."""
    backend = config.backend.lower()
    if backend == "keyboard":
        return KeyboardTrigger()
    if backend == "always":
        return AlwaysTrigger()

    try:
        if backend == "button":
            return ButtonTrigger(config.pin)
        if backend == "motion":
            return MotionTrigger(
                config.pin,
                settle_seconds=config.settle_seconds,
                cooldown=config.cooldown,
            )
    except Exception as exc:
        log.error(
            "GPIO trigger '%s' unavailable (%s). Falling back to keyboard.",
            backend, exc,
        )
        return KeyboardTrigger()

    raise ValueError(
        f"Unknown trigger backend '{config.backend}'. "
        "Expected one of: keyboard, always, button, motion."
    )
