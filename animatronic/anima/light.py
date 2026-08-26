"""Light as the body.

The sculpture has no moving parts. Everything a visitor sees it *do* is done
with light, which means the light is not decoration — it is the whole of the
performance, and it has to carry states that a face would otherwise carry:
dormant, attending, thinking, speaking.

Two things matter more here than they did with a servo:

1. **Gamma.** Human brightness perception is roughly a power law, so a linear
   PWM duty cycle looks wrong — it slams up at the bottom of the range and
   barely changes at the top. Every level written to hardware goes through
   ``gamma_correct`` first. Without it a smooth fade looks like a stutter.

2. **Never reaching zero.** A light that goes fully dark reads as switched off,
   i.e. broken. A presence keeps a floor under itself. The speaking animation
   modulates *between a baseline and full*, rather than between off and full —
   the difference between an intelligence speaking and a VU meter bouncing.

Update rate is no longer limited by a servo's 50 Hz, so the envelope is
computed finer and the light can respond to detail a jaw physically could not.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from .types import Utterance

log = logging.getLogger(__name__)

# 2.2 matches sRGB and most LEDs closely enough. Raise toward 2.8 if low-end
# fades still look steppy on your particular fixture.
DEFAULT_GAMMA = 2.2


def gamma_correct(level: float, gamma: float = DEFAULT_GAMMA) -> float:
    """Map a perceptual level in [0,1] to a PWM duty cycle in [0,1]."""
    return float(np.clip(level, 0.0, 1.0)) ** gamma


def speech_envelope(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop_ms: float = 10.0,
    attack: float = 0.7,
    release: float = 0.12,
    gate: float = 0.05,
    floor: float = 0.02,
) -> np.ndarray:
    """Amplitude envelope in [0, 1], one value per ``hop_ms``.

    Args:
        hop_ms: update interval. 10 ms (100 Hz) — twice what a servo could
            follow, and light has no inertia to smear the difference.
        attack: smoothing when brightening. Fast, so consonants register.
        release: smoothing when dimming. Slow, giving an afterglow that reads
            as warmth rather than as strobing.
        gate: values below this collapse to 0, killing room-tone flicker.
        floor: minimum peak used for normalisation, so a quiet line is not
            amplified into a shout.
    """
    samples = np.asarray(samples, dtype=np.float32)
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    frame_count = len(samples) // hop
    if frame_count == 0:
        return np.zeros(1, dtype=np.float32)

    frames = samples[: frame_count * hop].reshape(frame_count, hop)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))

    # Normalise against a high percentile rather than the max: one plosive
    # should not set the scale for a whole sentence.
    peak = max(float(np.percentile(rms, 95)), floor)
    envelope = np.clip(rms / peak, 0.0, 1.0)
    envelope[envelope < gate] = 0.0

    smoothed = np.empty_like(envelope)
    current = 0.0
    for i, value in enumerate(envelope):
        coefficient = attack if value > current else release
        current += coefficient * (value - current)
        smoothed[i] = current

    return smoothed


class ConsoleLight:
    """No hardware. Draws the light level in the terminal.

    Genuinely useful, not just a stub: the state machine, the timing, and the
    envelope shaping are all visible here, so the entire performance can be
    tuned before a single LED is wired.
    """

    def __init__(self, *, draw: bool = True, width: int = 40) -> None:
        # The bar redraws in place with a carriage return, which is right in a
        # terminal and megabytes of noise in a pipe or a log file.
        self.draw = draw and sys.stdout.isatty()
        self.width = width
        self.level = 0.0
        self._last_drawn = -1.0

    def set_level(self, level: float) -> None:
        self.level = float(np.clip(level, 0.0, 1.0))
        if not self.draw:
            return
        # Only redraw on a visible change; otherwise a 100 Hz loop floods the
        # terminal and the log becomes unreadable.
        if abs(self.level - self._last_drawn) < 0.02:
            return
        self._last_drawn = self.level
        filled = int(self.level * self.width)
        bar = "█" * filled + "·" * (self.width - filled)
        print(f"\r  {bar} {self.level:4.0%}", end="", flush=True)

    def rest(self) -> None:
        self.set_level(0.0)
        if self.draw:
            print()

    def close(self) -> None:
        self.rest()


class PWMLight:
    """A single dimmable channel on a GPIO pin.

    Right for one lamp, one LED behind a diffuser, or a MOSFET driving a
    high-power emitter. For anything above a few hundred milliamps the pin
    switches a driver, never the load itself.

    Args:
        pin: BCM pin number.
        frequency: PWM frequency. Keep well above 1 kHz — below that, a camera
            pointed at the piece will show banding, and some people perceive
            flicker directly.
        gamma: perceptual correction exponent.
        max_level: hard ceiling in [0,1], for fixtures brighter than the piece
            wants at full travel.
    """

    def __init__(
        self,
        pin: int,
        *,
        frequency: int = 2000,
        gamma: float = DEFAULT_GAMMA,
        max_level: float = 1.0,
    ) -> None:
        from gpiozero import PWMLED

        factory = None
        try:
            from gpiozero.pins.pigpio import PiGPIOFactory

            factory = PiGPIOFactory()
        except Exception as exc:
            log.warning(
                "pigpio unavailable (%s); using software PWM. Expect visible "
                "flicker on fades. Fix with: sudo systemctl enable --now pigpiod",
                exc,
            )

        self._led = PWMLED(pin, frequency=frequency, pin_factory=factory)
        self.gamma = gamma
        self.max_level = float(np.clip(max_level, 0.0, 1.0))

    def set_level(self, level: float) -> None:
        level = float(np.clip(level, 0.0, self.max_level))
        self._led.value = gamma_correct(level, self.gamma)

    def rest(self) -> None:
        self.set_level(0.0)

    def close(self) -> None:
        try:
            self.rest()
            self._led.close()
        except Exception:
            pass


class AddressableLight:
    """A ring or strip of addressable LEDs, driven as one mass.

    Every pixel shows the same value: this is a single presence, not a pattern.
    A ring behind frosted acrylic is the classic single-eye look.

    On chipsets: SK9822/APA102 dim far more smoothly than WS2812 because their
    internal PWM runs much faster, and they use a clocked SPI protocol that
    does not fight the CPU for precise timing. If the piece does slow fades —
    and it does — that difference is visible.
    """

    def __init__(
        self,
        count: int,
        *,
        pin: str = "D18",
        color: tuple[int, int, int] = (255, 255, 255),
        gamma: float = DEFAULT_GAMMA,
        max_level: float = 1.0,
    ) -> None:
        import board
        import neopixel

        self._pixels = neopixel.NeoPixel(
            getattr(board, pin), count, auto_write=False, brightness=1.0
        )
        self.color = color
        self.gamma = gamma
        self.max_level = float(np.clip(max_level, 0.0, 1.0))

    def set_level(self, level: float) -> None:
        level = float(np.clip(level, 0.0, self.max_level))
        scale = gamma_correct(level, self.gamma)
        self._pixels.fill(tuple(int(c * scale) for c in self.color))
        self._pixels.show()

    def rest(self) -> None:
        self.set_level(0.0)

    def close(self) -> None:
        try:
            self.rest()
            self._pixels.deinit()
        except Exception:
            pass


def play_audio(samples: np.ndarray, sample_rate: int):
    """Start non-blocking playback. Returns a wait-for-finish callable, or None."""
    try:
        import sounddevice
    except Exception as exc:
        log.warning("No audio output (%s); running silent.", exc)
        return None
    try:
        sounddevice.play(samples, sample_rate)
        return sounddevice.wait
    except Exception as exc:
        log.warning("Playback failed (%s); running silent.", exc)
        return None


def build_light(config):
    """Construct the configured light, degrading to console off a Pi."""
    backend = config.backend.lower()

    if backend == "console":
        return ConsoleLight(draw=config.draw)

    try:
        if backend == "pwm":
            return PWMLight(
                config.pin,
                frequency=config.pwm_frequency,
                gamma=config.gamma,
                max_level=config.max_level,
            )
        if backend == "addressable":
            return AddressableLight(
                config.led_count,
                pin=config.addressable_pin,
                color=tuple(config.color),
                gamma=config.gamma,
                max_level=config.max_level,
            )
    except Exception as exc:
        log.error(
            "Light backend '%s' unavailable (%s). Falling back to console — "
            "expected if you are not on the sculpture's own hardware.",
            backend, exc,
        )
        return ConsoleLight(draw=config.draw)

    raise ValueError(
        f"Unknown light backend '{config.backend}'. "
        "Expected one of: console, pwm, addressable."
    )
