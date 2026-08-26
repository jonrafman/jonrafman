"""Making the mouth move in time with the voice.

The technique is the old singing-fish one: take the amplitude envelope of the
speech and drive the jaw servo with it. It is not phoneme-accurate and it does
not need to be -- at conversational distance, a jaw that opens on the loud parts
and closes on the quiet parts reads as speech.

Three details separate a convincing jaw from a twitching one:

1. **Asymmetric smoothing.** Real jaws snap open and fall closed more slowly.
   Fast attack, slow release. Symmetric smoothing looks mechanical and makes
   the servo buzz.
2. **Per-utterance normalisation, with a floor.** Normalising to the loudest
   moment of *this* sentence keeps a quiet line from barely moving. The floor
   stops a near-silent line from being amplified into shouting.
3. **A noise gate.** Without one the jaw trembles constantly on room tone,
   which is worse than not moving at all.

Sync strategy: start playback, note the wall clock, then drive the servo from
elapsed time rather than by counting frames. Servo writes are slow and jittery;
tying them to the clock means a slow write loses one frame instead of
accumulating drift until the mouth is moving a second behind the voice.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .types import Utterance

log = logging.getLogger(__name__)


def speech_envelope(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop_ms: float = 20.0,
    attack: float = 0.6,
    release: float = 0.18,
    gate: float = 0.06,
    floor: float = 0.02,
) -> np.ndarray:
    """Amplitude envelope in [0, 1], one value per ``hop_ms``.

    Args:
        hop_ms: update interval. 20 ms gives 50 Hz, which matches the update
            rate of a standard hobby servo. Going faster gains nothing and
            makes the servo chatter.
        attack: smoothing when opening (0-1, higher is snappier).
        release: smoothing when closing. Keep well below ``attack``.
        gate: envelope values below this become 0, killing room-tone tremble.
        floor: minimum peak used for normalisation, as a fraction of full
            scale. Stops a whispered line being normalised up to a shout.
    """
    samples = np.asarray(samples, dtype=np.float32)
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    frame_count = len(samples) // hop
    if frame_count == 0:
        return np.zeros(1, dtype=np.float32)

    frames = samples[: frame_count * hop].reshape(frame_count, hop)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))

    # Normalise against a high percentile, not the max: a single click or plosive
    # should not define the scale for the whole sentence.
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


class NullJaw:
    """No servo. Waits out the utterance so the loop keeps realistic timing."""

    def __init__(self, *, log_envelope: bool = False) -> None:
        self.log_envelope = log_envelope
        self._opening = 0.0

    def perform(self, utterance: Utterance) -> None:
        if self.log_envelope:
            envelope = speech_envelope(
                np.asarray(utterance.samples), utterance.sample_rate
            )
            bars = "".join(" ▁▂▃▄▅▆▇█"[int(v * 8)] for v in envelope[:120])
            log.info("jaw: %s", bars)
        time.sleep(utterance.duration)

    def rest(self) -> None:
        self._opening = 0.0

    def set_opening(self, amount: float) -> None:
        self._opening = float(np.clip(amount, 0.0, 1.0))


class ServoJaw:
    """Drives a jaw servo in sync with played audio.

    Args:
        pin: BCM pin number for the servo signal wire.
        closed_angle / open_angle: the mechanical limits of *your* jaw. Find
            these with ``python -m anima.calibrate`` before you connect the
            linkage -- driving a servo past what the jaw allows strips gears
            and snaps printed parts. These are hard clamps, not suggestions.
        max_opening: extra safety ceiling in [0, 1]. Set below 1.0 to keep the
            jaw from ever reaching its full travel.
        use_pigpio: hardware-timed PWM. Strongly recommended -- the default
            software PWM on a Pi jitters audibly and makes the servo hum
            through the whole figure.
    """

    def __init__(
        self,
        pin: int,
        *,
        closed_angle: float = 0.0,
        open_angle: float = 35.0,
        max_opening: float = 1.0,
        use_pigpio: bool = True,
        hop_ms: float = 20.0,
    ) -> None:
        self.closed_angle = closed_angle
        self.open_angle = open_angle
        self.max_opening = float(np.clip(max_opening, 0.0, 1.0))
        self.hop_ms = hop_ms
        self._servo = self._make_servo(pin, use_pigpio)
        self.rest()

    def _make_servo(self, pin: int, use_pigpio: bool):
        from gpiozero import AngularServo

        factory = None
        if use_pigpio:
            try:
                from gpiozero.pins.pigpio import PiGPIOFactory

                factory = PiGPIOFactory()
            except Exception as exc:
                log.warning(
                    "pigpio unavailable (%s); falling back to software PWM. "
                    "The servo will jitter. Start the daemon with: "
                    "sudo systemctl enable --now pigpiod",
                    exc,
                )

        low, high = sorted((self.closed_angle, self.open_angle))
        return AngularServo(
            pin,
            min_angle=low,
            max_angle=high,
            min_pulse_width=0.5 / 1000,
            max_pulse_width=2.5 / 1000,
            pin_factory=factory,
        )

    def set_opening(self, amount: float) -> None:
        amount = float(np.clip(amount, 0.0, self.max_opening))
        self._servo.angle = self.closed_angle + amount * (
            self.open_angle - self.closed_angle
        )

    def rest(self) -> None:
        self.set_opening(0.0)

    def perform(self, utterance: Utterance) -> None:
        samples = np.asarray(utterance.samples, dtype=np.float32)
        envelope = speech_envelope(
            samples, utterance.sample_rate, hop_ms=self.hop_ms
        )
        hop_seconds = self.hop_ms / 1000.0

        playback = self._start_playback(samples, utterance.sample_rate)
        started = time.perf_counter()

        try:
            last_index = -1
            duration = len(envelope) * hop_seconds
            while True:
                elapsed = time.perf_counter() - started
                if elapsed >= duration:
                    break
                # Index from the clock, not a counter: a slow servo write drops
                # a frame rather than pushing the mouth out of sync.
                index = min(int(elapsed / hop_seconds), len(envelope) - 1)
                if index != last_index:
                    self.set_opening(float(envelope[index]))
                    last_index = index
                time.sleep(hop_seconds / 4.0)
        finally:
            self.rest()
            if playback is not None:
                playback()

    def _start_playback(self, samples: np.ndarray, sample_rate: int):
        """Begin non-blocking playback. Returns a wait-for-finish callable."""
        try:
            import sounddevice
        except Exception as exc:
            log.warning("No audio output (%s); jaw will move silently.", exc)
            return None
        try:
            sounddevice.play(samples, sample_rate)
            return sounddevice.wait
        except Exception as exc:
            log.warning("Playback failed (%s); jaw will move silently.", exc)
            return None

    def close(self) -> None:
        try:
            self.rest()
            self._servo.close()
        except Exception:
            pass


class AudioOnlyJaw:
    """Plays audio with no servo attached. For bench-testing the voice."""

    def perform(self, utterance: Utterance) -> None:
        samples = np.asarray(utterance.samples, dtype=np.float32)
        try:
            import sounddevice

            sounddevice.play(samples, utterance.sample_rate)
            sounddevice.wait()
        except Exception as exc:
            log.warning("Playback failed (%s); waiting out the utterance.", exc)
            time.sleep(utterance.duration)

    def rest(self) -> None:
        pass

    def set_opening(self, amount: float) -> None:
        pass


def build_jaw(config):
    """Construct the configured jaw, degrading to audio-only off a Pi."""
    if not config.enabled:
        return AudioOnlyJaw() if config.play_audio else NullJaw(
            log_envelope=config.log_envelope
        )

    try:
        return ServoJaw(
            pin=config.servo_pin,
            closed_angle=config.closed_angle,
            open_angle=config.open_angle,
            max_opening=config.max_opening,
            use_pigpio=config.use_pigpio,
        )
    except Exception as exc:
        log.error(
            "Servo jaw unavailable (%s). Running audio-only -- expected if you "
            "are not on a Raspberry Pi.",
            exc,
        )
        return AudioOnlyJaw()
