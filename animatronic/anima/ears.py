"""Hearing the visitor.

Two backends:

  text     type at the console instead of speaking. This is how you should do
           most of your development -- it removes microphone quality, room
           noise, and transcription errors from the loop while you are still
           working out what the character says.
  whisper  faster-whisper running locally. No network, no per-use cost.

On microphones: this is the part people underestimate. A gallery is loud, and a
cheap omnidirectional mic three feet from a visitor will transcribe the room,
not the person. Get the microphone close -- built into the sculpture's face,
in something the visitor leans toward, or a handset they pick up -- and the
whole project gets easier.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

log = logging.getLogger(__name__)


class TextEars:
    """Reads a typed line. Ignores ``timeout`` -- it waits as long as you do."""

    def __init__(self, prompt: str = "visitor> ") -> None:
        self.prompt = prompt

    def listen(self, timeout: float) -> str | None:
        try:
            sys.stdout.write(self.prompt)
            sys.stdout.flush()
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            return None
        if not line:
            return None
        return line.strip() or None


class WhisperEars:
    """Records until the visitor stops talking, then transcribes locally.

    Args:
        model_size: ``tiny.en`` / ``base.en`` / ``small.en``. On a Pi 5,
            ``base.en`` transcribes a short utterance in roughly a second;
            ``small.en`` is noticeably better and noticeably slower. Start at
            ``base.en``.
        device_index: input device index, or None for the system default. List
            them with ``python -m anima.devices``.
        silence_threshold: RMS below which a chunk counts as silence. Tune this
            in the actual room -- the default is for a quiet space.
        silence_duration: seconds of continuous silence that ends the turn.
        max_duration: hard cap, so one person monologuing does not hang the
            installation.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        *,
        device_index: int | None = None,
        sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_duration: float = 1.2,
        max_duration: float = 20.0,
        language: str = "en",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "faster-whisper is required for the whisper ears. "
                "pip install faster-whisper"
            ) from exc

        log.info("Loading whisper model '%s' (first run downloads it)...", model_size)
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.language = language

    def _record(self, timeout: float) -> np.ndarray | None:
        """Wait for speech, then record until it stops. None if nobody spoke."""
        import sounddevice

        chunk_seconds = 0.05
        chunk_frames = int(self.sample_rate * chunk_seconds)
        collected: list[np.ndarray] = []
        speech_started = False
        silent_for = 0.0
        waited = 0.0

        with sounddevice.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            device=self.device_index,
        ) as stream:
            while True:
                chunk, overflowed = stream.read(chunk_frames)
                if overflowed:
                    log.debug("Input overflow (dropped audio)")
                chunk = chunk[:, 0]
                level = float(np.sqrt(np.mean(np.square(chunk))))

                if level >= self.silence_threshold:
                    speech_started = True
                    silent_for = 0.0
                elif speech_started:
                    silent_for += chunk_seconds

                if speech_started:
                    collected.append(chunk.copy())
                    if silent_for >= self.silence_duration:
                        break
                    if len(collected) * chunk_seconds >= self.max_duration:
                        log.info("Hit max recording duration; cutting off.")
                        break
                else:
                    waited += chunk_seconds
                    if waited >= timeout:
                        return None

        return np.concatenate(collected) if collected else None

    def listen(self, timeout: float) -> str | None:
        try:
            audio = self._record(timeout)
        except Exception as exc:
            log.error("Recording failed: %s", exc)
            time.sleep(0.5)
            return None

        if audio is None or len(audio) < self.sample_rate * 0.3:
            return None  # Too short to be speech; almost certainly a cough.

        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,  # Greedy: the accuracy loss is small, the speedup is not.
            vad_filter=True,
        )
        text = " ".join(segment.text for segment in segments).strip()
        return text or None


def build_ears(config):
    """Construct the configured ears, degrading to typed input on failure."""
    backend = config.backend.lower()
    if backend == "text":
        return TextEars()
    if backend == "whisper":
        try:
            return WhisperEars(
                model_size=config.whisper_model,
                device_index=config.device_index,
                silence_threshold=config.silence_threshold,
                silence_duration=config.silence_duration,
                max_duration=config.max_duration,
            )
        except Exception as exc:
            log.error("Whisper ears unavailable (%s). Falling back to typed input.", exc)
            return TextEars()
    raise ValueError(
        f"Unknown ears backend '{config.backend}'. Expected one of: text, whisper."
    )
