"""Text to audio -- where the character's voice actually lives.

Four backends, worst to best:

  silent      no audio at all, correct duration. Develop the jaw and the loop
              on a machine with no sound card.
  espeak      espeak-ng. Free, offline, instant, and frankly robotic. Do not
              dismiss it on those grounds: for a figure that is supposed to be
              a not-quite-person, a degraded voice is often the stronger
              artistic choice than a smooth one.
  piper       good neural speech, fully offline, fast enough for a Pi. The
              sensible default for an installation that must not depend on
              wifi.
  elevenlabs  uncanny and cloneable. Needs network and costs money per
              utterance. Use when the specific voice carries the piece.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from .types import Utterance

log = logging.getLogger(__name__)


def _wav_to_utterance(path: str | Path, text: str) -> Utterance:
    """Read a WAV into mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")

    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if dtype is np.uint8:  # 8-bit WAV is unsigned, centred on 128
        samples = (samples - 128.0) / 128.0
    else:
        samples /= float(np.iinfo(dtype).max)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return Utterance(samples=samples, sample_rate=rate, text=text)


class SilentVoice:
    """Near-silent audio with a realistic speech shape. For development.

    Not literally zeros. It emits very quiet noise modulated at roughly the
    syllable rate, which means the jaw envelope still has something to chew on
    -- so you can build, calibrate, and tune the whole mouth mechanism before
    you have a TTS engine working, or on a machine with no sound card at all.
    Timing is estimated from word count so the loop paces realistically.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        words_per_minute: float = 150.0,
        amplitude: float = 0.05,
        syllable_hz: float = 4.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.words_per_minute = words_per_minute
        self.amplitude = amplitude
        self.syllable_hz = syllable_hz

    def synthesize(self, text: str) -> Utterance:
        words = max(1, len(text.split()))
        duration = words / (self.words_per_minute / 60.0)
        count = max(1, int(duration * self.sample_rate))

        t = np.arange(count, dtype=np.float32) / self.sample_rate
        # Rectified sine at syllable rate, so the envelope rises and falls the
        # way speech does rather than sitting at a constant level.
        syllables = np.abs(np.sin(2 * np.pi * self.syllable_hz * t))
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        noise = rng.standard_normal(count).astype(np.float32)

        return Utterance(
            samples=(noise * syllables * self.amplitude).astype(np.float32),
            sample_rate=self.sample_rate,
            text=text,
        )


class EspeakVoice:
    """espeak-ng. Always available, always instant, unmistakably synthetic."""

    def __init__(
        self,
        voice: str = "en",
        speed: int = 150,
        pitch: int = 50,
        binary: str | None = None,
    ) -> None:
        self.binary = binary or shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.binary:
            raise RuntimeError(
                "espeak-ng not found. Install it with: sudo apt install espeak-ng"
            )
        self.voice = voice
        self.speed = speed
        self.pitch = pitch

    def synthesize(self, text: str) -> Utterance:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out = tmp.name
        try:
            subprocess.run(
                [
                    self.binary,
                    "-v", self.voice,
                    "-s", str(self.speed),
                    "-p", str(self.pitch),
                    "-w", out,
                    text,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return _wav_to_utterance(out, text)
        finally:
            os.unlink(out)


class PiperVoice:
    """Piper neural TTS. Offline, good, fast enough for a Pi 4/5.

    Args:
        model_path: path to a ``.onnx`` voice. Download voices from the Piper
            releases; the matching ``.onnx.json`` must sit beside it.
        length_scale: >1.0 slows the delivery. Slower usually reads as older,
            heavier, more deliberate -- often what a figure like this wants.
        noise_scale / noise_w: raise for a less even, more unstable voice.
    """

    def __init__(
        self,
        model_path: str,
        *,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        binary: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise RuntimeError(f"Piper voice model not found: {self.model_path}")
        self.binary = binary or shutil.which("piper")
        if not self.binary:
            raise RuntimeError(
                "piper not found. Install it with: pip install piper-tts"
            )
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w

    def synthesize(self, text: str) -> Utterance:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out = tmp.name
        try:
            subprocess.run(
                [
                    self.binary,
                    "--model", str(self.model_path),
                    "--output_file", out,
                    "--length_scale", str(self.length_scale),
                    "--noise_scale", str(self.noise_scale),
                    "--noise_w", str(self.noise_w),
                ],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=60,
            )
            return _wav_to_utterance(out, text)
        finally:
            os.unlink(out)


class ElevenLabsVoice:
    """Cloud TTS with voice cloning. Best quality, needs network, costs money."""

    def __init__(
        self,
        voice_id: str,
        *,
        api_key: str | None = None,
        model_id: str = "eleven_turbo_v2_5",
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ElevenLabs needs an API key. Set ELEVENLABS_API_KEY or put it "
                "in config under voice.elevenlabs.api_key."
            )
        self.voice_id = voice_id
        self.model_id = model_id
        self.timeout = timeout

    def synthesize(self, text: str) -> Utterance:
        import requests

        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
            headers={"xi-api-key": self.api_key, "accept": "audio/wav"},
            json={
                "text": text,
                "model_id": self.model_id,
                "output_format": "pcm_22050",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(response.content)
            out = tmp.name
        try:
            return _wav_to_utterance(out, text)
        finally:
            os.unlink(out)


def build_voice(config):
    """Construct the configured voice, degrading gracefully if it is missing.

    A voice backend that fails to initialise should not stop the show: fall
    back to espeak, then to silence, and log loudly. An installation that opens
    with a bad voice is better than one that does not open.
    """
    backend = config.backend.lower()

    try:
        if backend == "silent":
            return SilentVoice()
        if backend == "espeak":
            return EspeakVoice(
                voice=config.espeak_voice,
                speed=config.espeak_speed,
                pitch=config.espeak_pitch,
            )
        if backend == "piper":
            return PiperVoice(
                model_path=config.piper_model,
                length_scale=config.piper_length_scale,
            )
        if backend == "elevenlabs":
            return ElevenLabsVoice(
                voice_id=config.elevenlabs_voice_id,
                api_key=config.elevenlabs_api_key or None,
            )
        raise ValueError(
            f"Unknown voice backend '{config.backend}'. "
            "Expected one of: silent, espeak, piper, elevenlabs."
        )
    except Exception as exc:
        log.error("Voice backend '%s' failed to start: %s", backend, exc)
        try:
            log.warning("Falling back to espeak.")
            return EspeakVoice()
        except Exception:
            log.warning("Falling back to silence. The puppet will mime.")
            return SilentVoice()
