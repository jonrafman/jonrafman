"""Text to audio -- where the character's voice actually lives.

Backends, roughly worst to best:

  silent      near-silent but speech-shaped, so the light still moves.
              Build the whole performance on a machine with no sound card.
  espeak      espeak-ng. Free, offline, instant, and frankly robotic. Do not
              dismiss it on those grounds: for an intelligence that is not
              supposed to be a person, a degraded voice is often the
              stronger artistic choice than a smooth one.
  piper       decent neural speech, offline, fast enough for a Pi. Superseded
              for quality by the two below, but the lightest to deploy.
  kokoro      82M parameters, Apache 2.0, near real-time on Apple Silicon.
              Fifty-odd fixed voices, no cloning. The best quality-per-effort
              option, and the cleanest licence.
  chatterbox  MIT, clones a voice from a few seconds of reference audio, with
              an emotion dial. The strongest offline option for this piece.
  elevenlabs  the cloud benchmark. Needs network and costs per utterance --
              which is exactly what this piece was built to avoid. Kept for
              comparison, so you can hear what you are giving up.

On licences, which matter more than they look: an artwork that is exhibited
or sold is commercial use. XTTS-v2 and F5-TTS both sound excellent and both
carry non-commercial licences, so neither belongs in a gallery piece however
good the demo is. Kokoro (Apache 2.0) and Chatterbox (MIT) are unencumbered.

On archaic spelling: everything here receives text already normalised by
anima.pronounce. A speech model reads Blake's `thro` as "throw" and `shewd`
as "shooed"; see that module for why the fix does not loosen the lexicon.
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
    syllable rate, which means the light envelope still has something to work
    with -- so the whole performance can be built and tuned before a TTS
    engine is running, or on a machine with no sound card at all.
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
            heavier, more deliberate -- often what a piece like this wants.
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


class KokoroVoice:
    """Kokoro-82M. Apache 2.0, tiny, and far better than its size suggests.

    82 million parameters, so it runs near real-time on Apple Silicon and
    acceptably on CPU. Fifty-odd fixed voices; it cannot clone. For a piece
    that wants *a* good voice rather than *one specific* voice, the fixed set
    is not a limitation, and the licence is the cleanest of any option here.

    Voice names are prefixed by accent and gender -- ``bm_`` for British male,
    ``bf_`` British female, ``am_``/``af_`` American. A British voice suits a
    lexicon drawn from Blake.

    NOTE: this binding is written against Kokoro's documented API but has not
    been executed -- no model weights could reach the machine it was written
    on. Expect to adjust the call, not the architecture.
    """

    def __init__(
        self,
        voice: str = "bm_george",
        *,
        lang_code: str = "b",
        speed: float = 0.9,
        sample_rate: int = 24000,
    ) -> None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro is not installed. pip install kokoro soundfile\n"
                "On macOS you may also need: brew install espeak-ng"
            ) from exc

        self._pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> Utterance:
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _, _, audio in self._pipeline(text, voice=self.voice, speed=self.speed)
        ]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for {text!r}")
        return Utterance(
            samples=np.concatenate(chunks),
            sample_rate=self.sample_rate,
            text=text,
        )


class ChatterboxVoice:
    """Chatterbox (Resemble AI). MIT licensed, clones from a few seconds of audio.

    The strongest open option for this piece, and the licence is why: an
    artwork that is exhibited or sold is commercial use, which rules out the
    non-commercial licences on XTTS-v2 and F5-TTS however good they sound.
    MIT does not care what you do with it.

    ``exaggeration`` drives emotional intensity. For an unhurried, level
    presence, keep it low -- the default 0.5 is already more animated than
    this character should be.

    Args:
        reference_audio: a clean 5-20 second WAV of the voice to clone. The
            recording quality sets the ceiling on the result; room tone and
            compression in the reference come through in every line.
        device: "mps" on Apple Silicon, "cuda" on NVIDIA, "cpu" otherwise.

    NOTE: written against Chatterbox's documented API, not executed here.
    """

    def __init__(
        self,
        reference_audio: str = "",
        *,
        device: str = "mps",
        exaggeration: float = 0.3,
        cfg_weight: float = 0.5,
        sample_rate: int = 24000,
    ) -> None:
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise RuntimeError(
                "Chatterbox is not installed. pip install chatterbox-tts"
            ) from exc

        if reference_audio and not Path(reference_audio).is_file():
            raise RuntimeError(f"Reference audio not found: {reference_audio}")

        self._model = ChatterboxTTS.from_pretrained(device=device)
        self.reference_audio = reference_audio
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.sample_rate = getattr(self._model, "sr", sample_rate)

    def synthesize(self, text: str) -> Utterance:
        kwargs = {"exaggeration": self.exaggeration, "cfg_weight": self.cfg_weight}
        if self.reference_audio:
            kwargs["audio_prompt_path"] = self.reference_audio

        wav = self._model.generate(text, **kwargs)
        samples = np.asarray(
            wav.squeeze().detach().cpu().numpy() if hasattr(wav, "detach") else wav,
            dtype=np.float32,
        )
        return Utterance(samples=samples, sample_rate=self.sample_rate, text=text)


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
        if backend == "kokoro":
            return KokoroVoice(
                voice=config.kokoro_voice,
                lang_code=config.kokoro_lang,
                speed=config.kokoro_speed,
            )
        if backend == "chatterbox":
            return ChatterboxVoice(
                reference_audio=config.chatterbox_reference,
                device=config.chatterbox_device,
                exaggeration=config.chatterbox_exaggeration,
            )
        if backend == "elevenlabs":
            return ElevenLabsVoice(
                voice_id=config.elevenlabs_voice_id,
                api_key=config.elevenlabs_api_key or None,
            )
        raise ValueError(
            f"Unknown voice backend '{config.backend}'. Expected one of: "
            "silent, espeak, piper, kokoro, chatterbox, elevenlabs."
        )
    except Exception as exc:
        log.error("Voice backend '%s' failed to start: %s", backend, exc)
        try:
            log.warning("Falling back to espeak.")
            return EspeakVoice()
        except Exception:
            log.warning("Falling back to silence. The light will still speak.")
            return SilentVoice()
