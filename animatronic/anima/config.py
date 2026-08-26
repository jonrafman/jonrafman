"""Loading config.yaml into typed settings.

Everything that differs between "on Jon's laptop" and "in the gallery" lives in
config.yaml. The code never checks whether it is on a Raspberry Pi; it asks the
config what to build and degrades gracefully when the hardware is not there.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    """Substitute ``${VAR}`` in strings, recursively.

    Keeps API keys out of the config file and out of git.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _section(data: dict, *path: str) -> dict:
    """Fetch a nested section, returning {} rather than raising on absence."""
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


@dataclass
class CharacterConfig:
    name: str = "The Figure"
    persona_file: str = "persona/example.md"
    greeting: str = ""
    max_reply_sentences: int = 3

    @classmethod
    def load(cls, data: dict) -> "CharacterConfig":
        s = _section(data, "character")
        return cls(
            name=s.get("name", cls.name),
            persona_file=s.get("persona_file", cls.persona_file),
            greeting=s.get("greeting", cls.greeting),
            max_reply_sentences=int(s.get("max_reply_sentences", cls.max_reply_sentences)),
        )


@dataclass
class ConversationConfig:
    max_history_turns: int = 12
    listen_timeout: float = 12.0
    max_turns: int = 40

    @classmethod
    def load(cls, data: dict) -> "ConversationConfig":
        s = _section(data, "conversation")
        return cls(
            max_history_turns=int(s.get("max_history_turns", cls.max_history_turns)),
            listen_timeout=float(s.get("listen_timeout", cls.listen_timeout)),
            max_turns=int(s.get("max_turns", cls.max_turns)),
        )


@dataclass
class BrainConfig:
    backend: str = "scripted"
    temperature: float = 0.85
    max_tokens: int = 160
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    claude_model: str = "claude-sonnet-5"
    scripted_lines: list[str] = field(default_factory=list)
    scripted_keyed: dict[str, str] = field(default_factory=dict)
    fallback_lines: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, data: dict) -> "BrainConfig":
        s = _section(data, "brain")
        ollama = _section(s, "ollama")
        claude = _section(s, "claude")
        scripted = _section(s, "scripted")
        return cls(
            backend=s.get("backend", cls.backend),
            temperature=float(s.get("temperature", cls.temperature)),
            max_tokens=int(s.get("max_tokens", cls.max_tokens)),
            ollama_host=ollama.get("host", cls.ollama_host),
            ollama_model=ollama.get("model", cls.ollama_model),
            claude_model=claude.get("model", cls.claude_model),
            scripted_lines=list(scripted.get("lines", []) or []),
            scripted_keyed=dict(scripted.get("keyed", {}) or {}),
            fallback_lines=list(s.get("fallback_lines", []) or []),
        )


@dataclass
class EarsConfig:
    backend: str = "text"
    whisper_model: str = "base.en"
    device_index: int | None = None
    silence_threshold: float = 0.015
    silence_duration: float = 1.2
    max_duration: float = 20.0

    @classmethod
    def load(cls, data: dict) -> "EarsConfig":
        s = _section(data, "ears")
        whisper = _section(s, "whisper")
        index = whisper.get("device_index", None)
        return cls(
            backend=s.get("backend", cls.backend),
            whisper_model=whisper.get("model", cls.whisper_model),
            device_index=None if index is None else int(index),
            silence_threshold=float(s.get("silence_threshold", cls.silence_threshold)),
            silence_duration=float(s.get("silence_duration", cls.silence_duration)),
            max_duration=float(s.get("max_duration", cls.max_duration)),
        )


@dataclass
class VoiceConfig:
    backend: str = "silent"
    espeak_voice: str = "en"
    espeak_speed: int = 150
    espeak_pitch: int = 50
    piper_model: str = ""
    piper_length_scale: float = 1.0
    elevenlabs_voice_id: str = ""
    elevenlabs_api_key: str = ""

    @classmethod
    def load(cls, data: dict) -> "VoiceConfig":
        s = _section(data, "voice")
        espeak = _section(s, "espeak")
        piper = _section(s, "piper")
        eleven = _section(s, "elevenlabs")
        return cls(
            backend=s.get("backend", cls.backend),
            espeak_voice=espeak.get("voice", cls.espeak_voice),
            espeak_speed=int(espeak.get("speed", cls.espeak_speed)),
            espeak_pitch=int(espeak.get("pitch", cls.espeak_pitch)),
            piper_model=piper.get("model", cls.piper_model),
            piper_length_scale=float(piper.get("length_scale", cls.piper_length_scale)),
            elevenlabs_voice_id=eleven.get("voice_id", cls.elevenlabs_voice_id),
            elevenlabs_api_key=eleven.get("api_key", cls.elevenlabs_api_key),
        )


@dataclass
class JawConfig:
    enabled: bool = False
    play_audio: bool = True
    log_envelope: bool = True
    servo_pin: int = 18
    closed_angle: float = 0.0
    open_angle: float = 35.0
    max_opening: float = 1.0
    use_pigpio: bool = True

    @classmethod
    def load(cls, data: dict) -> "JawConfig":
        s = _section(data, "jaw")
        return cls(
            enabled=bool(s.get("enabled", cls.enabled)),
            play_audio=bool(s.get("play_audio", cls.play_audio)),
            log_envelope=bool(s.get("log_envelope", cls.log_envelope)),
            servo_pin=int(s.get("servo_pin", cls.servo_pin)),
            closed_angle=float(s.get("closed_angle", cls.closed_angle)),
            open_angle=float(s.get("open_angle", cls.open_angle)),
            max_opening=float(s.get("max_opening", cls.max_opening)),
            use_pigpio=bool(s.get("use_pigpio", cls.use_pigpio)),
        )


@dataclass
class TriggerConfig:
    backend: str = "keyboard"
    pin: int = 17
    settle_seconds: float = 0.8
    cooldown: float = 5.0

    @classmethod
    def load(cls, data: dict) -> "TriggerConfig":
        s = _section(data, "trigger")
        return cls(
            backend=s.get("backend", cls.backend),
            pin=int(s.get("pin", cls.pin)),
            settle_seconds=float(s.get("settle_seconds", cls.settle_seconds)),
            cooldown=float(s.get("cooldown", cls.cooldown)),
        )


@dataclass
class IdleConfig:
    enabled: bool = True
    breath_amplitude: float = 0.05
    breath_period: float = 5.0
    swallow_every: float = 25.0

    @classmethod
    def load(cls, data: dict) -> "IdleConfig":
        s = _section(data, "idle")
        return cls(
            enabled=bool(s.get("enabled", cls.enabled)),
            breath_amplitude=float(s.get("breath_amplitude", cls.breath_amplitude)),
            breath_period=float(s.get("breath_period", cls.breath_period)),
            swallow_every=float(s.get("swallow_every", cls.swallow_every)),
        )


@dataclass
class Config:
    character: CharacterConfig
    conversation: ConversationConfig
    brain: BrainConfig
    ears: EarsConfig
    voice: VoiceConfig
    jaw: JawConfig
    trigger: TriggerConfig
    idle: IdleConfig
    root: Path = field(default_factory=Path.cwd)

    def resolve(self, path: str) -> Path:
        """Resolve a config-relative path against the config file's directory."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a YAML mapping: {config_path}")
        data = _expand_env(raw)

        return cls(
            character=CharacterConfig.load(data),
            conversation=ConversationConfig.load(data),
            brain=BrainConfig.load(data),
            ears=EarsConfig.load(data),
            voice=VoiceConfig.load(data),
            jaw=JawConfig.load(data),
            trigger=TriggerConfig.load(data),
            idle=IdleConfig.load(data),
            root=config_path.parent.resolve(),
        )
