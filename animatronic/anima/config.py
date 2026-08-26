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
    style_rules: bool = False
    lexicon_source: str = ""

    @classmethod
    def load(cls, data: dict) -> "CharacterConfig":
        s = _section(data, "character")
        return cls(
            name=s.get("name", cls.name),
            persona_file=s.get("persona_file", cls.persona_file),
            greeting=s.get("greeting", cls.greeting),
            max_reply_sentences=int(s.get("max_reply_sentences", cls.max_reply_sentences)),
            style_rules=bool(s.get("style_rules", cls.style_rules)),
            lexicon_source=s.get("lexicon_source", cls.lexicon_source),
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
    repeat_penalty: float = 1.15
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    claude_model: str = "claude-sonnet-5"
    llamacpp_model: str = ""
    llamacpp_grammar: str = ""
    llamacpp_n_ctx: int = 4096
    llamacpp_n_gpu_layers: int = -1
    scripted_lines: list[str] = field(default_factory=list)
    scripted_keyed: dict[str, str] = field(default_factory=dict)
    fallback_lines: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, data: dict) -> "BrainConfig":
        s = _section(data, "brain")
        ollama = _section(s, "ollama")
        claude = _section(s, "claude")
        scripted = _section(s, "scripted")
        llamacpp = _section(s, "llamacpp")
        return cls(
            backend=s.get("backend", cls.backend),
            temperature=float(s.get("temperature", cls.temperature)),
            max_tokens=int(s.get("max_tokens", cls.max_tokens)),
            repeat_penalty=float(s.get("repeat_penalty", cls.repeat_penalty)),
            ollama_host=ollama.get("host", cls.ollama_host),
            ollama_model=ollama.get("model", cls.ollama_model),
            claude_model=claude.get("model", cls.claude_model),
            llamacpp_model=llamacpp.get("model", cls.llamacpp_model),
            llamacpp_grammar=llamacpp.get("grammar", cls.llamacpp_grammar),
            llamacpp_n_ctx=int(llamacpp.get("n_ctx", cls.llamacpp_n_ctx)),
            llamacpp_n_gpu_layers=int(
                llamacpp.get("n_gpu_layers", cls.llamacpp_n_gpu_layers)
            ),
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
class LightConfig:
    backend: str = "console"
    draw: bool = True
    gamma: float = 2.2
    max_level: float = 1.0
    pin: int = 18
    pwm_frequency: int = 2000
    addressable_pin: str = "D18"
    led_count: int = 24
    color: tuple[int, int, int] = (255, 255, 255)

    @classmethod
    def load(cls, data: dict) -> "LightConfig":
        s = _section(data, "light")
        pwm = _section(s, "pwm")
        addressable = _section(s, "addressable")
        color = addressable.get("color", list(cls.color))
        return cls(
            backend=s.get("backend", cls.backend),
            draw=bool(s.get("draw", cls.draw)),
            gamma=float(s.get("gamma", cls.gamma)),
            max_level=float(s.get("max_level", cls.max_level)),
            pin=int(pwm.get("pin", cls.pin)),
            pwm_frequency=int(pwm.get("frequency", cls.pwm_frequency)),
            addressable_pin=addressable.get("pin", cls.addressable_pin),
            led_count=int(addressable.get("count", cls.led_count)),
            color=tuple(int(c) for c in color)[:3],
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
class PresenceConfig:
    enabled: bool = True
    dormant_level: float = 0.08
    dormant_swing: float = 0.05
    dormant_period: float = 9.0
    attending_level: float = 0.35
    thinking_level: float = 0.5
    thinking_swing: float = 0.22
    thinking_rate: float = 1.6
    speaking_baseline: float = 0.28
    ease: float = 0.12

    @classmethod
    def load(cls, data: dict) -> "PresenceConfig":
        s = _section(data, "presence")
        dormant = _section(s, "dormant")
        thinking = _section(s, "thinking")
        return cls(
            enabled=bool(s.get("enabled", cls.enabled)),
            dormant_level=float(dormant.get("level", cls.dormant_level)),
            dormant_swing=float(dormant.get("swing", cls.dormant_swing)),
            dormant_period=float(dormant.get("period", cls.dormant_period)),
            attending_level=float(s.get("attending_level", cls.attending_level)),
            thinking_level=float(thinking.get("level", cls.thinking_level)),
            thinking_swing=float(thinking.get("swing", cls.thinking_swing)),
            thinking_rate=float(thinking.get("rate", cls.thinking_rate)),
            speaking_baseline=float(s.get("speaking_baseline", cls.speaking_baseline)),
            ease=float(s.get("ease", cls.ease)),
        )


@dataclass
class Config:
    character: CharacterConfig
    conversation: ConversationConfig
    brain: BrainConfig
    ears: EarsConfig
    voice: VoiceConfig
    light: LightConfig
    trigger: TriggerConfig
    presence: PresenceConfig
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

        config = cls(
            character=CharacterConfig.load(data),
            conversation=ConversationConfig.load(data),
            brain=BrainConfig.load(data),
            ears=EarsConfig.load(data),
            voice=VoiceConfig.load(data),
            light=LightConfig.load(data),
            trigger=TriggerConfig.load(data),
            presence=PresenceConfig.load(data),
            root=config_path.parent.resolve(),
        )

        # Model and grammar paths are written relative to the config file, so
        # the repo works the same wherever it is cloned. Resolve them here,
        # where the config's own location is known, rather than making every
        # consumer remember to.
        for field_name in ("llamacpp_model", "llamacpp_grammar"):
            value = getattr(config.brain, field_name)
            if value:
                setattr(config.brain, field_name, str(config.resolve(value)))

        return config
