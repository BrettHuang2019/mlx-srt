"""Configuration loading with packaged defaults and location-aware overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class STTConfig:
    model_path: str


@dataclass(frozen=True)
class PunctuationConfig:
    enabled: bool
    model_path: str
    chunk_words: int


@dataclass(frozen=True)
class AlignConfig:
    model_path: str


@dataclass(frozen=True)
class MergeConfig:
    max_chars: int
    min_chars: int
    min_duration: float


@dataclass(frozen=True)
class TranslateConfig:
    model_path: str
    batch_size: int
    max_tokens: int
    temperature: float
    max_retries: int
    retry_delay: float
    prompt_file: Path


@dataclass(frozen=True)
class SystemConfig:
    min_ram_gb: float
    task_check_interval: float
    max_wait_time_minutes: float


@dataclass(frozen=True)
class Config:
    stt: STTConfig
    punctuation: PunctuationConfig
    align: AlignConfig
    merge: MergeConfig
    translate: TranslateConfig
    system: SystemConfig
    source: Path | None = None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}") from None
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return data


def _walk_up_config(input_path: Path | None) -> Path | None:
    if input_path is None:
        return None
    directory = input_path.expanduser().resolve().parent
    for candidate_dir in (directory, *directory.parents):
        candidate = candidate_dir / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def find_override_config(
    input_path: str | Path | None = None,
    explicit: str | Path | None = None,
) -> tuple[Path | None, bool]:
    """Return the highest-priority override and whether it was explicitly requested."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve(), True
    env_path = os.environ.get("MLX_SRT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve(), True
    home_path = Path.home() / ".mlx-srt" / "config.yaml"
    if home_path.is_file():
        return home_path.resolve(), False
    walked = _walk_up_config(Path(input_path) if input_path is not None else None)
    return walked, False


def load_config(
    input_path: str | Path | None = None,
    explicit: str | Path | None = None,
) -> Config:
    package_dir = resources.files("mlx_srt")
    defaults = yaml.safe_load(
        package_dir.joinpath("defaults.yaml").read_text(encoding="utf-8")
    )
    override_path, required = find_override_config(input_path, explicit)
    override: dict[str, Any] = {}
    if override_path is not None:
        if override_path.is_file():
            override = _read_yaml(override_path)
        elif required:
            raise FileNotFoundError(f"Configuration file not found: {override_path}")
        else:
            override_path = None

    merged = {section: dict(values) for section, values in defaults.items()}
    for section, values in override.items():
        if section not in merged or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid configuration section: {section}")
        merged[section].update(values)

    prompt_value = merged["translate"]["prompt_file"]
    prompt_declared_by_override = (
        override_path is not None
        and "prompt_file" in override.get("translate", {})
    )
    if Path(prompt_value).is_absolute():
        prompt_path = Path(prompt_value)
    elif prompt_declared_by_override:
        prompt_path = override_path.parent / prompt_value
    else:
        prompt_path = Path(str(package_dir)) / prompt_value

    return Config(
        stt=STTConfig(**merged["stt"]),
        punctuation=PunctuationConfig(**merged["punctuation"]),
        align=AlignConfig(**merged["align"]),
        merge=MergeConfig(**merged["merge"]),
        translate=TranslateConfig(
            **{**merged["translate"], "prompt_file": prompt_path.resolve()}
        ),
        system=SystemConfig(**merged["system"]),
        source=override_path.resolve() if override_path else None,
    )
