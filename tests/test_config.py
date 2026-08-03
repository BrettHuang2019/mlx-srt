from pathlib import Path

import pytest

from mlx_srt.config import load_config


def test_packaged_defaults_and_walk_up_override(tmp_path):
    media_dir = tmp_path / "nested" / "media"
    media_dir.mkdir(parents=True)
    media = media_dir / "clip.mp4"
    media.touch()
    prompt = tmp_path / "custom.txt"
    prompt.write_text("{context} {segments}", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "merge:\n  max_chars: 88\ntranslate:\n  prompt_file: custom.txt\n",
        encoding="utf-8",
    )

    config = load_config(media)

    assert config.merge.max_chars == 88
    assert config.merge.min_chars == 30
    assert config.translate.prompt_file == prompt.resolve()
    assert config.stt.model_path.endswith("whisper-large-v3-asr-4bit")


def test_explicit_missing_config_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(explicit=tmp_path / "missing.yaml")


def test_default_prompt_is_packaged():
    config = load_config()
    assert config.translate.prompt_file.is_absolute()
    assert config.translate.prompt_file.is_file()
