from types import SimpleNamespace

import pytest

from mlx_srt.align import align
from mlx_srt.stt import transcribe


def test_stt_returns_text_only(tmp_path, monkeypatch):
    wav = tmp_path / "audio.wav"
    wav.touch()
    model = SimpleNamespace(generate=lambda path: SimpleNamespace(text="bonjour"))
    monkeypatch.setattr("mlx_audio.stt.load", lambda model_id: model)
    assert transcribe(wav, "model") == {"text": "bonjour"}


def test_aligner_unwraps_result_objects(tmp_path, monkeypatch):
    wav = tmp_path / "audio.wav"
    wav.touch()
    item = SimpleNamespace(start_time=1.25, end_time=1.8, text="bonjour")
    model = SimpleNamespace(generate=lambda **kwargs: [item])
    monkeypatch.setattr("mlx_audio.stt.load", lambda model_id: model)
    assert align(wav, " bonjour ", "aligner") == [
        {"start": 1.25, "end": 1.8, "text": "bonjour"}
    ]


def test_model_stages_validate_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "missing", "model")
    wav = tmp_path / "audio.wav"
    wav.touch()
    with pytest.raises(ValueError, match="must not be empty"):
        align(wav, "  ", "aligner")
