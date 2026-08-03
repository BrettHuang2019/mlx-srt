import subprocess

import pytest

from mlx_srt.audio import extract_audio


def test_extract_audio_builds_expected_ffmpeg_command(tmp_path, monkeypatch):
    source = tmp_path / "input.mp3"
    source.write_bytes(b"media")
    output = tmp_path / "out.wav"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output.write_bytes(b"wav")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert extract_audio(source, output) == output.resolve()
    command, kwargs = calls[0]
    assert command[-1] == str(output)
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-ac") + 1] == "1"
    assert "pcm_s16le" in command
    assert kwargs["check"] is True


def test_extract_audio_validates_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "missing", tmp_path / "out.wav")
    source = tmp_path / "input"
    source.touch()
    with pytest.raises(FileNotFoundError):
        extract_audio(source, tmp_path / "missing" / "out.wav")
