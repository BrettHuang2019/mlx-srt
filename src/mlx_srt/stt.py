"""Speech-to-text stage backed by mlx-audio Whisper."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-asr-4bit"


def transcribe(wav_path: str | Path, model_id: str = DEFAULT_MODEL_ID) -> dict[str, str]:
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    from mlx_audio.stt import load  # lazy: model is large

    result = load(model_id).generate(str(wav_path))
    return {"text": str(result.text)}
