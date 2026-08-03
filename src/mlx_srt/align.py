"""Word-level forced alignment backed by mlx-audio."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


def align(
    wav_path: str | Path,
    text: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> list[dict[str, object]]:
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")
    from mlx_audio.stt import load  # lazy: model is large

    result = load(model_id).generate(audio=str(wav_path), text=text)
    return [
        {"start": item.start_time, "end": item.end_time, "text": item.text}
        for item in result
    ]
