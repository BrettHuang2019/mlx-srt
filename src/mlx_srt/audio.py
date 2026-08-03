"""Normalize any ffmpeg-supported media input to a shared WAV artifact."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def extract_audio(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")

    cmd = [
        "ffmpeg", "-i", str(input_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y", str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    if not output_path.exists():
        raise RuntimeError(f"Audio extraction failed - output file not created: {output_path}")
    if os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Audio extraction failed - output file is empty: {output_path}")
    return output_path.resolve()
