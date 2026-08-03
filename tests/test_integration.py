import subprocess
from pathlib import Path

import pytest

from mlx_srt.config import load_config
from mlx_srt.pipeline import run_pipeline
from mlx_srt.srt import parse_srt


@pytest.mark.integration
def test_real_short_audio_produces_bilingual_srt(tmp_path):
    source = Path(__file__).parent / "audio" / "Test1.mp3"
    clip = tmp_path / "clip.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source), "-t", "15", "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(clip),
        ],
        check=True,
        capture_output=True,
    )
    output = run_pipeline(clip, load_config(clip), use_lock=False)
    segments = parse_srt(output.read_text(encoding="utf-8"))
    assert segments
    assert any("\u4e00" <= char <= "\u9fff" for segment in segments for char in segment.text)
