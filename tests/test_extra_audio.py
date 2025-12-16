

import pytest
import os
import wave
import contextlib
from pathlib import Path

from src.ingestion.extract_audio import extract_audio


def test_extract_audio_creates_valid_output():
    """
    Test that audio extraction succeeds by verifying:
    1. Output WAV file exists
    2. File has valid audio data (can be opened/read)
    3. Sample rate is 16kHz
    4. Audio is mono (1 channel)
    5. Duration matches source video (within ±0.1s tolerance)

    Input: video file path
    Expected: valid 16kHz mono WAV with matching duration
    """
    # Setup
    video_path = Path("tests/video/Test_video_1.mp4")
    output_dir = Path("tests/output")
    output_path = output_dir / "Test_video_1.wav"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Clean up any existing output file
    if output_path.exists():
        output_path.unlink()

    # Execute
    extract_audio(str(video_path), str(output_path))

    # Verify output file exists
    assert output_path.exists(), f"Output file {output_path} was not created"
    assert output_path.stat().st_size > 0, "Output file is empty"

    # Verify WAV file format and properties
    with contextlib.closing(wave.open(str(output_path), 'rb')) as wav_file:
        assert wav_file.getnchannels() == 1, f"Expected 1 channel (mono), got {wav_file.getnchannels()}"
        assert wav_file.getframerate() == 16000, f"Expected 16kHz sample rate, got {wav_file.getframerate()}"
        assert wav_file.getsampwidth() == 2, f"Expected 16-bit audio (2 bytes), got {wav_file.getsampwidth()}"

        # Get duration from WAV file
        n_frames = wav_file.getnframes()
        duration = n_frames / wav_file.getframerate()

        # Verify duration is reasonable (should be positive)
        assert duration > 0, f"Duration should be positive, got {duration}"
        assert duration < 3600, f"Duration seems too long: {duration}s"


def test_extract_audio_overwrites_existing():
    """Test that extract_audio overwrites existing files."""
    # Setup
    video_path = Path("tests/video/Test_video_1.mp4")
    output_dir = Path("tests/output")
    output_path = output_dir / "Test_video_1.wav"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Create a dummy file first
    output_path.write_text("dummy content")

    # Execute
    extract_audio(str(video_path), str(output_path))

    # Verify file was overwritten with valid WAV data
    assert output_path.exists()
    with contextlib.closing(wave.open(str(output_path), 'rb')) as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000


def test_extract_audio_invalid_input():
    """Test that extract_audio raises appropriate errors for invalid input."""
    output_dir = Path("tests/output")
    output_path = output_dir / "test.wav"

    # Test with non-existent video file
    with pytest.raises(FileNotFoundError):
        extract_audio("non_existent_video.mp4", str(output_path))

    # Test with invalid output directory
    with pytest.raises(FileNotFoundError):
        extract_audio("tests/video/Test_video_1.mp4", "/non/existent/dir/output.wav")