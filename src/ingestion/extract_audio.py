import os
import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_path: str) -> None:
    """
    Extract audio from video file and convert to 16kHz mono WAV.

    Args:
        video_path: Path to input video file
        output_path: Path to output WAV file

    Raises:
        FileNotFoundError: If input video doesn't exist or output directory is invalid
        subprocess.CalledProcessError: If ffmpeg extraction fails
    """
    # Validate input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Validate output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    # Ensure output directory exists (in case output_path has no dirname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_path,           # input file
        '-vn',                       # no video
        '-acodec', 'pcm_s16le',      # 16-bit PCM
        '-ar', '16000',              # 16kHz sample rate
        '-ac', '1',                  # mono
        '-y',                        # overwrite output
        output_path
    ]

    # Execute ffmpeg command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            e.returncode,
            cmd,
            e.stdout,
            e.stderr
        )

    # Verify output file was created
    if not os.path.exists(output_path):
        raise RuntimeError(f"Audio extraction failed - output file not created: {output_path}")

    if os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Audio extraction failed - output file is empty: {output_path}")