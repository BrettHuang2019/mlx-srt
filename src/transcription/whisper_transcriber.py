import json
import os
import yaml

try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        dict: Configuration parameters
    """
    if not os.path.exists(config_path):
        # Return default config if file doesn't exist
        return {
            "whisper": {
                "model_path": "models/large",
                "initial_prompt": None,
                "language": "french"
            }
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def transcribe_audio(speech_file, config_path="config.yaml"):
    """
    Transcribe audio file using mlx_whisper

    Args:
        speech_file: Path to audio file
        config_path: Path to config file

    Returns:
        dict: Transcription result with 'text' and 'segments' keys
    """
    if not MLX_WHISPER_AVAILABLE:
        raise ImportError("mlx_whisper is not installed. Install with: pip install mlx-whisper")

    if not os.path.exists(speech_file):
        raise FileNotFoundError(f"Audio file not found: {speech_file}")

    config = load_config(config_path)
    whisper_config = config.get("whisper", {})

    # Prepare whisper parameters
    whisper_params = {
        "path_or_hf_repo": whisper_config.get("model_path", "models/large")
    }

    # Add language if specified
    language = whisper_config.get("language")
    if language:
        whisper_params["language"] = language

    # Add initial_prompt if specified
    initial_prompt = whisper_config.get("initial_prompt")
    if initial_prompt:
        whisper_params["initial_prompt"] = initial_prompt

    result = mlx_whisper.transcribe(speech_file, **whisper_params)

    # Clean up segments to only include required fields
    clean_segments = []
    for segment in result["segments"]:
        clean_segment = {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        }
        clean_segments.append(clean_segment)

    return {
        "text": result["text"],
        "segments": clean_segments
    }
