import json
import os
from dataclasses import fields
from pathlib import Path

import yaml

try:
    import mlx_whisper
    import mlx.nn as nn
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_unflatten
    from safetensors.numpy import load_file as load_safetensors_file

    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False


def _patch_mlx_whisper_loader():
    """Ignore extra config keys from newer Whisper repos."""
    if not MLX_WHISPER_AVAILABLE:
        return

    load_models = mlx_whisper.load_models
    transcribe_module = mlx_whisper.transcribe.__module__
    transcribe_module = __import__(transcribe_module, fromlist=["load_model"])
    whisper_module = load_models.whisper

    if getattr(load_models.load_model, "_mlx_srt_patched", False):
        return

    valid_keys = {field.name for field in fields(whisper_module.ModelDimensions)}
    def normalize_model_config(config):
        if valid_keys.issubset(config.keys()):
            return {key: config[key] for key in valid_keys}

        mapped = {
            "n_mels": config.get("n_mels", config.get("num_mel_bins")),
            "n_audio_ctx": config.get("n_audio_ctx", config.get("max_source_positions")),
            "n_audio_state": config.get("n_audio_state", config.get("d_model")),
            "n_audio_head": config.get("n_audio_head", config.get("encoder_attention_heads")),
            "n_audio_layer": config.get("n_audio_layer", config.get("encoder_layers")),
            "n_vocab": config.get("n_vocab", config.get("vocab_size")),
            "n_text_ctx": config.get("n_text_ctx", config.get("max_target_positions", config.get("max_length"))),
            "n_text_state": config.get("n_text_state", config.get("d_model")),
            "n_text_head": config.get("n_text_head", config.get("decoder_attention_heads")),
            "n_text_layer": config.get("n_text_layer", config.get("decoder_layers")),
        }
        return {key: value for key, value in mapped.items() if value is not None}

    def patched_load_model(path_or_hf_repo, dtype):
        model_path = Path(path_or_hf_repo)
        if not model_path.exists():
            model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = whisper_module.ModelDimensions(**normalize_model_config(config))

        weight_file = model_path / "weights.safetensors"
        if weight_file.exists():
            weights = mlx_whisper.load_models.mx.load(str(weight_file))
        else:
            weight_file = model_path / "weights.npz"
            if weight_file.exists():
                weights = mlx_whisper.load_models.mx.load(str(weight_file))
            else:
                weight_file = model_path / "model.safetensors"
                weights = {
                    key: mlx_whisper.load_models.mx.array(value)
                    for key, value in load_safetensors_file(str(weight_file)).items()
                }
        model = whisper_module.Whisper(model_args, dtype)

        if quantization is not None:
            class_predicate = (
                lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                and f"{p}.scales" in weights
            )
            nn.quantize(model, **quantization, class_predicate=class_predicate)

        model.update(tree_unflatten(list(weights.items())))
        mlx_whisper.load_models.mx.eval(model.parameters())
        return model

    patched_load_model._mlx_srt_patched = True
    load_models.load_model = patched_load_model
    transcribe_module.load_model = patched_load_model


_patch_mlx_whisper_loader()


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
