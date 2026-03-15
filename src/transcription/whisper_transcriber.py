import json
import os
import re
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

from transcription.punctuation_kredor import (
    DEFAULT_PUNCTUATION_CONFIG,
    apply_punctuation_to_payload as _apply_punctuation_to_payload_impl,
    get_punctuation_runtime_config as _get_punctuation_runtime_config,
)


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


PUNCTUATION_CHARS = {".", "!", "?", "…"}


class TranscriptionPipelineError(RuntimeError):
    """Raised when transcription fails after collecting diagnostic metadata."""

    def __init__(self, message, metadata=None, partial_result=None):
        super().__init__(message)
        self.metadata = metadata or {}
        self.partial_result = partial_result


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
                "language": "french",
                "min_punctuation_ratio": 0.01,
                "punctuation": DEFAULT_PUNCTUATION_CONFIG.copy(),
            }
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def _calculate_punctuation_ratio(text):
    if not text:
        return 0.0

    non_whitespace_chars = [char for char in text if not char.isspace()]
    if not non_whitespace_chars:
        return 0.0

    collapsed_text = re.sub(r"\.\.\.|…", "…", "".join(non_whitespace_chars))
    punctuation_count = sum(1 for char in collapsed_text if char in PUNCTUATION_CHARS)
    return punctuation_count / len(non_whitespace_chars)


def _clean_transcription_result(result):
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


def _apply_punctuation_step(base_result, punctuation_config):
    return _apply_punctuation_to_payload_impl(
        base_result,
        punctuation_config,
        _calculate_punctuation_ratio,
    )


def transcribe_audio(speech_file, config_path="config.yaml", return_metadata=False):
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
    model_path = whisper_config.get("model_path", "models/large")
    if isinstance(model_path, (list, tuple)):
        model_path = next((path for path in model_path if path), "models/large")
    min_punctuation_ratio = whisper_config.get("min_punctuation_ratio", 0.01)

    # Prepare whisper parameters
    whisper_params = {}

    # Add language if specified
    language = whisper_config.get("language")
    if language:
        whisper_params["language"] = language

    # Add initial_prompt if specified
    initial_prompt = whisper_config.get("initial_prompt")
    if initial_prompt:
        whisper_params["initial_prompt"] = initial_prompt

    last_error = None
    last_successful_result = None
    last_successful_ratio = None
    attempt_details = []

    print(f"🎯 Punctuation threshold: {min_punctuation_ratio:.4f}")

    def build_failure_metadata(error_message):
        metadata = {
            "selected_strategy": "failed",
            "selected_model_path": None,
            "final_punctuation_ratio": last_successful_ratio or 0.0,
            "min_punctuation_ratio": min_punctuation_ratio,
            "punctuation_pass_applied": False,
            "status": "failed",
            "error": error_message,
            "attempts": list(attempt_details),
        }
        if last_successful_ratio is not None:
            metadata["source_whisper_punctuation_ratio"] = last_successful_ratio
        return metadata

    print(f"🤖 Whisper attempt 1/1: {model_path}")
    try:
        result = mlx_whisper.transcribe(
            speech_file,
            path_or_hf_repo=model_path,
            **whisper_params,
        )
    except Exception as exc:
        last_error = exc
        attempt_details.append(
            {
                "type": "whisper",
                "model_path": model_path,
                "status": "failed",
                "error": str(exc),
            }
        )
        print(f"❌ Whisper failed: {exc}")

    if last_error is None:
        clean_result = _clean_transcription_result(result)
        punctuation_ratio = _calculate_punctuation_ratio(clean_result["text"])
        last_successful_result = clean_result
        last_successful_ratio = punctuation_ratio
        attempt_details.append(
            {
                "type": "whisper",
                "model_path": model_path,
                "status": "completed",
                "punctuation_ratio": punctuation_ratio,
                "response": clean_result,
            }
        )
        meets_threshold = punctuation_ratio >= min_punctuation_ratio
        print(
            f"📊 Punctuation ratio: {punctuation_ratio:.4f} "
            f"(threshold {min_punctuation_ratio:.4f})"
        )
        print("✅ Meets threshold: yes" if meets_threshold else "⚠️ Meets threshold: no")

        if meets_threshold:
            print("➡️ Next step: accept this Whisper result and continue.")
            metadata = {
                "selected_strategy": "whisper",
                "selected_model_path": model_path,
                "final_punctuation_ratio": punctuation_ratio,
                "min_punctuation_ratio": min_punctuation_ratio,
                "punctuation_pass_applied": False,
                "attempts": attempt_details,
            }
            return (clean_result, metadata) if return_metadata else clean_result

        punctuation_config = _get_punctuation_runtime_config(whisper_config)
        print("📝 Whisper output stayed below the punctuation threshold.")
        print(
            f"➡️ Next step: run punctuation step with "
            f"'{punctuation_config['model_path']}'."
        )
        try:
            punctuated_result, punctuation_metadata = _apply_punctuation_step(
                last_successful_result,
                punctuation_config,
            )
        except Exception as exc:
            error_details = getattr(exc, "details", None)
            attempt_details.append(
                {
                    "type": "punctuation",
                    "model_path": punctuation_config["model_path"],
                    "status": "failed",
                    "input_punctuation_ratio": _calculate_punctuation_ratio(
                        last_successful_result["text"]
                    ),
                    "error": str(exc),
                    "error_details": error_details,
                }
            )
            failure_metadata = build_failure_metadata(str(exc))
            if error_details:
                failure_metadata["error_details"] = error_details
            print("❌ Punctuation step failed.")
            print("➡️ Next step: save the raw Whisper transcript and stop the pipeline.")
            raise TranscriptionPipelineError(
                str(exc),
                metadata=failure_metadata,
                partial_result=last_successful_result,
            ) from exc
        final_ratio = punctuation_metadata["output_punctuation_ratio"]
        attempt_details.append(
            {
                "type": "punctuation",
                "model_path": punctuation_metadata["model_path"],
                "status": "completed",
                "chunk_count": punctuation_metadata["chunk_count"],
                "chunk_words": punctuation_metadata["chunk_words"],
                "input_punctuation_ratio": punctuation_metadata["input_punctuation_ratio"],
                "output_punctuation_ratio": final_ratio,
                "mapping_stats": punctuation_metadata["mapping_stats"],
                "punctuation_summary": punctuation_metadata["punctuation_summary"],
                "response": punctuated_result,
            }
        )
        print(
            f"📊 Punctuation step ratio: {final_ratio:.4f} "
            f"(input {punctuation_metadata['input_punctuation_ratio']:.4f}, "
            f"threshold {min_punctuation_ratio:.4f})"
        )
        print("✅ Meets threshold: yes" if final_ratio >= min_punctuation_ratio else "⚠️ Meets threshold: no")
        print("➡️ Next step: use the punctuated transcript and continue.")
        metadata = {
            "selected_strategy": "punctuation",
            "selected_model_path": punctuation_metadata["model_path"],
            "source_whisper_punctuation_ratio": last_successful_ratio,
            "final_punctuation_ratio": final_ratio,
            "min_punctuation_ratio": min_punctuation_ratio,
            "punctuation_pass_applied": True,
            "punctuation_model": punctuation_metadata["model_path"],
            "punctuation_chunk_count": punctuation_metadata["chunk_count"],
            "punctuation_chunk_words": punctuation_metadata["chunk_words"],
            "punctuation_mapping_stats": punctuation_metadata["mapping_stats"],
            "punctuation_summary": punctuation_metadata["punctuation_summary"],
            "attempts": attempt_details,
        }
        return (punctuated_result, metadata) if return_metadata else punctuated_result

    if last_error is not None:
        print("❌ Transcription failed before any usable Whisper transcript was produced.")
        raise TranscriptionPipelineError(
            str(last_error),
            metadata=build_failure_metadata(str(last_error)),
        ) from last_error

    raise RuntimeError("Whisper transcription failed without producing a result.")
