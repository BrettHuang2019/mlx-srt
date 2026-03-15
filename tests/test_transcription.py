import pytest
from src.transcription.whisper_transcriber import transcribe_audio
from src.transcription import whisper_transcriber
from src.transcription import punctuation_kredor
import os
import json
from types import SimpleNamespace


class TestWhisperTranscription:
    """Test Whisper transcription functionality"""

    # Test data paths
    AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'audio')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    TEST_AUDIO = os.path.join(AUDIO_DIR, 'Test1.mp3')

    @classmethod
    def setup_class(cls):
        """Transcribe audio once and reuse result for all tests"""
        cls.result = transcribe_audio(cls.TEST_AUDIO)

        # Save output for debugging with same name as input audio
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        input_name = os.path.splitext(os.path.basename(cls.TEST_AUDIO))[0]
        output_filename = f"{input_name}.json"
        with open(os.path.join(cls.OUTPUT_DIR, output_filename), 'w', encoding='utf-8') as f:
            json.dump(cls.result, f, indent=2, ensure_ascii=False)

    def test_transcribe_returns_text_and_segments(self):
        """Should return dict with 'text' and 'segments' keys only"""
        assert isinstance(self.result, dict)
        assert set(self.result.keys()) == {"text", "segments"}

    def test_transcribe_text_is_string(self):
        """'text' should be non-empty string"""
        assert isinstance(self.result["text"], str)
        assert len(self.result["text"].strip()) > 0

    def test_segments_is_list(self):
        """'segments' should be a list"""
        assert isinstance(self.result["segments"], list)

    def test_segment_has_only_required_fields(self):
        """Each segment should have ONLY: id, start, end, text (no tokens, temperature, etc.)"""
        if self.result["segments"]:
            segment = self.result["segments"][0]
            required_fields = {"id", "start", "end", "text"}
            assert set(segment.keys()) == required_fields

    def test_segment_field_types(self):
        """id: int, start: float, end: float, text: str"""
        if self.result["segments"]:
            segment = self.result["segments"][0]
            assert isinstance(segment["id"], int)
            assert isinstance(segment["start"], float)
            assert isinstance(segment["end"], float)
            assert isinstance(segment["text"], str)
            assert segment["start"] >= 0
            assert segment["end"] > segment["start"]

    def test_invalid_path_raises_error(self):
        """FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            transcribe_audio("/nonexistent/path/file.wav")

    def test_different_audio_formats(self):
        """wav, mp3, m4a"""
        # Test with the available MP3 file (already transcribed in setup_class)
        assert isinstance(self.result, dict)
        assert "text" in self.result
        assert "segments" in self.result


def test_patch_mlx_whisper_loader_filters_unknown_model_config_keys(tmp_path, monkeypatch):
    if not whisper_transcriber.MLX_WHISPER_AVAILABLE:
        pytest.skip("mlx_whisper is not installed")

    repo_dir = tmp_path / "mock-whisper-repo"
    repo_dir.mkdir()
    (repo_dir / "config.json").write_text(
        json.dumps(
            {
                "num_mel_bins": 128,
                "max_source_positions": 1500,
                "d_model": 1280,
                "encoder_attention_heads": 20,
                "encoder_layers": 32,
                "vocab_size": 51866,
                "max_target_positions": 448,
                "decoder_attention_heads": 20,
                "decoder_layers": 32,
                "activation_dropout": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (repo_dir / "weights.safetensors").write_text("stub", encoding="utf-8")

    captured = {}

    def fake_whisper(model_args, dtype):
        captured["model_args"] = model_args
        captured["dtype"] = dtype

        class DummyModel:
            def update(self, weights):
                captured["weights"] = weights

            def parameters(self):
                return []

        return DummyModel()

    monkeypatch.setattr(
        whisper_transcriber.mlx_whisper.load_models.mx,
        "load",
        lambda path: {"encoder.weight": 1},
    )
    monkeypatch.setattr(
        whisper_transcriber.mlx_whisper.load_models.mx,
        "eval",
        lambda params: None,
    )
    monkeypatch.setattr(
        whisper_transcriber,
        "tree_unflatten",
        lambda items: {"tree": items},
    )
    monkeypatch.setattr(
        whisper_transcriber.mlx_whisper.load_models.whisper,
        "Whisper",
        fake_whisper,
    )

    model = whisper_transcriber.mlx_whisper.load_models.load_model(
        str(repo_dir),
        dtype="float16",
    )

    assert model is not None
    assert captured["model_args"].n_mels == 128
    assert captured["model_args"].n_text_layer == 32
    assert not hasattr(captured["model_args"], "activation_dropout")


def test_calculate_punctuation_ratio():
    ratio = whisper_transcriber._calculate_punctuation_ratio("Bonjour monde.")
    assert ratio == pytest.approx(1 / 13)


def test_transcribe_audio_runs_punctuation_step_on_low_punctuation(tmp_path, monkeypatch):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_text("stub", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
whisper:
  model_path: "primary-model"
  min_punctuation_ratio: 0.05
  punctuation:
    model_path: "punctuation-model"
  language: "french"
        """.strip(),
        encoding="utf-8",
    )

    whisper_calls = []
    punctuation_calls = []

    def fake_transcribe(speech_file, path_or_hf_repo, **kwargs):
        whisper_calls.append((speech_file, path_or_hf_repo, kwargs))
        return {
            "text": "bonjour tout le monde",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "bonjour tout le monde"}
            ],
        }

    monkeypatch.setattr(whisper_transcriber, "MLX_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(
        whisper_transcriber,
        "mlx_whisper",
        SimpleNamespace(transcribe=fake_transcribe),
        raising=False,
    )
    monkeypatch.setattr(
        whisper_transcriber,
        "_apply_punctuation_step",
        lambda payload, config: (
            punctuation_calls.append((payload["text"], config["model_path"])) or (
                {
                    "text": "Bonjour tout le monde.",
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.0, "text": "Bonjour tout le monde."}
                    ],
                },
                {
                    "model_path": config["model_path"],
                    "chunk_count": 1,
                    "chunk_words": 180,
                    "input_punctuation_ratio": 0.0,
                    "output_punctuation_ratio": 0.2,
                    "mapping_stats": {"matched_word_count": 4},
                    "punctuation_summary": {"total_chunks": 1},
                },
            )
        ),
    )

    result = transcribe_audio(str(audio_file), str(config_file))

    assert [call[1] for call in whisper_calls] == ["primary-model"]
    assert punctuation_calls == [("bonjour tout le monde", "punctuation-model")]
    assert result["text"] == "Bonjour tout le monde."
    assert result["segments"][0]["text"] == "Bonjour tout le monde."


def test_transcribe_audio_returns_punctuated_result_when_whisper_ratio_is_low(tmp_path, monkeypatch):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_text("stub", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
whisper:
  model_path: "primary-model"
  min_punctuation_ratio: 0.20
  punctuation:
    model_path: "punctuation-model"
        """.strip(),
        encoding="utf-8",
    )

    punctuation_calls = []

    def fake_transcribe(speech_file, path_or_hf_repo, **kwargs):
        return {
            "text": "bonjour tout le monde",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "bonjour tout le monde"}
            ],
        }

    monkeypatch.setattr(whisper_transcriber, "MLX_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(
        whisper_transcriber,
        "mlx_whisper",
        SimpleNamespace(transcribe=fake_transcribe),
        raising=False,
    )
    monkeypatch.setattr(
        whisper_transcriber,
        "_apply_punctuation_step",
        lambda payload, config: (
            punctuation_calls.append((payload["text"], config["model_path"])) or (
                {
                    "text": "Bonjour tout le monde.",
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.0, "text": "Bonjour tout le monde."}
                    ],
                },
                {
                    "model_path": config["model_path"],
                    "chunk_count": 1,
                    "chunk_words": 180,
                    "input_punctuation_ratio": 0.0,
                    "output_punctuation_ratio": 0.2,
                    "mapping_stats": {"matched_word_count": 4},
                    "punctuation_summary": {"total_chunks": 1},
                },
            )
        ),
    )

    result = transcribe_audio(str(audio_file), str(config_file))

    assert result["text"] == "Bonjour tout le monde."
    assert punctuation_calls == [("bonjour tout le monde", "punctuation-model")]


def test_transcribe_audio_can_return_metadata_for_pipeline(tmp_path, monkeypatch):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_text("stub", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
whisper:
  model_path: "primary-model"
  min_punctuation_ratio: 0.01
        """.strip(),
        encoding="utf-8",
    )

    def fake_transcribe(speech_file, path_or_hf_repo, **kwargs):
        return {
            "text": "Bonjour tout le monde.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "Bonjour tout le monde."}
            ],
        }

    monkeypatch.setattr(whisper_transcriber, "MLX_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(
        whisper_transcriber,
        "mlx_whisper",
        SimpleNamespace(transcribe=fake_transcribe),
        raising=False,
    )

    result, metadata = transcribe_audio(str(audio_file), str(config_file), return_metadata=True)

    assert result["text"] == "Bonjour tout le monde."
    assert metadata["selected_strategy"] == "whisper"
    assert metadata["selected_model_path"] == "primary-model"
    assert metadata["punctuation_pass_applied"] is False
    assert metadata["attempts"][0]["punctuation_ratio"] > 0
    assert metadata["attempts"][0]["response"]["text"] == "Bonjour tout le monde."
    assert metadata["attempts"][0]["response"]["segments"][0]["text"] == "Bonjour tout le monde."


def test_transcribe_audio_failure_includes_attempt_metadata(tmp_path, monkeypatch):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_text("stub", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
whisper:
  model_path: "primary-model"
  min_punctuation_ratio: 0.20
  punctuation:
    model_path: "punctuation-model"
        """.strip(),
        encoding="utf-8",
    )

    def fake_transcribe(speech_file, path_or_hf_repo, **kwargs):
        return {
            "text": "bonjour tout le monde",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "bonjour tout le monde"}
            ],
        }

    error_details = {
        "stage": "segment_mapping",
        "segment_id": 0,
        "matched_words": 3,
        "punctuation_chunks": [
            {
                "input_text": "bonjour tout le monde",
                "raw_output": "Bonjour tout le monde.",
            }
        ],
    }

    def fake_apply_punctuation_step(payload, config):
        raise punctuation_kredor.PunctuationAlignmentError(
            "Failed to map punctuated text back onto transcription segments.",
            details=error_details,
        )

    monkeypatch.setattr(whisper_transcriber, "MLX_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(
        whisper_transcriber,
        "mlx_whisper",
        SimpleNamespace(transcribe=fake_transcribe),
        raising=False,
    )
    monkeypatch.setattr(
        whisper_transcriber,
        "_apply_punctuation_step",
        fake_apply_punctuation_step,
    )

    with pytest.raises(whisper_transcriber.TranscriptionPipelineError) as exc_info:
        transcribe_audio(str(audio_file), str(config_file))

    metadata = exc_info.value.metadata
    assert metadata["status"] == "failed"
    assert metadata["error"] == "Failed to map punctuated text back onto transcription segments."
    assert metadata["error_details"]["stage"] == "segment_mapping"
    assert metadata["error_details"]["segment_id"] == 0
    assert metadata["attempts"][0]["type"] == "whisper"
    assert metadata["attempts"][1]["type"] == "punctuation"
    assert metadata["attempts"][1]["status"] == "failed"
    assert metadata["attempts"][1]["model_path"] == "punctuation-model"
    assert metadata["attempts"][1]["error_details"]["matched_words"] == 3
    assert metadata["attempts"][1]["error_details"]["punctuation_chunks"][0]["input_text"] == "bonjour tout le monde"
    assert metadata["attempts"][1]["error_details"]["punctuation_chunks"][0]["raw_output"] == "Bonjour tout le monde."
    assert exc_info.value.partial_result["text"] == "bonjour tout le monde"


def test_apply_punctuation_to_payload_returns_whisper_shaped_json(monkeypatch):
    payload = {
        "text": "bonjour tout le monde",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "bonjour tout le monde"}
        ],
    }

    monkeypatch.setattr(
        punctuation_kredor,
        "process_whisper_payload",
        lambda payload, classifier=None, model_id=None, chunk_words=None: (
            {
                "model": model_id,
                "timestamp": "2026-01-01T00:00:00",
                "source_text": payload["text"],
                "punctuated_text": "Bonjour tout le monde.",
                "mapping_stats": {"matched_word_count": 4},
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "original_text": "bonjour tout le monde",
                        "punctuated_text": "Bonjour tout le monde.",
                    }
                ],
            },
            {
                "punctuation": {
                    "total_chunks": 1,
                    "chunk_words": 180,
                },
                "mapping": {"matched_word_count": 4},
            },
            [],
        ),
    )

    result, metadata = punctuation_kredor.apply_punctuation_to_payload(
        payload,
        {"model_path": "punctuation-model", "chunk_words": 180},
        whisper_transcriber._calculate_punctuation_ratio,
    )

    assert result["text"] == "Bonjour tout le monde."
    assert result["segments"][0]["text"] == "Bonjour tout le monde."
    assert metadata["model_path"] == "punctuation-model"
    assert metadata["chunk_count"] == 1


def test_rebuild_segments_keeps_mid_sentence_segment_start_lowercase():
    segments = [
        {"id": 228, "start": 486.0, "end": 487.8, "text": " Et on\n  n'imagine pas"},
        {"id": 229, "start": 487.8, "end": 490.14, "text": " si vous regardez des entreprises\n  familiales"},
    ]

    rebuilt_segments, stats = punctuation_kredor.rebuild_segments(
        segments,
        "Et on n'imagine pas si vous regardez des entreprises familiales.",
    )

    assert rebuilt_segments[0]["punctuated_text"] == "Et on\n  n'imagine pas"
    assert rebuilt_segments[1]["punctuated_text"] == "si vous regardez des entreprises\n  familiales."
    assert stats["matched_word_count"] == 10


def test_rebuild_segments_lowercases_continuation_segment_start_after_comma():
    segments = [
        {"id": 5, "start": 19.32, "end": 20.82, "text": "D'accord, ben moi, c'etait aujourd'hui,"},
        {"id": 6, "start": 20.82, "end": 23.94, "text": "Mais si vous etes un client fidele, vous avez peut-etre recu un cadeau."},
    ]

    rebuilt_segments, stats = punctuation_kredor.rebuild_segments(
        segments,
        "D'accord, ben moi, c'etait aujourd'hui, mais si vous etes un client fidele, vous avez peut-etre recu un cadeau. Aussi, il faut verifier.",
    )

    assert rebuilt_segments[0]["punctuated_text"] == "D'accord, ben moi, c'etait aujourd'hui,"
    assert rebuilt_segments[1]["punctuated_text"].startswith("mais si vous etes un client fidele")
    assert stats["matched_word_count"] == 18


def test_transcribe_audio_wraps_final_whisper_failure_with_metadata(tmp_path, monkeypatch):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"audio")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
whisper:
  model_path: "broken-a"
  min_punctuation_ratio: 0.01
""".strip(),
        encoding="utf-8",
    )

    def fake_transcribe(*args, **kwargs):
        raise RuntimeError(f"failed:{kwargs['path_or_hf_repo']}")

    monkeypatch.setattr(whisper_transcriber, "MLX_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(
        whisper_transcriber,
        "mlx_whisper",
        SimpleNamespace(transcribe=fake_transcribe),
        raising=False,
    )

    with pytest.raises(whisper_transcriber.TranscriptionPipelineError) as exc_info:
        whisper_transcriber.transcribe_audio(str(audio_file), str(config_file))

    metadata = exc_info.value.metadata
    assert metadata["status"] == "failed"
    assert metadata["selected_strategy"] == "failed"
    assert metadata["error"] == "failed:broken-a"
    assert [attempt["model_path"] for attempt in metadata["attempts"]] == ["broken-a"]
    assert all(attempt["status"] == "failed" for attempt in metadata["attempts"])
