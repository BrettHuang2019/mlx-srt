import pytest
from src.transcription.whisper_transcriber import transcribe_audio
from src.transcription import whisper_transcriber
import os
import json


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
