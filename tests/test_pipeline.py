import json
from dataclasses import replace

import pytest

from mlx_srt.config import load_config
from mlx_srt.pipeline import DONE, FRESH, artifact_paths, resolve_start_step, run_pipeline


def _state(input_file, last=None, current=None, final=None):
    return {
        "schema_version": 2,
        "input_file": str(input_file.resolve()),
        "last_completed_step": last,
        "current_step": current,
        "pipeline_info": {"final_srt": str(final) if final else ""},
    }


def test_resume_matrix_and_backward_artifact_ladder(tmp_path):
    media = tmp_path / "clip.mp4"
    media.touch()
    artifacts = tmp_path / "clip_output"
    artifacts.mkdir()
    paths = artifact_paths(artifacts)
    final = tmp_path / "clip.srt"

    assert resolve_start_step(None, artifacts, "translate", input_file=media, final_srt=final) == FRESH
    old = {"schema_version": 1, "input_file": str(media.resolve())}
    assert resolve_start_step(old, artifacts, "translate", input_file=media, final_srt=final) == FRESH

    paths["audio"].touch()
    paths["stt"].write_text('{"text": "x"}')
    paths["punctuate"].write_text('{"text": "x"}')
    paths["align"].write_text("[]")
    state = _state(media, last="align", final=final)
    assert resolve_start_step(state, artifacts, "translate", input_file=media, final_srt=final) == "merge"

    state["last_completed_step"] = "merge"
    paths["merge"].write_text("srt")
    assert resolve_start_step(state, artifacts, "translate", input_file=media, final_srt=final) == "translate"
    assert resolve_start_step(state, artifacts, "merge", input_file=media, final_srt=final) == "merge"
    final.touch()
    assert resolve_start_step(state, artifacts, "merge", input_file=media, final_srt=final) == DONE


def test_interrupted_step_is_rerun_or_falls_back(tmp_path):
    media = tmp_path / "clip.wav"
    media.touch()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifact_paths(artifacts)["audio"].touch()
    state = _state(media, current="align", final=tmp_path / "clip.srt")
    assert resolve_start_step(state, artifacts, "translate", input_file=media) == "stt"


def test_pipeline_no_translate_outputs_final_and_cleans(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    artifacts = tmp_path / "custom-artifacts"
    final = tmp_path / "elsewhere" / "result.srt"

    def fake_audio(source, destination):
        destination.write_bytes(b"wav")
        return destination

    monkeypatch.setattr("mlx_srt.pipeline.audio.extract_audio", fake_audio)
    monkeypatch.setattr("mlx_srt.pipeline.stt.transcribe", lambda *args: {"text": "Bonjour monde"})
    monkeypatch.setattr(
        "mlx_srt.pipeline.align_stage.align",
        lambda *args: [
            {"text": "Bonjour", "start": 0, "end": 0.5},
            {"text": "monde", "start": 0.5, "end": 1},
        ],
    )
    config = load_config()
    config = replace(config, punctuation=replace(config.punctuation, enabled=False))
    result = run_pipeline(
        media, config, output_dir=artifacts, srt_output=final,
        no_translate=True, use_lock=False,
    )
    assert result == final.resolve()
    assert final.read_text(encoding="utf-8").endswith("Bonjour monde\n")
    assert not artifacts.exists()


def test_cleanup_preserves_unrelated_output_contents(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.touch()
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep")
    monkeypatch.setattr("mlx_srt.pipeline.audio.extract_audio", lambda source, destination: destination.write_bytes(b"wav"))
    monkeypatch.setattr("mlx_srt.pipeline.stt.transcribe", lambda *args: {"text": "Bonjour"})
    monkeypatch.setattr(
        "mlx_srt.pipeline.align_stage.align",
        lambda *args: [{"text": "Bonjour", "start": 0, "end": 1}],
    )
    config = load_config()
    config = replace(config, punctuation=replace(config.punctuation, enabled=False))
    run_pipeline(media, config, output_dir=tmp_path, no_translate=True, use_lock=False)
    assert unrelated.read_text() == "keep"
    assert media.exists()


def test_pipeline_keeps_state_and_auto_resumes_translate(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.touch()
    artifacts = tmp_path / "clip_output"
    final = tmp_path / "clip.srt"
    artifacts.mkdir()
    paths = artifact_paths(artifacts)
    paths["merge"].write_text("1\n00:00:00,000 --> 00:00:01,000\nBonjour\n", encoding="utf-8")
    state = _state(media, last="merge", final=final)
    (artifacts / "state.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("mlx_srt.pipeline.translate.translate_srt", lambda *args, **kwargs: "bilingual\n")
    config = load_config()
    result = run_pipeline(media, config, output_dir=artifacts, keep_artifacts=True, use_lock=False)
    assert result == final.resolve()
    assert final.read_text() == "bilingual\n"
    saved = json.loads((artifacts / "state.json").read_text())
    assert saved["last_completed_step"] == "translate"


def test_explicit_resume_without_state_errors(tmp_path):
    media = tmp_path / "clip.wav"
    media.touch()
    with pytest.raises(RuntimeError, match="no compatible resumable state"):
        run_pipeline(media, load_config(), resume=True, use_lock=False)
