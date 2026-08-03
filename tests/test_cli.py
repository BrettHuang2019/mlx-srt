import json

from click.testing import CliRunner

from mlx_srt.cli import main


def test_bare_path_dispatches_to_run(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.touch()
    called = {}

    def fake_run(input_file, config, **kwargs):
        called["input"] = input_file
        return tmp_path / "clip.srt"

    monkeypatch.setattr("mlx_srt.cli.run_pipeline", fake_run)
    result = CliRunner().invoke(main, [str(media), "--no-translate"])
    assert result.exit_code == 0, result.output
    assert called["input"] == media


def test_command_name_wins_over_same_named_file(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        open("stt", "w").close()
        result = runner.invoke(main, ["stt"])
    assert result.exit_code != 0
    assert "Missing argument 'INPUT_FILE'" in result.output


def test_explicit_run_is_escape_hatch_for_collision(tmp_path, monkeypatch):
    media = tmp_path / "stt"
    media.touch()
    monkeypatch.setattr("mlx_srt.cli.run_pipeline", lambda *args, **kwargs: tmp_path / "stt.srt")
    result = CliRunner().invoke(main, ["run", str(media)])
    assert result.exit_code == 0


def test_merge_stdout_is_json_and_writes_nothing(tmp_path):
    words = tmp_path / "words.json"
    text = tmp_path / "text.json"
    words.write_text(json.dumps([{"text": "Bonjour", "start": 0, "end": 1}]))
    text.write_text(json.dumps({"text": "Bonjour."}))
    result = CliRunner().invoke(main, ["merge", "--words", str(words), "--text", str(text), "--stdout"])
    assert result.exit_code == 0, result.output
    assert "Bonjour." in json.loads(result.output)
