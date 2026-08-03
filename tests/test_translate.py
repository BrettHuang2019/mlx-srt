import json

import pytest

from mlx_srt.translate import (
    TranslationSettings,
    ValidationError,
    batch_translate,
    is_valid_translation,
    parse_and_validate,
    translate_srt,
)


def test_translation_acceptance_rules():
    assert is_valid_translation("你好", "bonjour")
    assert is_valid_translation("30,5", "30,5")
    assert is_valid_translation("Macron", "Macron")
    assert is_valid_translation("Different", "une phrase longue ici")
    assert not is_valid_translation("", "bonjour")
    assert not is_valid_translation("une phrase longue ici", "une phrase longue ici")


def test_validation_requires_ids_and_nonempty_translation():
    batch = [{"id": 1, "fr": "Bonjour"}]
    assert parse_and_validate('[{"id": 1, "zh": "你好"}]', batch)[0]["id"] == 1
    with pytest.raises(ValidationError):
        parse_and_validate('[{"id": 2, "zh": "你好"}]', batch)


def test_batch_recursively_splits_on_failure(monkeypatch):
    calls = []

    def fake_generate(prompt, settings):
        ids = [int(value) for value in __import__("re").findall(r'"id": (\d+)', prompt)]
        calls.append(ids)
        if len(ids) > 1:
            return "bad"
        return json.dumps([{"id": ids[0], "zh": "你好"}])

    monkeypatch.setattr("mlx_srt.translate.generate_text", fake_generate)
    settings = TranslationSettings("model", batch_size=2, max_retries=0)
    result = batch_translate(
        [{"id": 1, "fr": "Un"}, {"id": 2, "fr": "Deux"}],
        settings,
        "{context}\n{segments}",
    )
    assert result == {1: "你好", 2: "你好"}
    assert calls[0] == [1, 2]


def test_translate_srt_outputs_french_then_chinese(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{context}\n{segments}", encoding="utf-8")
    monkeypatch.setattr(
        "mlx_srt.translate.generate_text",
        lambda prompt, settings: '[{"id": 1, "zh": "你好"}]',
    )
    source = "1\n00:00:00,000 --> 00:00:01,000\nBonjour\n"
    result = translate_srt(source, settings=TranslationSettings("model"), prompt_file=prompt)
    assert "Bonjour\n你好" in result


def test_packaged_prompt_formats_json_examples(monkeypatch):
    prompt = __import__("mlx_srt.config", fromlist=["load_config"]).load_config().translate.prompt_file
    monkeypatch.setattr(
        "mlx_srt.translate.generate_text",
        lambda rendered, settings: '[{"id": 1, "zh": "你好"}]',
    )
    source = "1\n00:00:00,000 --> 00:00:01,000\nBonjour\n"
    assert "你好" in translate_srt(source, settings=TranslationSettings("model"), prompt_file=prompt)
