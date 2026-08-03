"""Batch French SRT translation with strict response validation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .srt import format_bilingual_subtitles, parse_srt

_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_NUMBERS_RE = re.compile(r"^[\d\s,.\-:;]+$")


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TranslationSettings:
    model_path: str
    batch_size: int = 10
    max_tokens: int = 2048
    temperature: float = 0.0
    max_retries: int = 1
    retry_delay: float = 1.0


_model = None
_tokenizer = None
_loaded_path: str | None = None


def generate_text(prompt: str, settings: TranslationSettings) -> str:
    global _model, _tokenizer, _loaded_path
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    if _loaded_path != settings.model_path:
        _model, _tokenizer = load(settings.model_path)
        _loaded_path = settings.model_path
    formatted = _tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return generate(
        _model,
        _tokenizer,
        prompt=formatted,
        max_tokens=settings.max_tokens,
        sampler=make_sampler(settings.temperature),
        verbose=False,
    )


def is_valid_translation(zh: str, source: str = "") -> bool:
    zh = zh.strip()
    source = source.strip()
    if not zh:
        return False
    if _CHINESE_RE.search(zh):
        return zh != source
    if _NUMBERS_RE.fullmatch(zh):
        return True
    if len(source.split()) <= 3 and zh.lower() == source.lower():
        return True
    return zh.lower() != source.lower()


def _extract_json_array(raw: str) -> list:
    raw = raw.translate(_SMART_QUOTES)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValidationError("JSON parse failed") from None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValidationError("JSON parse failed") from exc
    if not isinstance(parsed, list):
        raise ValidationError("Expected JSON array")
    return parsed


def parse_and_validate(raw: str, batch: list[dict]) -> list[dict]:
    parsed = _extract_json_array(raw)
    if len(parsed) != len(batch):
        raise ValidationError(f"Length mismatch: expected {len(batch)}, got {len(parsed)}")
    for item, expected in zip(parsed, batch):
        if not isinstance(item, dict) or set(("id", "zh")) - item.keys():
            raise ValidationError(f"Missing id/zh keys: {item!r}")
        if not isinstance(item["id"], int) or item["id"] != expected["id"]:
            raise ValidationError(f"ID mismatch: expected {expected['id']}, got {item.get('id')!r}")
        if not isinstance(item["zh"], str) or not is_valid_translation(item["zh"], expected["fr"]):
            raise ValidationError(f"Invalid translation for id={item['id']}")
    return parsed


def _build_prompt(batch: list[dict], context: list[dict], template: str) -> str:
    return template.format(
        context="\n".join(item["fr"] for item in context),
        segments=json.dumps(batch, ensure_ascii=False),
    )


def _translate_batch(
    batch: list[dict],
    context: list[dict],
    settings: TranslationSettings,
    template: str,
) -> list[dict]:
    last_error: Exception | None = None
    prompt = _build_prompt(batch, context, template)
    for attempt in range(settings.max_retries + 1):
        if attempt:
            time.sleep(settings.retry_delay)
        try:
            return parse_and_validate(generate_text(prompt, settings), batch)
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def process_batch_recursive(
    batch: list[dict],
    context: list[dict],
    settings: TranslationSettings,
    template: str,
) -> list[dict]:
    try:
        return _translate_batch(batch, context, settings, template)
    except Exception:
        if len(batch) <= 1:
            raise
        midpoint = len(batch) // 2
        return process_batch_recursive(batch[:midpoint], context, settings, template) + process_batch_recursive(
            batch[midpoint:], context, settings, template
        )


def batch_translate(
    items: list[dict],
    settings: TranslationSettings,
    template: str,
    *,
    context_window: int = 3,
) -> dict[int, str]:
    translations = {}
    for start in range(0, len(items), settings.batch_size):
        batch = items[start : start + settings.batch_size]
        context = items[max(0, start - context_window) : start]
        for item in process_batch_recursive(batch, context, settings, template):
            translations[item["id"]] = item["zh"]
    return translations


def translate_srt(
    srt_content: str,
    *,
    settings: TranslationSettings,
    prompt_file: str | Path,
) -> str:
    segments = [segment for segment in parse_srt(srt_content) if not re.match(r"^\.{3}|^\s*$", segment.text)]
    segments.sort(key=lambda segment: segment.id)
    for new_id, segment in enumerate(segments, 1):
        segment.id = new_id
    items = [{"id": segment.id, "fr": segment.text} for segment in segments]
    template = Path(prompt_file).read_text(encoding="utf-8")
    translations = batch_translate(items, settings, template)
    return format_bilingual_subtitles(segments, translations)
