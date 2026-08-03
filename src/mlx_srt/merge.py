"""Align punctuation to word timestamps and construct readable SRT cues."""

from __future__ import annotations

import re
from typing import TypeAlias

from .srt import Segment, format_srt_time, render_srt

MAX_CHARS = 100
MIN_CHARS = 30
MIN_DURATION = 1.0
SPLIT_WORDS = {"et", "mais", "donc", "parce", "bref", "enfin", "puis", "car", "or"}
DANGLING_WORDS = {
    "de", "du", "le", "la", "les", "un", "une", "et", "à", "en", "des", "se",
    "ce", "sa", "son", "ses", "au", "aux", "je", "tu", "il", "elle", "on", "nous",
    "vous", "ils", "elles", "me", "te", "ne", "qui", "que", "où",
}
Word: TypeAlias = tuple[str, float, float]


def strip_punctuation(word: str) -> str:
    return re.sub(r"[^\w\s]", "", word, flags=re.UNICODE).strip()


def sub_text(words: list[Word]) -> str:
    return " ".join(word for word, _, _ in words)


def sub_len(words: list[Word]) -> int:
    return len(sub_text(words))


def sub_duration(words: list[Word]) -> float:
    return words[-1][2] - words[0][1] if words else 0.0


def find_best_split(words: list[Word]) -> int | None:
    midpoint = len(sub_text(words)) // 2
    max_gap = max((words[i][1] - words[i - 1][2] for i in range(1, len(words))), default=0)
    best = None
    best_score = float("-inf")
    char_pos = 0
    for i in range(1, len(words)):
        char_pos += len(words[i - 1][0]) + 1
        gap = words[i][1] - words[i - 1][2]
        has_gap = gap > 0.1
        has_punct = bool(re.search(r"[,;:.!?]$", words[i - 1][0]))
        tier = 60 if has_gap and has_punct else 40 if has_gap else 20 if has_punct else 0
        score = -abs(char_pos - midpoint)
        score += gap / max_gap * 30 if max_gap > 0 else 0
        score += 20 if has_punct else 0
        score += tier
        score += 15 if strip_punctuation(words[i][0]).lower() in SPLIT_WORDS else 0
        score -= 25 if strip_punctuation(words[i - 1][0]).lower() in DANGLING_WORDS else 0
        score -= 50 if words[i - 1][0].rstrip(".,;:!?").isdigit() else 0
        if score > best_score:
            best, best_score = i, score
    return best


def split_long(subs: list[list[Word]], max_chars: int = MAX_CHARS) -> list[list[Word]]:
    result = []
    for words in subs:
        if sub_len(words) <= max_chars:
            result.append(words)
            continue
        split = find_best_split(words)
        if not split:
            result.append(words)
        else:
            result.extend(split_long([words[:split]], max_chars))
            result.extend(split_long([words[split:]], max_chars))
    return result


def merge_short(
    subs: list[list[Word]],
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    min_duration: float = MIN_DURATION,
) -> list[list[Word]]:
    result = []
    i = 0
    while i < len(subs):
        words = subs[i]
        if sub_len(words) < min_chars and sub_duration(words) < min_duration:
            if i + 1 < len(subs) and sub_len(words) + sub_len(subs[i + 1]) + 1 <= max_chars:
                result.append(words + subs[i + 1])
                i += 2
                continue
            if result and sub_len(result[-1]) + sub_len(words) + 1 <= max_chars:
                result[-1].extend(words)
                i += 1
                continue
        result.append(words)
        i += 1
    return result


def align_words(timestamps: list[dict], punct_text: str) -> list[Word]:
    aligned = []
    ts_index = 0
    for punct_word in punct_text.split():
        clean = strip_punctuation(punct_word).lower()
        found = next(
            (i for i in range(ts_index, min(ts_index + 3, len(timestamps)))
             if strip_punctuation(str(timestamps[i]["text"])).lower() == clean),
            None,
        )
        if found is not None:
            stamp = timestamps[found]
            aligned.append((punct_word, float(stamp["start"]), float(stamp["end"])))
            ts_index = found + 1
        else:
            fallback = aligned[-1][1] if aligned else 0.0
            aligned.append((punct_word, fallback, fallback))
    return aligned


def build_subtitles(
    aligned: list[Word],
    *,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    min_duration: float = MIN_DURATION,
) -> list[list[Word]]:
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for word in aligned:
        current.append(word)
        if re.search(r"[.!?]+\s*$", word[0]):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return merge_short(split_long(sentences, max_chars), max_chars, min_chars, min_duration)


def merge_srt(
    timestamps: list[dict],
    punct_text: str,
    *,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    min_duration: float = MIN_DURATION,
) -> str:
    groups = build_subtitles(
        align_words(timestamps, punct_text),
        max_chars=max_chars,
        min_chars=min_chars,
        min_duration=min_duration,
    )
    segments = [
        Segment(i, format_srt_time(words[0][1]), format_srt_time(words[-1][2]), sub_text(words))
        for i, words in enumerate(groups, 1)
    ]
    return render_srt(segments)
