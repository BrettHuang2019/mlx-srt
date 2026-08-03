"""Punctuate a flat transcript with kredor/punctuate-all."""

from __future__ import annotations

import re

TRAILING_PUNCT_RE = re.compile(r"[,.?:;\-…!]+$")
DOUBLE_SPACE_RE = re.compile(r"\s+")
WORD_CHARS = r"0-9A-Za-zÀ-ÖØ-öø-ÿŒœÆæ"
WORD_RE = re.compile(rf"[{WORD_CHARS}]+(?:['’\-][{WORD_CHARS}]+)*")
LABEL_TO_PUNCT = {"0": "", ".": ".", ",": ",", "?": "?", "-": "-", ":": ":"}
# Verbs of speech mislabelled with a colon; the colon becomes a space instead.
SPEECH_VERBS = (
    "je dis", "je lui dis", "tu dis", "il dit", "elle dit", "on dit", "me dit",
    "m'a dit", "me répond", "m'a répondu", "elle répond", "il répond",
    "je réponds", "demande", "demandez", "ajoute", "ajouté",
)
SPEECH_COLON_RE = re.compile(rf"\b({'|'.join(SPEECH_VERBS)}):\s+([a-zà-ÿ])", re.I)


def load_classifier(model_id: str):
    from transformers import pipeline

    return pipeline("token-classification", model=model_id, aggregation_strategy="first")


def normalize_label(raw_label: str) -> str:
    if raw_label in LABEL_TO_PUNCT:
        return raw_label
    if raw_label.startswith("LABEL_"):
        return raw_label.split("_", 1)[1]
    return raw_label


def compute_word_spans(words: list[str]) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for word in words:
        spans.append((cursor, cursor + len(word)))
        cursor += len(word) + 1
    return spans


def find_word_index(spans: list[tuple[int, int]], start: int, end: int) -> int | None:
    return next(
        (i for i, (word_start, word_end) in enumerate(spans)
         if start < word_end and end > word_start),
        None,
    )


def normalize_output_text(text: str) -> str:
    text = re.sub(r"([,.?:;!]){2,}", lambda match: match.group(0)[-1], text)
    text = re.sub(r"([,.?:;!])(?:\s+\1)+", r"\1", text)
    text = re.sub(r",\.|\.,", ".", text)
    text = re.sub(r"\?\.|\.\?", "?", text)
    text = re.sub(r":\.", ":", text)
    text = re.sub(r":\s+(pas|oui|non|ok|bah|bon)\.", r" \1.", text, flags=re.I)
    text = SPEECH_COLON_RE.sub(lambda match: f"{match.group(1)} {match.group(2)}", text)
    text = re.sub(r"\s+([,.?:;!])", r"\1", text)
    text = re.sub(r"([,.?:;!])([^\s])", r"\1 \2", text)
    text = re.sub(
        r"(^|(?<=[.!?]\s))([a-zà-öø-ÿ])",
        lambda match: match.group(1) + match.group(2).upper(),
        text,
    )
    return DOUBLE_SPACE_RE.sub(" ", text).strip()


def punctuate_chunk(classifier, words: list[str]) -> str:
    sanitized = [TRAILING_PUNCT_RE.sub("", word) or word for word in words]
    spans = compute_word_spans(sanitized)
    labels = ["0"] * len(words)
    for item in classifier(" ".join(sanitized)):
        label = normalize_label(item.get("entity_group") or item.get("entity") or "0")
        index = find_word_index(spans, int(item.get("start", 0)), int(item.get("end", 0)))
        if label in LABEL_TO_PUNCT and index is not None:
            labels[index] = label
    return normalize_output_text(
        " ".join(word + LABEL_TO_PUNCT[label] for word, label in zip(words, labels, strict=True))
    )


def find_second_to_last_sentence_end(text: str) -> int | None:
    collapsed = re.sub(r"\.{3}|…", "\x00", text)
    positions = [match.end() for match in re.finditer(r"[.!?\x00]", collapsed)]
    return positions[-2] if len(positions) >= 2 else None


def punctuate_text(
    text: str,
    classifier=None,
    *,
    model_id: str,
    chunk_words: int,
) -> str:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    words = WORD_RE.findall(text)
    if not words:
        return ""
    classifier = classifier or load_classifier(model_id)
    output = []
    pos = 0
    while pos < len(words):
        chunk = words[pos : pos + chunk_words]
        punctuated = punctuate_chunk(classifier, chunk)
        advance = len(chunk)
        accepted = punctuated
        if pos + chunk_words < len(words):
            split = find_second_to_last_sentence_end(punctuated)
            if split is not None:
                accepted = punctuated[:split].rstrip()
                rollback = len(WORD_RE.findall(punctuated[split:]))
                advance = max(1, len(chunk) - rollback)
        output.append(accepted)
        pos += advance
    return normalize_output_text(" ".join(output))


def punctuate(
    text: str,
    classifier=None,
    *,
    model_id: str,
    chunk_words: int,
) -> dict[str, str]:
    return {"text": punctuate_text(text, classifier, model_id=model_id, chunk_words=chunk_words)}
