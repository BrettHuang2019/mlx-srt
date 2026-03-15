import argparse
import json
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL_ID = "kredor/punctuate-all"
DEFAULT_INPUT_PATH = Path("whisper_output2.json")
DEFAULT_CHUNK_WORDS = 180
DEFAULT_PUNCTUATION_CONFIG = {
    "enabled": True,
    "model_path": DEFAULT_MODEL_ID,
    "chunk_words": DEFAULT_CHUNK_WORDS,
}

TRAILING_PUNCT_RE = re.compile(r"[,.?:;\-…!]+$")
DOUBLE_SPACE_RE = re.compile(r"\s+")
WORD_CHARS = r"0-9A-Za-zÀ-ÖØ-öø-ÿŒœÆæ"
WORD_RE = re.compile(rf"[{WORD_CHARS}]+(?:['’\-][{WORD_CHARS}]+)*")
TOKEN_RE = re.compile(
    rf"[{WORD_CHARS}]+(?:['’\-][{WORD_CHARS}]+)*|\s+|[^\w\s]",
    re.UNICODE,
)
PUNCT_SEQUENCE_RE = re.compile(r"[,.?:;!…-]+")

LABEL_TO_PUNCT = {
    "0": "",
    ".": ".",
    ",": ",",
    "?": "?",
    "-": "-",
    ":": ":",
}


class PunctuationAlignmentError(ValueError):
    """Raised when punctuated output cannot be mapped back onto Whisper segments."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}")


def load_classifier(model_id: str = DEFAULT_MODEL_ID):
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is not installed. Install with: pip install transformers")
    log(f"Loading model: {model_id}")
    return pipeline(
        "token-classification",
        model=model_id,
        aggregation_strategy="first",
    )


def normalize_label(raw_label: str) -> str:
    if raw_label in LABEL_TO_PUNCT:
        return raw_label
    if raw_label.startswith("LABEL_"):
        label_id = raw_label.split("_", 1)[1]
        if label_id in LABEL_TO_PUNCT:
            return label_id
    return raw_label


def compute_word_spans(words: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        start = cursor
        end = start + len(word)
        spans.append((start, end))
        cursor = end + 1
    return spans


def find_word_index(spans: list[tuple[int, int]], start: int, end: int) -> int | None:
    for index, (word_start, word_end) in enumerate(spans):
        if start < word_end and end > word_start:
            return index
    return None


def sanitize_word(word: str) -> str:
    cleaned = TRAILING_PUNCT_RE.sub("", word)
    return cleaned or word


def normalize_output_text(text: str) -> str:
    normalized = text
    normalized = re.sub(r"([,.?:;!]){2,}", lambda match: match.group(0)[-1], normalized)
    normalized = re.sub(r"([,.?:;!])(?:\s+\1)+", r"\1", normalized)
    normalized = re.sub(r",\.", ".", normalized)
    normalized = re.sub(r"\.,", ".", normalized)
    normalized = re.sub(r"\?\.", "?", normalized)
    normalized = re.sub(r"\.\?", "?", normalized)
    normalized = re.sub(r":\.", ":", normalized)
    normalized = re.sub(r":\s+(pas|oui|non|ok|bah|bon)\.", r" \1.", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\b(je dis|je lui dis|tu dis|il dit|elle dit|on dit|me dit|m'a dit|me répond|m'a répondu|elle répond|il répond|je réponds|demande|demandez|ajoute|ajouté):\s+([a-zà-ÿ])",
        lambda match: f"{match.group(1)} {match.group(2)}",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+([,.?:;!])", r"\1", normalized)
    normalized = re.sub(r"([,.?:;!])([^\s])", r"\1 \2", normalized)
    normalized = re.sub(
        r"(^|(?<=[.!?]\s))([a-zà-öø-ÿ])",
        lambda match: match.group(1) + match.group(2).upper(),
        normalized,
    )
    normalized = DOUBLE_SPACE_RE.sub(" ", normalized)
    return normalized.strip()


def find_second_to_last_sentence_end(text: str) -> int | None:
    collapsed = re.sub(r"\.{3}|…", "\x00", text)
    positions = [match.end() for match in re.finditer(r"[.!?\x00]", collapsed)]
    if len(positions) < 2:
        return None
    return positions[-2]


def tail_word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def punctuate_chunk(classifier, words: list[str]) -> dict:
    sanitized_words = [sanitize_word(word) for word in words]
    chunk_text = " ".join(sanitized_words)
    spans = compute_word_spans(sanitized_words)
    predictions = classifier(chunk_text)
    labels_by_word: list[str] = ["0"] * len(words)
    score_by_word: list[float | None] = [None] * len(words)

    for item in predictions:
        label = normalize_label(item.get("entity_group") or item.get("entity") or "0")
        if label not in LABEL_TO_PUNCT:
            continue
        start = int(item.get("start", 0))
        end = int(item.get("end", start))
        word_index = find_word_index(spans, start, end)
        if word_index is None:
            continue
        labels_by_word[word_index] = label
        score_by_word[word_index] = float(item.get("score", 0.0))

    output_parts: list[str] = []
    punctuation_counts: Counter[str] = Counter()
    for index, word in enumerate(words):
        punct = LABEL_TO_PUNCT[labels_by_word[index]]
        if punct:
            punctuation_counts[punct] += 1
            output_parts.append(f"{word}{punct}")
        else:
            output_parts.append(word)

    assigned_scores = [score for score in score_by_word if score is not None]
    raw_output_text = " ".join(output_parts)
    normalized_output_text = normalize_output_text(raw_output_text)
    return {
        "input_text": chunk_text,
        "raw_output_text": raw_output_text,
        "normalized_output_text": normalized_output_text,
        "word_count": len(words),
        "prediction_count": len(predictions),
        "assigned_prediction_count": sum(1 for score in score_by_word if score is not None),
        "avg_assigned_score": round(sum(assigned_scores) / len(assigned_scores), 6) if assigned_scores else None,
        "punctuation_counts": dict(sorted(punctuation_counts.items())),
    }


def punctuate_text_with_rollbacks(classifier, text: str, chunk_words: int = DEFAULT_CHUNK_WORDS) -> tuple[dict, list[dict]]:
    words = text.split()
    total_words = len(words)
    pos = 0
    chunk_index = 0
    chunk_reports: list[dict] = []
    raw_full_output_parts: list[str] = []
    normalized_full_output_parts: list[str] = []
    total_punctuation_counts: Counter[str] = Counter()

    while pos < total_words:
        chunk_index += 1
        chunk = words[pos:pos + chunk_words]
        is_last = pos + chunk_words >= total_words
        chunk_report = punctuate_chunk(classifier, chunk)

        split_pos = None
        tail = ""
        rollback = 0
        naive_tail_words = 0
        decision = ""
        accepted_output = chunk_report["normalized_output_text"]
        advance = len(chunk)

        if is_last:
            decision = "last_chunk_accept_all"
        else:
            split_pos = find_second_to_last_sentence_end(chunk_report["normalized_output_text"])
            if split_pos is None:
                decision = "no_boundary_accept_all"
            else:
                accepted_output = chunk_report["normalized_output_text"][:split_pos].rstrip()
                tail = chunk_report["normalized_output_text"][split_pos:].strip()
                naive_tail_words = len(tail.split())
                rollback = tail_word_count(tail)
                advance = max(1, len(chunk) - rollback)
                decision = "rollback_tail_after_second_to_last_boundary"

        read_word_end = pos + len(chunk) - 1
        next_pos = pos + advance
        accepted_word_end = next_pos - 1
        processed_at = datetime.now().isoformat(timespec="seconds")
        log(
            f"Processing chunk {chunk_index} "
            f"(accepted words {pos}-{accepted_word_end}, "
            f"read {pos}-{read_word_end}, rollback={rollback}, last={is_last})"
        )
        chunk_report["chunk"] = chunk_index
        chunk_report["processed_at"] = processed_at
        chunk_report["word_start"] = pos
        chunk_report["word_end"] = accepted_word_end
        chunk_report["accepted_output"] = accepted_output
        chunk_report["calculation"] = {
            "is_last_chunk": is_last,
            "decision": decision,
            "split_pos": split_pos,
            "tail": tail,
            "naive_tail_word_count": naive_tail_words,
            "tail_word_count": rollback,
            "read_word_end": read_word_end,
            "advance": advance,
            "next_word_start": next_pos,
        }
        chunk_reports.append(chunk_report)
        raw_full_output_parts.append(chunk_report["raw_output_text"])
        normalized_full_output_parts.append(accepted_output)
        total_punctuation_counts.update(chunk_report["punctuation_counts"])
        pos = next_pos

    return {
        "input_word_count": total_words,
        "chunk_words": chunk_words,
        "total_chunks": len(chunk_reports),
        "total_punctuation_counts": dict(sorted(total_punctuation_counts.items())),
        "raw_full_result": normalize_output_text(" ".join(raw_full_output_parts)),
        "normalized_full_result": normalize_output_text(" ".join(normalized_full_output_parts)),
    }, chunk_reports


def normalize_word(text: str) -> str:
    text = text.lower().replace("’", "'")
    return re.sub(r"[^0-9a-zà-öø-ÿœæ']", "", text)


def substitution_cost(left: str, right: str) -> float:
    if left == right:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio()
    if ratio >= 0.92:
        return 0.15
    if ratio >= 0.80:
        return 0.45
    if ratio >= 0.68:
        return 0.8
    return 1.5


def extract_segment_words(segments: list[dict]) -> list[list[dict]]:
    segment_words: list[list[dict]] = []
    global_index = 0
    for segment_index, segment in enumerate(segments):
        words: list[dict] = []
        for match in WORD_RE.finditer(segment["text"]):
            word_text = match.group(0)
            words.append(
                {
                    "text": word_text,
                    "norm": normalize_word(word_text),
                    "start": match.start(),
                    "end": match.end(),
                    "segment_index": segment_index,
                    "global_index": global_index,
                }
            )
            global_index += 1
        segment_words.append(words)
    return segment_words


def tokenize_punctuated_text(text: str) -> tuple[list[dict], list[str]]:
    tokens: list[dict] = []
    punct_words: list[str] = []
    word_index = 0

    for match in TOKEN_RE.finditer(text):
        token_text = match.group(0)
        if token_text.isspace():
            tokens.append({"text": token_text, "kind": "space"})
            continue
        if WORD_RE.fullmatch(token_text):
            tokens.append({"text": token_text, "kind": "word", "word_index": word_index})
            punct_words.append(normalize_word(token_text))
            word_index += 1
            continue
        tokens.append({"text": token_text, "kind": "punct"})

    return tokens, punct_words


def collapse_punctuation_sequence(text: str) -> str:
    if not text:
        return ""
    normalized = normalize_output_text(f"x{text}")
    suffix = normalized[1:]
    matches = PUNCT_SEQUENCE_RE.findall(suffix)
    if not matches:
        return ""
    sequence = matches[-1]
    return sequence[-1] if sequence else ""


def lowercase_segment_start(text: str) -> str:
    return re.sub(
        r"^([^A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]*)([A-ZÀ-ÖØ-Þ])",
        lambda match: match.group(1) + match.group(2).lower(),
        text,
        count=1,
    )


def normalize_rebuilt_segment_text(text: str, *, capitalize_sentence_start: bool = True) -> str:
    normalized = text.strip()
    normalized = re.sub(
        r"([,.?:;!])(?:\s*[,.?:;!])+",
        lambda match: collapse_punctuation_sequence(re.sub(r"\s+", "", match.group(0))),
        normalized,
    )
    normalized = re.sub(r"\s+([,.?:;!])", r"\1", normalized)
    normalized = re.sub(r"([.!?…])([^\s])", r"\1 \2", normalized)
    if capitalize_sentence_start:
        normalized = re.sub(
            r"(^|(?<=[.!?…]\s))([a-zà-öø-ÿ])",
            lambda match: match.group(1) + match.group(2).upper(),
            normalized,
        )
    else:
        normalized = re.sub(
            r"(?<=[.!?…]\s)([a-zà-öø-ÿ])",
            lambda match: match.group(1).upper(),
            normalized,
        )
        normalized = lowercase_segment_start(normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized


def segment_starts_new_sentence(previous_text: str | None) -> bool:
    if previous_text is None:
        return True

    trimmed = previous_text.rstrip()
    if not trimmed:
        return True

    return bool(re.search(r"[.!?…]\s*$", trimmed))


def align_words(original_words: list[dict], punct_words: list[str]) -> dict[int, int]:
    original_count = len(original_words)
    punctuated_count = len(punct_words)
    original_norms = [word["norm"] for word in original_words]

    if original_count == punctuated_count and original_norms == punct_words:
        return {index: index for index in range(original_count)}

    prefix = 0
    while prefix < original_count and prefix < punctuated_count and original_norms[prefix] == punct_words[prefix]:
        prefix += 1

    suffix = 0
    while (
        suffix < original_count - prefix
        and suffix < punctuated_count - prefix
        and original_norms[original_count - 1 - suffix] == punct_words[punctuated_count - 1 - suffix]
    ):
        suffix += 1

    if prefix == original_count and prefix == punctuated_count:
        return {index: index for index in range(original_count)}

    mapping: dict[int, int] = {index: index for index in range(prefix)}
    for offset in range(suffix):
        mapping[punctuated_count - suffix + offset] = original_count - suffix + offset

    original_slice = original_words[prefix:original_count - suffix if suffix else original_count]
    punct_slice = punct_words[prefix:punctuated_count - suffix if suffix else punctuated_count]
    if not original_slice or not punct_slice:
        return mapping

    original_count = len(original_slice)
    punctuated_count = len(punct_slice)
    delete_cost = 1.0
    insert_cost = 1.0

    back = [bytearray(punctuated_count + 1) for _ in range(original_count + 1)]
    prev = [float(j) * insert_cost for j in range(punctuated_count + 1)]

    for i in range(1, original_count + 1):
        curr = [0.0] * (punctuated_count + 1)
        curr[0] = float(i) * delete_cost
        back[i][0] = 1
        left = original_slice[i - 1]["norm"]
        for j in range(1, punctuated_count + 1):
            right = punct_slice[j - 1]
            diag_cost = prev[j - 1] + substitution_cost(left, right)
            up_cost = prev[j] + delete_cost
            left_cost = curr[j - 1] + insert_cost
            best = diag_cost
            direction = 0
            if up_cost < best:
                best = up_cost
                direction = 1
            if left_cost < best:
                best = left_cost
                direction = 2
            curr[j] = best
            back[i][j] = direction
        prev = curr

    i = original_count
    j = punctuated_count
    while i > 0 or j > 0:
        direction = back[i][j]
        if i > 0 and j > 0 and direction == 0:
            cost = substitution_cost(original_slice[i - 1]["norm"], punct_slice[j - 1])
            if cost <= 0.8:
                mapping[prefix + j - 1] = prefix + i - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or direction == 1):
            i -= 1
        else:
            j -= 1

    return mapping


def align_words_by_segment(
    segment_words: list[list[dict]],
    punct_words: list[str],
    lookahead: int = 12,
) -> tuple[dict[int, int], dict]:
    mapping: dict[int, int] = {}
    punct_cursor = 0
    exact_segment_matches = 0
    fallback_segment_matches = 0

    for words in segment_words:
        if not words:
            continue

        expected_count = len(words)
        original_norms = [word["norm"] for word in words]
        remaining = len(punct_words) - punct_cursor
        if remaining <= 0:
            break

        exact_start: int | None = None
        max_start = min(len(punct_words) - expected_count, punct_cursor + lookahead)
        for start in range(punct_cursor, max_start + 1):
            if punct_words[start:start + expected_count] == original_norms:
                exact_start = start
                break

        if exact_start is not None:
            exact_segment_matches += 1
            for offset, word in enumerate(words):
                mapping[exact_start + offset] = word["global_index"]
            punct_cursor = exact_start + expected_count
            continue

        window_end = min(len(punct_words), punct_cursor + expected_count + lookahead)
        local_mapping = align_words(words, punct_words[punct_cursor:window_end])
        if local_mapping:
            fallback_segment_matches += 1
            mapped_punct_indices = sorted(local_mapping)
            for punct_index, original_index in local_mapping.items():
                mapping[punct_cursor + punct_index] = words[original_index]["global_index"]
            punct_cursor += mapped_punct_indices[-1] + 1

    stats = {
        "exact_segment_matches": exact_segment_matches,
        "fallback_segment_matches": fallback_segment_matches,
        "unaligned_segments": sum(1 for words in segment_words if words) - exact_segment_matches - fallback_segment_matches,
    }
    return mapping, stats


def rebuild_segments(segments: list[dict], punctuated_text: str) -> tuple[list[dict], dict]:
    segment_words = extract_segment_words(segments)
    original_words = [word for words in segment_words for word in words]
    tokens, punct_words = tokenize_punctuated_text(punctuated_text)
    word_mapping, alignment_stats = align_words_by_segment(segment_words, punct_words)
    matched_words = 0
    unmatched_output_words = 0
    suggested_punct_by_word: dict[int, str] = {}

    word_token_positions = [index for index, token in enumerate(tokens) if token["kind"] == "word"]
    for position, token_index in enumerate(word_token_positions):
        token = tokens[token_index]
        original_index = word_mapping.get(token["word_index"])
        if original_index is None:
            unmatched_output_words += 1
            continue
        matched_words += 1
        next_word_token_index = (
            word_token_positions[position + 1] if position + 1 < len(word_token_positions) else len(tokens)
        )
        punct_text = "".join(
            item["text"]
            for item in tokens[token_index + 1:next_word_token_index]
            if item["kind"] == "punct"
        )
        collapsed = collapse_punctuation_sequence(punct_text)
        if collapsed:
            suggested_punct_by_word[original_index] = collapsed

    rebuilt_segments: list[dict] = []
    previous_rebuilt_text: str | None = None
    for index, segment in enumerate(segments):
        original_text = segment["text"].strip()
        words = segment_words[index]
        capitalize_sentence_start = segment_starts_new_sentence(previous_rebuilt_text)
        if not words:
            rebuilt_text = normalize_rebuilt_segment_text(
                original_text,
                capitalize_sentence_start=capitalize_sentence_start,
            )
        else:
            rebuilt_parts: list[str] = []
            cursor = 0
            for word_position, word in enumerate(words):
                rebuilt_parts.append(original_text[cursor:word["end"]])
                next_start = words[word_position + 1]["start"] if word_position + 1 < len(words) else len(original_text)
                gap = original_text[word["end"]:next_start]
                suggested_punct = suggested_punct_by_word.get(word["global_index"], "")
                if suggested_punct and not PUNCT_SEQUENCE_RE.search(gap):
                    gap = f"{suggested_punct}{gap}"
                rebuilt_parts.append(gap)
                cursor = next_start
            rebuilt_text = normalize_rebuilt_segment_text(
                "".join(rebuilt_parts),
                capitalize_sentence_start=capitalize_sentence_start,
            )
        rebuilt_segments.append(
            {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "original_text": segment["text"],
                "punctuated_text": rebuilt_text,
            }
        )
        previous_rebuilt_text = rebuilt_text

    stats = {
        "segment_count": len(segments),
        "original_word_count": len(original_words),
        "punctuated_word_count": len(punct_words),
        "matched_word_count": matched_words,
        "unmatched_output_words": unmatched_output_words,
        **alignment_stats,
    }
    return rebuilt_segments, stats


def process_whisper_payload(
    payload: dict,
    classifier=None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> tuple[dict, dict, list[dict]]:
    if classifier is None:
        classifier = load_classifier(model_id)

    source_text = payload["text"].strip()
    segments = payload["segments"]
    punctuation_summary, chunk_reports = punctuate_text_with_rollbacks(
        classifier,
        source_text,
        chunk_words=chunk_words,
    )
    punctuated_segments, mapping_stats = rebuild_segments(
        segments,
        punctuation_summary["normalized_full_result"],
    )

    mapped_segments_payload = {
        "model": model_id,
        "timestamp": datetime.now().isoformat(),
        "source_text": source_text,
        "punctuated_text": punctuation_summary["normalized_full_result"],
        "mapping_stats": mapping_stats,
        "segments": punctuated_segments,
    }
    report = {
        "model": model_id,
        "timestamp": datetime.now().isoformat(),
        "punctuation": punctuation_summary,
        "mapping": mapping_stats,
        "chunks": chunk_reports,
    }
    return mapped_segments_payload, report, punctuated_segments


def get_punctuation_runtime_config(whisper_config: dict) -> dict:
    punctuation_config = DEFAULT_PUNCTUATION_CONFIG.copy()
    punctuation_config.update(whisper_config.get("punctuation", {}) or {})
    return punctuation_config


def apply_punctuation_to_payload(
    payload: dict,
    punctuation_config: dict,
    calculate_punctuation_ratio,
) -> tuple[dict, dict]:
    if not punctuation_config.get("enabled", True):
        raise RuntimeError("Punctuation step is disabled in config.")

    model_id = punctuation_config["model_path"]
    chunk_words = punctuation_config.get("chunk_words", DEFAULT_CHUNK_WORDS)
    mapped_segments_payload, report, _ = process_whisper_payload(
        payload,
        model_id=model_id,
        chunk_words=chunk_words,
    )

    mapped_segments = mapped_segments_payload["segments"]
    final_segments = [
        {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["punctuated_text"],
        }
        for segment in mapped_segments
    ]
    final_result = {
        "text": mapped_segments_payload["punctuated_text"],
        "segments": final_segments,
    }

    punctuation_summary = report["punctuation"]
    mapping_stats = report["mapping"]
    metadata = {
        "model_path": model_id,
        "chunk_count": punctuation_summary["total_chunks"],
        "chunk_words": punctuation_summary["chunk_words"],
        "input_punctuation_ratio": calculate_punctuation_ratio(payload["text"]),
        "output_punctuation_ratio": calculate_punctuation_ratio(final_result["text"]),
        "mapping_stats": mapping_stats,
        "punctuation_summary": punctuation_summary,
    }
    return final_result, metadata


def write_outputs(
    *,
    mapped_segments_payload: dict,
    report: dict,
    chunk_reports: list[dict],
    source_file: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapped_segments_path = output_dir / f"whisper_punctuated_segments_kredor_{timestamp}.json"
    report_path = output_dir / f"whisper_punctuation_report_kredor_{timestamp}.json"
    txt_path = output_dir / f"whisper_punctuation_report_kredor_{timestamp}.txt"

    mapped_segments_payload = {
        **mapped_segments_payload,
        "source_file": str(source_file),
    }
    report = {
        **report,
        "source_file": str(source_file),
        "mapped_segments_file": str(mapped_segments_path),
    }

    mapped_segments_path.write_text(json.dumps(mapped_segments_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    punctuation_summary = report["punctuation"]
    mapping_stats = report["mapping"]
    segments = mapped_segments_payload["segments"]

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Whisper Kredor Punctuation Report\n")
        handle.write(f"Model              : {report['model']}\n")
        handle.write(f"Source             : {source_file}\n")
        handle.write(f"Input words        : {punctuation_summary['input_word_count']}\n")
        handle.write(f"Chunk words        : {punctuation_summary['chunk_words']}\n")
        handle.write(f"Total chunks       : {punctuation_summary['total_chunks']}\n")
        handle.write(f"Mapped segments    : {mapping_stats['segment_count']}\n")
        handle.write(f"Matched words      : {mapping_stats['matched_word_count']}\n")
        handle.write(f"Unmatched out words: {mapping_stats['unmatched_output_words']}\n")
        handle.write(f"Date               : {report['timestamp']}\n")
        handle.write("=" * 80 + "\n\n")

        handle.write("TOTAL PUNCTUATION COUNTS:\n")
        for punct, count in punctuation_summary["total_punctuation_counts"].items():
            handle.write(f"{punct}: {count}\n")
        handle.write("\n")

        for chunk_report in chunk_reports:
            calc = chunk_report["calculation"]
            handle.write("─" * 80 + "\n")
            handle.write(
                f"CHUNK {chunk_report['chunk']} [{chunk_report['processed_at']}] "
                f"(accepted words {chunk_report['word_start']}–{chunk_report['word_end']}, "
                f"read through {calc['read_word_end']})\n"
            )
            handle.write("─" * 80 + "\n")
            handle.write("INPUT:\n")
            handle.write(chunk_report["input_text"] + "\n\n")
            handle.write("NORMALIZED OUTPUT:\n")
            handle.write(chunk_report["normalized_output_text"] + "\n\n")
            handle.write("ACCEPTED OUTPUT:\n")
            handle.write(chunk_report["accepted_output"] + "\n\n")
            handle.write("CHUNK ADVANCE CALCULATION:\n")
            handle.write(f"decision              : {calc['decision']}\n")
            handle.write(f"is_last_chunk         : {calc['is_last_chunk']}\n")
            handle.write(f"split_pos             : {calc['split_pos']}\n")
            handle.write(f"naive_tail_word_count : {calc['naive_tail_word_count']}\n")
            handle.write(f"tail_word_count       : {calc['tail_word_count']}\n")
            handle.write(f"read_word_end         : {calc['read_word_end']}\n")
            handle.write(f"advance               : {calc['advance']}\n")
            handle.write(f"next_word_start       : {calc['next_word_start']}\n")
            handle.write("tail:\n")
            handle.write((calc["tail"] or "<empty>") + "\n\n")

        handle.write("=" * 80 + "\n")
        handle.write("FULL PUNCTUATED TEXT:\n")
        handle.write("=" * 80 + "\n")
        handle.write(punctuation_summary["normalized_full_result"] + "\n\n")

        handle.write("=" * 80 + "\n")
        handle.write("SEGMENT PREVIEW:\n")
        handle.write("=" * 80 + "\n")
        for segment in segments[:8]:
            handle.write(f"[{segment['id']}] {segment['start']:.2f}-{segment['end']:.2f}\n")
            handle.write(segment["punctuated_text"] + "\n\n")

    return mapped_segments_path, report_path, txt_path


def process_whisper_file(
    input_path: Path = DEFAULT_INPUT_PATH,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    output_dir: Path | None = None,
):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    classifier = load_classifier(model_id)
    mapped_segments_payload, report, _ = process_whisper_payload(
        payload,
        classifier=classifier,
        model_id=model_id,
        chunk_words=chunk_words,
    )
    output_dir = output_dir or input_path.parent
    mapped_segments_path, report_path, txt_path = write_outputs(
        mapped_segments_payload=mapped_segments_payload,
        report=report,
        chunk_reports=report["chunks"],
        source_file=input_path,
        output_dir=output_dir,
    )
    log(f"Mapped segments saved to: {mapped_segments_path}")
    log(f"JSON report saved to: {report_path}")
    log(f"Text report saved to: {txt_path}")
    return {
        "mapped_segments_path": mapped_segments_path,
        "report_path": report_path,
        "txt_path": txt_path,
        "mapped_segments_payload": mapped_segments_payload,
        "report": report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Punctuate Whisper JSON output with kredor/punctuate-all.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to Whisper JSON payload.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated reports.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS, help="Words per punctuation chunk.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    process_whisper_file(
        input_path=args.input,
        model_id=args.model,
        chunk_words=args.chunk_words,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
