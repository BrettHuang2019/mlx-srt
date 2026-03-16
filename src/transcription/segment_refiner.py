

"""Segment refiner for improving Whisper transcription output."""

import re
from typing import Dict, Any, List, Tuple
from pathlib import Path

SENTENCE_END_PUNCTUATION = set(".!?。！？…")


def refine_segments(whisper_output: Dict[str, Any], output_dir: str = None) -> Dict[str, Any]:
    """Refine Whisper transcription segments with improved spacing, merging, and splitting.

    Args:
        whisper_output: Full Whisper JSON output with 'text' and 'segments' fields
        output_dir: Optional directory to save detailed merge reports

    Returns:
        Dict containing refined segments and processing statistics
    """
    # Initialize statistics
    stats = {
        "punctuation_fixed": 0,
        "segments_merged": 0,
        "segments_split": 0,
        "empty_segments_removed": 0,
        "repetitive_patterns_removed": 0,
        "leading_spaces_removed": 0,
        "duplicate_segments_removed": 0,
        "repeated_words_removed": 0,
        "punctuation_only_segments_removed": 0,
        "total_input_segments": 0,
        "total_output_segments": 0
    }

    # Extract segments
    original_segments = whisper_output.get("segments", [])
    stats["total_input_segments"] = len(original_segments)

    # Step 1: Filter out empty segments
    filtered_segments = [
        segment for segment in original_segments
        if segment.get("text", "").strip()
    ]
    stats["empty_segments_removed"] = len(original_segments) - len(filtered_segments)

    # Step 2: Remove segments with only punctuation/signs
    filtered_segments, punctuation_only_removed = _remove_punctuation_only_segments(filtered_segments)
    stats["punctuation_only_segments_removed"] = punctuation_only_removed

    # Step 3: Remove leading spaces from text
    filtered_segments, leading_spaces_removed = _remove_leading_spaces(filtered_segments)
    stats["leading_spaces_removed"] = leading_spaces_removed

    # Step 4: Remove duplicate segments occurring 4+ times
    filtered_segments, duplicate_segments_removed = _remove_duplicate_segments(filtered_segments)
    stats["duplicate_segments_removed"] = duplicate_segments_removed

    # Step 5: Fix punctuation spacing
    for segment in filtered_segments:
        original_text = segment.get("text", "")
        # Remove spaces before ! and ?
        fixed_text = re.sub(r'\s+([!?])', r'\1', original_text)
        fixed_text = _normalize_repeated_punctuation(fixed_text)
        if fixed_text != original_text:
            segment["text"] = fixed_text
            stats["punctuation_fixed"] += 1

    # Step 6: Remove repeated words occurring 4+ times in same segment
    filtered_segments, repeated_words_removed = _remove_repeated_words(filtered_segments)
    stats["repeated_words_removed"] = repeated_words_removed

    # Step 7: Remove repetitive word patterns (LLM artifacts)
    cleaned_segments, repetitive_removed = _remove_repetitive_patterns(filtered_segments)
    stats["repetitive_patterns_removed"] = repetitive_removed

    # Step 7.5: Remove repeated character-level patterns
    cleaned_segments, char_patterns_removed = _remove_repeated_char_patterns(cleaned_segments)
    stats["repetitive_patterns_removed"] += char_patterns_removed

    # Step 8: Merge fragmented sentences
    merged_segments, merge_count, merge_details = _merge_fragments(cleaned_segments)
    stats["segments_merged"] = merge_count

    # Generate merge report if output directory specified and merges occurred
    if output_dir and merge_details and merge_count > 0:
        _save_merge_report(merge_details, output_dir)

    # Step 9: Split long blocks
    final_segments, split_count, split_details = _split_long_blocks(merged_segments)
    stats["segments_split"] = split_count

    # Generate split report if output directory specified and splits occurred
    if output_dir and split_details and split_count > 0:
        _save_split_report(split_details, output_dir)

    stats["total_output_segments"] = len(final_segments)

    # Recombine text from refined segments
    combined_text = " ".join(segment["text"] for segment in final_segments)

    return {
        "text": combined_text,
        "segments": final_segments,
        "statistics": stats
    }


def _merge_fragments(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """Merge consecutive segments when the merge scoring rule passes."""
    if not segments:
        return [], 0, []

    merge_count = 0
    merge_details = []
    current_segments = [segment.copy() for segment in segments]

    while True:
        merged = []
        changed = False
        i = 0

        while i < len(current_segments):
            current = current_segments[i].copy()

            if i + 1 >= len(current_segments):
                merged.append(current)
                i += 1
                continue

            next_segment = current_segments[i + 1]
            if _should_merge_segments(current, next_segment):
                merged_segment = current.copy()
                merged_segment["text"] = _merge_segment_text(
                    current.get("text", ""),
                    next_segment.get("text", ""),
                )
                merged_segment["end"] = next_segment["end"]

                merge_count += 1
                merge_details.append({
                    "merge_id": merge_count,
                    "before_segments": [current.copy(), next_segment.copy()],
                    "after_segment": merged_segment.copy(),
                    "original_count": 2,
                    "merged_start": merged_segment["start"],
                    "merged_end": merged_segment["end"],
                    "original_texts": [
                        current.get("text", ""),
                        next_segment.get("text", ""),
                    ],
                    "merged_text": merged_segment["text"],
                })

                merged.append(merged_segment)
                changed = True
                i += 2
                continue

            merged.append(current)
            i += 1

        current_segments = merged
        if not changed:
            return current_segments, merge_count, merge_details


def _should_merge_segments(current: Dict[str, Any], next_segment: Dict[str, Any]) -> bool:
    """Return True when the configured score-based merge rule passes."""
    merged_duration = next_segment["end"] - current["start"]
    if merged_duration >= 8.0:
        return False

    current_text = current.get("text", "").strip()
    current_duration = current["end"] - current["start"]
    gap_to_next = next_segment["start"] - current["end"]

    score = 0
    if current_duration < 2.0:
        score += 2
    if current_text and current_text[-1] not in SENTENCE_END_PUNCTUATION:
        score += 1
    if len(current_text.split()) < 5:
        score += 1
    if gap_to_next < 0.1:
        score += 1

    return score >= 3


def _merge_segment_text(current_text: str, next_text: str) -> str:
    """Join adjacent segment text while preserving a leading continuation space."""
    has_leading_space = current_text.startswith(" ")
    merged_text = f"{current_text.strip()} {next_text.strip()}".strip()
    return f" {merged_text}" if has_leading_space else merged_text


def _split_long_blocks(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """Split segments containing multiple long sentences (>5 words each)."""
    split_segments = []
    split_count = 0
    split_details = []

    for segment in segments:
        text = segment.get("text", "").strip()

        # Check if text starts with » (continuation from previous subtitle)
        # If so, preserve it as a prefix and don't split on the first »
        starts_with_quote_end = text.startswith("»")
        if starts_with_quote_end:
            # Remove the leading » for splitting, will add back later
            text_without_leading_quote = text[1:].strip()
            # Split on sentence-ending punctuation and » (but not the leading one)
            sentences = re.split(r'([.!?»])\s*', text_without_leading_quote)
        else:
            sentences = re.split(r'([.!?»])\s*', text)

        # Track if we have » delimiters (before filtering)
        has_quote_delimiters = "»" in sentences

        # Reconstruct sentences with punctuation
        reconstructed_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                if sentence.strip():
                    # Check if this is a standalone »
                    if sentence.strip() == "»":
                        # Append to previous sentence
                        if reconstructed_sentences:
                            reconstructed_sentences[-1] += " »"
                    else:
                        reconstructed_sentences.append(sentence.strip())

        # Add back the leading » if it was removed
        if starts_with_quote_end and reconstructed_sentences:
            reconstructed_sentences[0] = "» " + reconstructed_sentences[0]

        # Check if we need to split
        # Split if: multiple sentences AND (all are >5 words OR we have » delimiters for avoiding long translations)
        should_split = (len(reconstructed_sentences) > 1 and
                        (all(len(sentence.split()) > 5 for sentence in reconstructed_sentences) or
                         (has_quote_delimiters and len(reconstructed_sentences) > 2)))

        if should_split:

            # Split into multiple segments
            total_duration = segment["end"] - segment["start"]
            split_count += 1
            original_segment = segment.copy()

            new_segments = []
            for i, sentence in enumerate(reconstructed_sentences):
                # Calculate proportional timing
                sentence_words = len(sentence.split())
                total_words = sum(len(s.split()) for s in reconstructed_sentences)

                new_start = segment["start"] + (total_duration * i / len(reconstructed_sentences))
                new_end = segment["start"] + (total_duration * (i + 1) / len(reconstructed_sentences))

                new_segment = {
                    "id": segment["id"] + i * 0.1,  # Slight offset to maintain uniqueness
                    "start": new_start,
                    "end": new_end,
                    "text": " " + sentence if i > 0 else sentence  # Add leading space for continuation
                }
                new_segments.append(new_segment)
                split_segments.append(new_segment)

            # Capture split details for reporting
            split_detail = {
                "split_id": split_count,
                "original_segment": original_segment,
                "split_segments": new_segments,
                "split_count": len(reconstructed_sentences),
                "original_start": original_segment["start"],
                "original_end": original_segment["end"],
                "original_text": original_segment.get("text", ""),
                "split_texts": [seg.get("text", "") for seg in new_segments],
                "split_sentences": reconstructed_sentences
            }
            split_details.append(split_detail)
        else:
            split_segments.append(segment)

    return split_segments, split_count, split_details


def _remove_repetitive_patterns(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove LLM artifacts where same word repeats excessively (>5 times)."""
    cleaned_segments = []
    repetitive_removed = 0

    for segment in segments:
        text = segment.get("text", "")

        # Check for repetitive word patterns
        words = text.split()
        cleaned_text = text
        segment_modified = False

        # Look for words that repeat more than 5 times consecutively or near-consecutively
        if words:
            i = 0
            while i < len(words):
                word = words[i].strip(" .!?,:;")
                if not word:
                    i += 1
                    continue

                # Count consecutive or near-consecutive repetitions of the same word
                count = 1
                j = i + 1

                while j < len(words):
                    next_word = words[j].strip(" .!?,:;")
                    if next_word.lower() == word.lower():
                        count += 1
                        j += 1
                    elif j < i + 3:  # Allow up to 2 different words between repetitions
                        j += 1
                    else:
                        break

                # If word repeats too many times, remove the entire repetitive sequence
                if count > 5:
                    print(f"Removing repetitive pattern: '{word}' repeated {count} times")
                    # Remove the repetitive sequence
                    start_idx = i
                    end_idx = j
                    words[start_idx:end_idx] = []
                    segment_modified = True
                    repetitive_removed += 1
                    # Don't increment i since we removed words
                else:
                    i += 1

        # Update segment text if modified
        if segment_modified:
            cleaned_text = " ".join(words)
            # Clean up any double spaces
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Only keep segment if it has meaningful content
        if cleaned_text.strip():
            segment_copy = segment.copy()
            segment_copy["text"] = cleaned_text
            cleaned_segments.append(segment_copy)
        else:
            # Segment became empty after cleaning, don't add it
            repetitive_removed += 1

    return cleaned_segments, repetitive_removed


def _remove_repeated_char_patterns(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove segments with repeated character-level patterns (2-15 chars repeated 5+ times)."""
    cleaned_segments = []
    removed_count = 0

    for segment in segments:
        text = segment.get("text", "")

        # Check for repetitive patterns: 2-15 char patterns repeated 5+ times
        if re.search(r'(.{2,15}?)(?:\1){4,}', text):
            # Remove the repetitive pattern
            cleaned_text = re.sub(r'(.{2,15}?)(?:\1){4,}', '', text)
            # Clean up any leftover whitespace/punctuation
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

            # If block is empty after removal, skip this segment
            if not cleaned_text:
                removed_count += 1
                print(f"Removed empty segment after pattern removal")
                continue

            # Otherwise, update the segment with cleaned text
            segment_copy = segment.copy()
            segment_copy["text"] = cleaned_text
            cleaned_segments.append(segment_copy)
            removed_count += 1
        else:
            cleaned_segments.append(segment)

    return cleaned_segments, removed_count


def _remove_punctuation_only_segments(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove segments that contain only punctuation/signs."""
    cleaned_segments = []
    removed_count = 0

    for segment in segments:
        text = segment.get("text", "").strip()
        # Check if text contains only punctuation/special characters
        if re.match(r'^[^\w\s]+$', text):
            removed_count += 1
            continue
        cleaned_segments.append(segment)

    return cleaned_segments, removed_count


def _remove_leading_spaces(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove leading spaces from segment text."""
    cleaned_segments = []
    removed_count = 0

    for segment in segments:
        original_text = segment.get("text", "")
        if original_text.startswith(" "):
            segment_copy = segment.copy()
            segment_copy["text"] = original_text.lstrip()
            cleaned_segments.append(segment_copy)
            removed_count += 1
        else:
            cleaned_segments.append(segment)

    return cleaned_segments, removed_count


def _remove_duplicate_segments(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove segments that occur 4+ times with identical text."""
    if not segments:
        return segments, 0

    # Count occurrences of each text
    text_counts = {}
    for segment in segments:
        text = segment.get("text", "").strip().lower()
        text_counts[text] = text_counts.get(text, 0) + 1

    # Remove segments that occur 4+ times
    cleaned_segments = []
    removed_count = 0

    for segment in segments:
        text = segment.get("text", "").strip().lower()
        if text_counts[text] >= 4:
            removed_count += 1
            continue
        cleaned_segments.append(segment)

    return cleaned_segments, removed_count


def _normalize_repeated_punctuation(text: str) -> str:
    """Collapse excessive repeated punctuation to a single occurrence."""
    # Convert repeated ellipsis runs like "... ... ..." or "........." to one ellipsis.
    text = re.sub(r'(?:\.\.\.)(?:\s*(?:\.\.\.))+', '...', text)
    text = re.sub(r'\.{4,}', '...', text)
    # Collapse repeated multi-character punctuation tokens separated by spaces.
    text = re.sub(r'([^\w\s]+)(?:\s+\1){2,}', r'\1', text)
    # Collapse long runs of the same punctuation character.
    text = re.sub(r'([^\w\s.])\1{2,}', r'\1', text)
    # Clean up extra spaces left around collapsed punctuation.
    return re.sub(r'\s{2,}', ' ', text).strip()


def _remove_repeated_words(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove words that repeat 4+ times in the same segment, keep only first occurrence."""
    cleaned_segments = []
    total_removed = 0

    for segment in segments:
        original_text = segment.get("text", "")
        words = original_text.split()

        if not words:
            cleaned_segments.append(segment)
            continue

        # Track word positions to remove
        words_to_remove = set()
        word_positions = {}

        for i, word in enumerate(words):
            # Clean word for comparison (remove punctuation)
            clean_word = word.strip(".,!?;:").lower()

            if clean_word not in word_positions:
                word_positions[clean_word] = []
            word_positions[clean_word].append(i)

        # Mark words for removal if they appear 4+ times
        for clean_word, positions in word_positions.items():
            if len(positions) >= 4:
                # Keep first occurrence, remove the rest
                for pos in positions[1:]:
                    words_to_remove.add(pos)
                total_removed += len(positions) - 1

        # Build cleaned text
        if words_to_remove:
            cleaned_words = [word for i, word in enumerate(words) if i not in words_to_remove]
            segment_copy = segment.copy()
            segment_copy["text"] = " ".join(cleaned_words)
            cleaned_segments.append(segment_copy)
        else:
            cleaned_segments.append(segment)

    return cleaned_segments, total_removed


def _save_merge_report(merge_details: List[Dict[str, Any]], output_dir: str) -> None:
    """Save a detailed report of all segment merges to a text file."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / "01_merge_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SEGMENT MERGE REPORT\n")
            f.write("=" * 80 + "\n\n")

            if not merge_details:
                f.write("No segments were merged during refinement.\n")
                return

            f.write(f"Total merges performed: {len(merge_details)}\n")
            f.write(f"Total segments merged: {sum(detail['original_count'] for detail in merge_details)}\n")
            f.write(f"Net segment reduction: {sum(detail['original_count'] - 1 for detail in merge_details)}\n\n")

            for merge in merge_details:
                f.write(f"MERGE #{merge['merge_id']}\n")
                f.write("-" * 40 + "\n")

                f.write(f"Segments merged: {merge['original_count']}\n")
                f.write(f"Time range: {merge['merged_start']:.2f}s - {merge['merged_end']:.2f}s\n")
                f.write(f"Duration: {merge['merged_end'] - merge['merged_start']:.2f}s\n\n")

                f.write("BEFORE MERGE:\n")
                for i, (segment, text) in enumerate(zip(merge['before_segments'], merge['original_texts'])):
                    segment_id = segment.get('id', i+1)
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    f.write(f"  Segment ID {segment_id} [{start_time:.2f}s-{end_time:.2f}s]: {text}\n")

                f.write("\nAFTER MERGE:\n")
                first_segment_id = merge['before_segments'][0].get('id', 'unknown')
                f.write(f"  Merged Segment (ID {first_segment_id}) [{merge['merged_start']:.2f}s-{merge['merged_end']:.2f}s]: {merge['merged_text']}\n")

                f.write("\n" + "=" * 80 + "\n\n")

        print(f"Merge report saved: {report_file}")

    except Exception as e:
        print(f"Warning: Could not save merge report: {e}")


def _save_split_report(split_details: List[Dict[str, Any]], output_dir: str) -> None:
    """Save a detailed report of all segment splits to a text file."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / "02_split_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SEGMENT SPLIT REPORT\n")
            f.write("=" * 80 + "\n\n")

            if not split_details:
                f.write("No segments were split during refinement.\n")
                return

            f.write(f"Total splits performed: {len(split_details)}\n")
            f.write(f"Total segments created: {sum(detail['split_count'] for detail in split_details)}\n")
            f.write(f"Net segment increase: {sum(detail['split_count'] - 1 for detail in split_details)}\n\n")

            for split in split_details:
                f.write(f"SPLIT #{split['split_id']}\n")
                f.write("-" * 40 + "\n")

                original_segment = split['original_segment']
                original_id = original_segment.get('id', 'unknown')
                f.write(f"Original Segment ID: {original_id}\n")
                f.write(f"Original Time Range: {split['original_start']:.2f}s - {split['original_end']:.2f}s\n")
                f.write(f"Original Duration: {split['original_end'] - split['original_start']:.2f}s\n")
                f.write(f"Split Into: {split['split_count']} segments\n\n")

                f.write("BEFORE SPLIT:\n")
                f.write(f"  Segment ID {original_id} [{split['original_start']:.2f}s-{split['original_end']:.2f}s]: {split['original_text']}\n")

                f.write("\nAFTER SPLIT:\n")
                for i, (new_segment, text) in enumerate(zip(split['split_segments'], split['split_texts'])):
                    new_id = new_segment.get('id', i+1)
                    start_time = new_segment.get('start', 0)
                    end_time = new_segment.get('end', 0)
                    f.write(f"  Segment ID {new_id} [{start_time:.2f}s-{end_time:.2f}s]: {text}\n")

                f.write("\n" + "=" * 80 + "\n\n")

        print(f"Split report saved: {report_file}")

    except Exception as e:
        print(f"Warning: Could not save split report: {e}")


def print_statistics(statistics: Dict[str, int]) -> None:
    """Print refinement statistics in a formatted way."""
    print("Segment Refinement Statistics:")
    print(f"- Total input segments: {statistics['total_input_segments']}")
    print(f"- Empty segments removed: {statistics['empty_segments_removed']}")
    print(f"- Punctuation-only segments removed: {statistics['punctuation_only_segments_removed']}")
    print(f"- Leading spaces removed: {statistics['leading_spaces_removed']}")
    print(f"- Duplicate segments removed: {statistics['duplicate_segments_removed']}")
    print(f"- Repeated words removed: {statistics['repeated_words_removed']}")
    print(f"- Punctuation fixes: {statistics['punctuation_fixed']}")
    print(f"- Repetitive patterns removed: {statistics['repetitive_patterns_removed']}")
    print(f"- Segments merged: {statistics['segments_merged']}")
    print(f"- Segments split: {statistics['segments_split']}")
    print(f"- Total output segments: {statistics['total_output_segments']}")
