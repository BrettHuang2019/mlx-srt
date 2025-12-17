

"""Segment refiner for improving Whisper transcription output."""

import re
from typing import Dict, Any, List, Tuple


def refine_segments(whisper_output: Dict[str, Any]) -> Dict[str, Any]:
    """Refine Whisper transcription segments with improved spacing, merging, and splitting.

    Args:
        whisper_output: Full Whisper JSON output with 'text' and 'segments' fields

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

    # Step 2: Fix punctuation spacing
    for segment in filtered_segments:
        original_text = segment.get("text", "")
        # Remove spaces before ! and ?
        fixed_text = re.sub(r'\s+([!?])', r'\1', original_text)
        if fixed_text != original_text:
            segment["text"] = fixed_text
            stats["punctuation_fixed"] += 1

    # Step 3: Remove repetitive word patterns (LLM artifacts)
    cleaned_segments, repetitive_removed = _remove_repetitive_patterns(filtered_segments)
    stats["repetitive_patterns_removed"] = repetitive_removed

    # Step 4: Merge fragmented sentences
    merged_segments, merge_count = _merge_fragments(cleaned_segments)
    stats["segments_merged"] = merge_count

    # Step 5: Split long blocks
    final_segments, split_count = _split_long_blocks(merged_segments)
    stats["segments_split"] = split_count

    stats["total_output_segments"] = len(final_segments)

    # Recombine text from refined segments
    combined_text = " ".join(segment["text"] for segment in final_segments)

    return {
        "text": combined_text,
        "segments": final_segments,
        "statistics": stats
    }


def _merge_fragments(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Merge consecutive segments where the first doesn't end with sentence-ending punctuation."""
    if not segments:
        return [], 0

    merged = []
    merge_count = 0
    i = 0

    while i < len(segments):
        current = segments[i].copy()
        original_text = current.get("text", "")

        # Preserve leading spaces but work with stripped text for logic
        has_leading_space = original_text.startswith(" ")
        text = original_text.strip()

        # Check if this segment ends with sentence-ending punctuation
        if text and text[-1] not in ".!?":
            # This is a fragment, merge with next segments until we find a complete sentence
            j = i + 1
            merged_text = text
            end_time = current["end"]
            has_merged = False

            while j < len(segments):
                next_segment = segments[j]
                next_text = next_segment.get("text", "").strip()
                if not next_text:
                    j += 1
                    continue

                merged_text += " " + next_text
                end_time = next_segment["end"]
                has_merged = True

                # If we found a complete sentence, stop merging
                if next_text[-1] in ".!?":
                    break
                j += 1

            # Only count as a merge if we actually merged with another segment
            if has_merged:
                merge_count += 1
                # Update current segment with merged content
                current["text"] = " " + merged_text if has_leading_space else merged_text
                current["end"] = end_time
                merged.append(current)
                i = j + 1  # Skip the segments we merged
            else:
                # No merge occurred, just add the original
                current["text"] = original_text
                merged.append(current)
                i += 1
        else:
            # This segment is already complete, just add it
            current["text"] = original_text
            merged.append(current)
            i += 1

    return merged, merge_count


def _split_long_blocks(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Split segments containing multiple long sentences (>5 words each)."""
    split_segments = []
    split_count = 0

    for segment in segments:
        text = segment.get("text", "").strip()
        sentences = re.split(r'([.!?])\s*', text)

        # Reconstruct sentences with punctuation
        reconstructed_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                if sentence.strip():
                    reconstructed_sentences.append(sentence.strip())

        # Check if we need to split (multiple long sentences)
        if (len(reconstructed_sentences) > 1 and
            all(len(sentence.split()) > 5 for sentence in reconstructed_sentences)):

            # Split into multiple segments
            total_duration = segment["end"] - segment["start"]
            split_count += 1

            for i, sentence in enumerate(reconstructed_sentences):
                # Calculate proportional timing
                sentence_words = len(sentence.split())
                total_words = sum(len(s.split()) for s in reconstructed_sentences)

                new_start = segment["start"] + (total_duration * i / len(reconstructed_sentences))
                new_end = segment["start"] + (total_duration * (i + 1) / len(reconstructed_sentences))

                split_segments.append({
                    "id": segment["id"] + i * 0.1,  # Slight offset to maintain uniqueness
                    "start": new_start,
                    "end": new_end,
                    "text": " " + sentence if i > 0 else sentence  # Add leading space for continuation
                })
        else:
            split_segments.append(segment)

    return split_segments, split_count


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


def print_statistics(statistics: Dict[str, int]) -> None:
    """Print refinement statistics in a formatted way."""
    print("Segment Refinement Statistics:")
    print(f"- Total input segments: {statistics['total_input_segments']}")
    print(f"- Empty segments removed: {statistics['empty_segments_removed']}")
    print(f"- Punctuation fixes: {statistics['punctuation_fixed']}")
    print(f"- Repetitive patterns removed: {statistics['repetitive_patterns_removed']}")
    print(f"- Segments merged: {statistics['segments_merged']}")
    print(f"- Segments split: {statistics['segments_split']}")
    print(f"- Total output segments: {statistics['total_output_segments']}")