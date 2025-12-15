"""
Segment Refinement Module

This module refines raw transcription segments from Whisper to improve
readability and natural flow. It merges over-segmented fragments,
respects punctuation boundaries, and optimizes segment timing.
"""

from typing import List
import re
import logging

from src.models import Segment
from src.config import get_transcription_settings


class SegmentRefiner:
    """Refines raw transcription segments for better readability"""

    def __init__(self):
        """Initialize segment refiner with settings"""
        self.settings = get_transcription_settings()["segment_refinement"]
        self.logger = logging.getLogger(__name__)
        self.punctuation_pattern = re.compile(
            r'[.!?;:]'
        )
        self.whitespace_pattern = re.compile(r'\s+')

    def refine_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Refine a list of raw transcription segments

        Args:
            segments: List of raw segments from Whisper

        Returns:
            List[Segment]: Refined segments with improved grouping
        """
        if not segments:
            return []

        # Sort segments by start time to ensure proper order
        sorted_segments = sorted(segments, key=lambda s: s.start_time)

        # Apply refinement rules
        refined_segments = self._merge_fragments(sorted_segments)
        refined_segments = self._respect_punctuation_boundaries(refined_segments)
        refined_segments = self._split_long_segments(refined_segments)
        refined_segments = self._clean_text(refined_segments)

        return refined_segments

    def _merge_fragments(self, segments: List[Segment]) -> List[Segment]:
        """
        Merge over-segmented fragments that should be together

        Args:
            segments: List of segments to process

        Returns:
            List[Segment]: Segments with fragments merged
        """
        if not segments:
            return []

        merged = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            # Check if segments should be merged
            if self._should_merge(current_segment, next_segment):
                # Merge segments
                current_segment = self._merge_two_segments(current_segment, next_segment)
            else:
                # Don't merge - add current and start new
                merged.append(current_segment)
                current_segment = next_segment

        # Add the last segment
        merged.append(current_segment)

        return merged

    def _should_merge(self, seg1: Segment, seg2: Segment) -> bool:
        """
        Determine if two segments should be merged

        Args:
            seg1: First segment
            seg2: Second segment

        Returns:
            bool: True if segments should be merged
        """
        # Don't merge if either segment ends with sentence-ending punctuation
        if self.punctuation_pattern.search(seg1.text.rstrip()[-1:]):
            return False

        # Don't merge if gap is too large
        gap = seg2.start_time - seg1.end_time
        if gap > self.settings["merge_gap_s"]:
            return False

        # Don't merge if first segment is already substantial
        if seg1.duration > self.settings["max_segment_length_s"]:
            return False

        # Don't merge if first segment has enough words
        word_count = len(seg1.text.split())
        if word_count >= self.settings["min_words"] * 3:
            return False

        # Merge if gap is small and second fragment is short
        if (gap <= self.settings["merge_gap_s"] and
            len(seg2.text.split()) <= self.settings["min_words"] * 2):
            return True

        return False

    def _merge_two_segments(self, seg1: Segment, seg2: Segment) -> Segment:
        """
        Merge two segments into one

        Args:
            seg1: First segment
            seg2: Second segment

        Returns:
            Segment: Merged segment
        """
        # Combine text with appropriate spacing
        if seg1.text.rstrip().endswith(('-', '–', '—')):
            # Hyphenated word - continue without space
            combined_text = seg1.text.rstrip() + seg2.text.lstrip()
        else:
            # Normal concatenation with space
            combined_text = seg1.text.rstrip() + ' ' + seg2.text.lstrip()

        # Use earliest start and latest end time
        start_time = min(seg1.start_time, seg2.start_time)
        end_time = max(seg1.end_time, seg2.end_time)

        # Average confidence if both have it
        confidence = None
        if seg1.confidence is not None and seg2.confidence is not None:
            # Weight by segment duration
            total_duration = seg1.duration + seg2.duration
            confidence = (seg1.confidence * seg1.duration + seg2.confidence * seg2.duration) / total_duration
        elif seg1.confidence is not None:
            confidence = seg1.confidence
        elif seg2.confidence is not None:
            confidence = seg2.confidence

        # Preserve language if consistent
        language = seg1.language if seg1.language == seg2.language else None

        return Segment(
            text=combined_text,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            language=language
        )

    def _respect_punctuation_boundaries(self, segments: List[Segment]) -> List[Segment]:
        """
        Ensure segments respect punctuation boundaries

        Args:
            segments: List of segments to process

        Returns:
            List[Segment]: Segments respecting punctuation boundaries
        """
        result = []
        for segment in segments:
            # Check if segment contains sentence-ending punctuation
            if self.punctuation_pattern.search(segment.text):
                # Split at punctuation boundaries if segment is too long
                split_segments = self._split_at_punctuation(segment)
                result.extend(split_segments)
            else:
                result.append(segment)

        return result

    def _split_at_punctuation(self, segment: Segment) -> List[Segment]:
        """
        Split a segment at punctuation boundaries

        Args:
            segment: Segment to split

        Returns:
            List[Segment]: Split segments
        """
        # Find sentence-ending punctuation positions
        punctuation_positions = []
        for match in self.punctuation_pattern.finditer(segment.text):
            punctuation_positions.append(match.end())

        if len(punctuation_positions) <= 1 or segment.duration <= self.settings["max_segment_length_s"]:
            # No need to split
            return [segment]

        # Create new segments
        segments = []
        start_pos = 0
        duration_per_char = segment.duration / len(segment.text)

        for i, pos in enumerate(punctuation_positions[:-1]):  # Don't split at last punctuation
            if i == len(punctuation_positions) - 1:
                break  # Keep last part with final punctuation

            text_part = segment.text[start_pos:pos].strip()
            if not text_part:
                continue

            char_duration = duration_per_char * len(text_part)
            start_time = segment.start_time + (start_pos * duration_per_char)
            end_time = start_time + char_duration

            segments.append(Segment(
                text=text_part,
                start_time=start_time,
                end_time=end_time,
                confidence=segment.confidence,
                language=segment.language
            ))

            start_pos = pos

        # Add final part
        final_text = segment.text[start_pos:].strip()
        if final_text:
            start_time = segment.start_time + (start_pos * duration_per_char)
            segments.append(Segment(
                text=final_text,
                start_time=start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                language=segment.language
            ))

        return segments

    def _split_long_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Split segments that are too long

        Args:
            segments: List of segments to process

        Returns:
            List[Segment]: Segments with appropriate length
        """
        result = []
        for segment in segments:
            if segment.duration <= self.settings["max_segment_length_s"]:
                result.append(segment)
            else:
                # Split long segment
                split_segments = self._split_long_segment(segment)
                result.extend(split_segments)

        return result

    def _split_long_segment(self, segment: Segment) -> List[Segment]:
        """
        Split a long segment into smaller parts

        Args:
            segment: Long segment to split

        Returns:
            List[Segment]: Split segments
        """
        # Simple splitting by time for now
        # TODO: Implement more intelligent splitting based on content
        target_duration = self.settings["max_segment_length_s"] * 0.8
        num_parts = max(2, int(segment.duration / target_duration))

        segments = []
        part_duration = segment.duration / num_parts
        words = segment.text.split()
        words_per_part = len(words) // num_parts

        for i in range(num_parts):
            start_idx = i * words_per_part
            end_idx = start_idx + words_per_part if i < num_parts - 1 else len(words)

            part_words = words[start_idx:end_idx]
            part_text = ' '.join(part_words)

            start_time = segment.start_time + (i * part_duration)
            end_time = start_time + part_duration if i < num_parts - 1 else segment.end_time

            segments.append(Segment(
                text=part_text,
                start_time=start_time,
                end_time=end_time,
                confidence=segment.confidence,
                language=segment.language
            ))

        return segments

    def _clean_text(self, segments: List[Segment]) -> List[Segment]:
        """
        Clean up text in segments

        Args:
            segments: List of segments to clean

        Returns:
            List[Segment]: Cleaned segments
        """
        cleaned = []
        for segment in segments:
            # Normalize whitespace
            cleaned_text = self.whitespace_pattern.sub(' ', segment.text.strip())

            # Remove leading/trailing punctuation that doesn't belong
            while cleaned_text and cleaned_text[0] in ',;:)]}':
                cleaned_text = cleaned_text[1:].strip()

            while cleaned_text and cleaned_text[-1] in '([{' and not cleaned_text.endswith('.!?'):
                cleaned_text = cleaned_text[:-1].strip()

            if cleaned_text:
                cleaned.append(Segment(
                    text=cleaned_text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    language=segment.language
                ))

        return cleaned


def refine_segments(segments: List[Segment]) -> List[Segment]:
    """
    Convenience function to refine transcription segments

    Args:
        segments: List of raw segments from Whisper

    Returns:
        List[Segment]: Refined segments
    """
    refiner = SegmentRefiner()
    return refiner.refine_segments(segments)