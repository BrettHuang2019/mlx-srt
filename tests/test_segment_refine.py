

import pytest
from typing import List, Dict, Any
from src.transcription.segment_refiner import refine_segments

class TestPunctuationSpacing:
    """Test punctuation spacing correction."""

    def test_remove_space_before_exclamation_mark(self):
        """Remove space before exclamation mark: 'Bravo !' → 'Bravo!'"""
        whisper_output = {
            "text": "Bravo !",
            "segments": [
                {"id": 0, "text": "Bravo !", "start": 0.0, "end": 2.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Bravo!"
        assert result["statistics"]["punctuation_fixed"] == 1

    def test_remove_space_before_question_mark(self):
        """Remove space before question mark: 'Comment ?' → 'Comment?'"""
        whisper_output = {
            "text": "Comment ?",
            "segments": [
                {"id": 1, "text": "Comment ?", "start": 1.0, "end": 3.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Comment?"
        assert result["statistics"]["punctuation_fixed"] == 1

    def test_preserve_other_punctuation_spacing(self):
        """Commas, periods, semicolons should remain unchanged"""
        whisper_output = {
            "text": "Hello , world .",
            "segments": [
                {"id": 2, "text": "Hello , world .", "start": 0.0, "end": 2.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Hello , world ."
        assert result["statistics"]["punctuation_fixed"] == 0


class TestSplitLongBlocks:
    """Test splitting long segments."""

    def test_split_two_long_sentences_in_one_segment(self):
        """Each sentence >5 words, split into 2 segments with proportional timing"""
        whisper_output = {
            "text": "This is a very long sentence with many words. And this is another long sentence that should be split.",
            "segments": [
                {
                    "id": 0,
                    "text": "This is a very long sentence with many words. And this is another long sentence that should be split.",
                    "start": 0.0,
                    "end": 10.0
                }
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 2
        assert "This is a very long sentence with many words." in result["segments"][0]["text"]
        assert "And this is another long sentence that should be split." in result["segments"][1]["text"]
        # Check timing is proportional - segments should not overlap
        assert result["segments"][0]["end"] <= result["segments"][1]["start"]
        assert result["segments"][0]["end"] == pytest.approx(5.0, rel=0.2)  # First part gets half the time
        assert result["statistics"]["segments_split"] == 1

    def test_no_split_when_sentences_are_short(self):
        """If either sentence ≤5 words, keep together"""
        whisper_output = {
            "text": "Short sentence. Another short one.",
            "segments": [
                {
                    "id": 1,
                    "text": "Short sentence. Another short one.",
                    "start": 0.0,
                    "end": 5.0
                }
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Short sentence. Another short one."
        assert result["statistics"]["segments_split"] == 0

    def test_no_split_for_single_sentence(self):
        """One long sentence stays in one segment"""
        whisper_output = {
            "text": "This is a single very long sentence that should remain in one segment.",
            "segments": [
                {
                    "id": 2,
                    "text": "This is a single very long sentence that should remain in one segment.",
                    "start": 0.0,
                    "end": 8.0
                }
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 1
        assert result["statistics"]["segments_split"] == 0


class TestMergeFragmentedSentences:
    """Test merging fragmented sentences."""

    def test_merge_incomplete_sentence_with_next_segment(self):
        """Segment without .!? at end merges with next"""
        whisper_output = {
            "text": "This is an incomplete sentence that continues.",
            "segments": [
                {"id": 0, "text": "This is an incomplete", "start": 0.0, "end": 2.0},
                {"id": 1, "text": "sentence that continues.", "start": 2.0, "end": 4.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "This is an incomplete sentence that continues."
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 4.0
        assert result["statistics"]["segments_merged"] == 1

    def test_merge_multiple_fragments_into_complete_sentence(self):
        """Chain of 3+ fragments merge until finding .!?"""
        whisper_output = {
            "text": "Part one part two part three. New sentence starts here",
            "segments": [
                {"id": 0, "text": "Part one", "start": 0.0, "end": 1.0},
                {"id": 1, "text": "part two", "start": 1.0, "end": 2.0},
                {"id": 2, "text": "part three.", "start": 2.0, "end": 3.0},
                {"id": 3, "text": "New sentence starts here", "start": 3.0, "end": 4.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Part one part two part three."
        assert result["segments"][1]["text"] == "New sentence starts here"
        assert result["statistics"]["segments_merged"] >= 1

    def test_no_merge_when_sentence_is_complete(self):
        """Segment ending with .!? stays separate"""
        whisper_output = {
            "text": "Complete sentence. Another complete.",
            "segments": [
                {"id": 0, "text": "Complete sentence.", "start": 0.0, "end": 2.0},
                {"id": 1, "text": "Another complete.", "start": 2.0, "end": 4.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Complete sentence."
        assert result["segments"][1]["text"] == "Another complete."
        assert result["statistics"]["segments_merged"] == 0

    def test_merged_timing_uses_combined_range(self):
        """Start from first segment, end from last segment"""
        whisper_output = {
            "text": "First part second part third part.",
            "segments": [
                {"id": 0, "text": "First part", "start": 1.0, "end": 2.5},
                {"id": 1, "text": "second part", "start": 2.5, "end": 3.8},
                {"id": 2, "text": "third part.", "start": 3.8, "end": 5.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 1
        assert result["segments"][0]["start"] == 1.0
        assert result["segments"][0]["end"] == 5.0
        assert result["statistics"]["segments_merged"] == 1


class TestRealWorldData:
    """Test with patterns from actual Whisper output."""

    def test_merge_fragments_seen_in_whisper_output(self):
        """Test merging like segments 35-36 in real data: 'quand il va commencer à faire froid' + 'ou quand il va pleuvoir.'"""
        whisper_output = {
            "text": " Mais tu vas voir quand il va commencer à faire froid ou quand il va pleuvoir. Ce ne sera pas une partie de plaisir.",
            "segments": [
                {"id": 35, "start": 92.24, "end": 94.36, "text": " Mais tu vas voir quand il va commencer à faire froid"},
                {"id": 36, "start": 94.36, "end": 95.7, "text": " ou quand il va pleuvoir."},
                {"id": 37, "start": 96.16, "end": 97.24, "text": " Ce ne sera pas une partie de plaisir."}
            ]
        }
        result = refine_segments(whisper_output)
        # Check that merging occurred - should have fewer segments than input
        assert len(result["segments"]) < len(whisper_output["segments"])
        # Find the merged segment
        merged_segment = None
        for segment in result["segments"]:
            if "froid ou quand il va pleuvoir" in segment["text"]:
                merged_segment = segment
                break
        assert merged_segment is not None
        assert merged_segment["start"] == 92.24
        assert merged_segment["end"] == 95.7
        assert result["statistics"]["segments_merged"] >= 1

    def test_handle_segments_with_leading_spaces(self):
        """Test segments that start with spaces like many in real output"""
        whisper_output = {
            "text": " Collection compétence. Compréhension orale niveau 3.",
            "segments": [
                {"id": 1, "start": 0.14, "end": 4.98, "text": " Collection compétence."},
                {"id": 2, "start": 6.2, "end": 8.02, "text": " Compréhension orale niveau 3."}
            ]
        }
        result = refine_segments(whisper_output)
        # Should preserve leading spaces for the first segment and any that had them originally
        assert len(result["segments"]) == 2
        # Find the segment that should start with space (the "Collection compétence" one)
        competence_segment = None
        for segment in result["segments"]:
            if "Collection compétence" in segment["text"]:
                competence_segment = segment
                break
        assert competence_segment is not None
        assert competence_segment["text"].startswith(" ")

    def test_remove_empty_segments(self):
        """Test removal of empty segments like ids 41-47 in real data"""
        whisper_output = {
            "text": " Vous avez disrespect.",
            "segments": [
                {"id": 40, "start": 99.14, "end": 114.78, "text": " Vous avez disrespect."},
                {"id": 41, "start": 114.78, "end": 114.78, "text": ""},
                {"id": 42, "start": 114.78, "end": 114.78, "text": ""},
                {"id": 43, "start": 114.78, "end": 114.78, "text": ""}
            ]
        }
        result = refine_segments(whisper_output)
        # Should filter out empty segments
        assert len(result["segments"]) == 1
        assert result["segments"][0]["id"] == 40
        assert result["statistics"]["empty_segments_removed"] == 3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_segments_are_handled(self):
        """Empty segments should not cause errors"""
        whisper_output = {
            "text": "Normal text.",
            "segments": [
                {"id": 0, "text": "", "start": 0.0, "end": 1.0},
                {"id": 1, "text": "Normal text.", "start": 1.0, "end": 2.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) >= 1  # Should handle gracefully
        assert result["statistics"]["empty_segments_removed"] == 1

    def test_segments_with_only_punctuation(self):
        """Segments with only punctuation should be handled"""
        whisper_output = {
            "text": "! Normal text",
            "segments": [
                {"id": 0, "text": "!", "start": 0.0, "end": 0.5},
                {"id": 1, "text": "Normal text", "start": 0.5, "end": 2.0}
            ]
        }
        result = refine_segments(whisper_output)
        # Should not crash and handle appropriately
        assert len(result["segments"]) == 2

    def test_last_segment_without_period_not_merged(self):
        """Last segment without period should not be merged (no next segment)"""
        whisper_output = {
            "text": "Complete sentence. Incomplete last segment",
            "segments": [
                {"id": 0, "text": "Complete sentence.", "start": 0.0, "end": 2.0},
                {"id": 1, "text": "Incomplete last segment", "start": 2.0, "end": 4.0}
            ]
        }
        result = refine_segments(whisper_output)
        # Last incomplete segment should remain as-is since there's no next segment to merge with
        assert len(result["segments"]) >= 1
        assert result["statistics"]["segments_merged"] == 0