

import pytest
from typing import List, Dict, Any
from src.transcription.segment_refiner import refine_segments
from src.translation.translate import regenerate_sequential_ids

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

    def test_collapse_repeated_spaced_ellipses(self):
        """Collapse repeated ellipsis tokens like '... ... ...' to one '...'."""
        whisper_output = {
            "text": "... ... ... ... ... ...",
            "segments": [
                {"id": 3, "text": "... ... ... ... ... ...", "start": 0.0, "end": 2.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "..."
        assert result["statistics"]["punctuation_fixed"] == 1

    def test_collapse_repeated_exclamation_and_question_marks(self):
        """Collapse excessive repeated punctuation to one mark."""
        whisper_output = {
            "text": "Wait!!!!!! Why???",
            "segments": [
                {"id": 4, "text": "Wait!!!!!! Why???", "start": 0.0, "end": 2.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Wait! Why?"
        assert result["statistics"]["punctuation_fixed"] == 1


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
        # New rule removes leading spaces from text
        assert len(result["segments"]) == 2
        # Find the segment with "Collection compétence"
        competence_segment = None
        for segment in result["segments"]:
            if "Collection compétence" in segment["text"]:
                competence_segment = segment
                break
        assert competence_segment is not None
        # Leading spaces should now be removed
        assert not competence_segment["text"].startswith(" ")
        assert competence_segment["text"] == "Collection compétence."
        assert result["statistics"]["leading_spaces_removed"] == 2

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
        """Segments with only punctuation should be handled (removed by new rule)"""
        whisper_output = {
            "text": "! Normal text",
            "segments": [
                {"id": 0, "text": "!", "start": 0.0, "end": 0.5},
                {"id": 1, "text": "Normal text", "start": 0.5, "end": 2.0}
            ]
        }
        result = refine_segments(whisper_output)
        # New rule removes punctuation-only segments
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Normal text"
        assert result["statistics"]["punctuation_only_segments_removed"] == 1

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


class TestRemoveLeadingSpaces:
    """Test removal of leading spaces from segment text."""

    def test_remove_leading_space_from_single_segment(self):
        """Remove leading space: ' Thank you.' → 'Thank you.'"""
        whisper_output = {
            "text": "Thank you.",
            "segments": [
                {"id": 0, "text": " Thank you.", "start": 0.0, "end": 29.98}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Thank you."
        assert result["statistics"]["leading_spaces_removed"] == 1

    def test_remove_leading_spaces_from_multiple_segments(self):
        """Remove leading spaces from multiple segments"""
        whisper_output = {
            "text": "Hello world. How are you?",
            "segments": [
                {"id": 0, "text": " Hello world.", "start": 0.0, "end": 2.0},
                {"id": 1, "text": " How are you?", "start": 2.0, "end": 4.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Hello world."
        assert result["segments"][1]["text"] == "How are you?"
        assert result["statistics"]["leading_spaces_removed"] == 2

    def test_preserve_segments_without_leading_spaces(self):
        """Segments without leading spaces should remain unchanged"""
        whisper_output = {
            "text": "Normal text.",
            "segments": [
                {"id": 0, "text": "Normal text.", "start": 0.0, "end": 2.0}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Normal text."
        assert result["statistics"]["leading_spaces_removed"] == 0


class TestRemoveDuplicateSegments:
    """Test removal of duplicate segments occurring 4+ times."""

    def test_remove_segments_occurring_4_times(self):
        """Remove segments that occur exactly 4 times with identical text"""
        whisper_output = {
            "text": "Thank you.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 29.98, "text": "Thank you."},
                {"id": 1, "start": 30.0, "end": 59.98, "text": "Thank you."},
                {"id": 2, "start": 60.0, "end": 89.98, "text": "Thank you."},
                {"id": 3, "start": 90.0, "end": 119.98, "text": "Thank you."}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 0
        assert result["statistics"]["duplicate_segments_removed"] == 4

    def test_remove_segments_occurring_more_than_4_times(self):
        """Remove segments that occur more than 4 times"""
        whisper_output = {
            "text": "Thank you.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 20.0, "text": "Thank you."},
                {"id": 1, "start": 20.0, "end": 40.0, "text": "Thank you."},
                {"id": 2, "start": 40.0, "end": 60.0, "text": "Thank you."},
                {"id": 3, "start": 60.0, "end": 80.0, "text": "Thank you."},
                {"id": 4, "start": 80.0, "end": 100.0, "text": "Thank you."}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 0
        assert result["statistics"]["duplicate_segments_removed"] == 5

    def test_preserve_segments_occurring_less_than_4_times(self):
        """Keep segments that occur less than 4 times"""
        whisper_output = {
            "text": "Hello. World.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello."},
                {"id": 1, "start": 2.0, "end": 4.0, "text": "Hello."},
                {"id": 2, "start": 4.0, "end": 6.0, "text": "World."}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 3  # All segments preserved
        assert result["statistics"]["duplicate_segments_removed"] == 0

    def test_case_insensitive_duplicate_detection(self):
        """Detect duplicates regardless of case"""
        whisper_output = {
            "text": "Thank you.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 20.0, "text": "Thank you."},
                {"id": 1, "start": 20.0, "end": 40.0, "text": "thank you."},
                {"id": 2, "start": 40.0, "end": 60.0, "text": "THANK YOU."},
                {"id": 3, "start": 60.0, "end": 80.0, "text": "Thank you."}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 0
        assert result["statistics"]["duplicate_segments_removed"] == 4


class TestRemoveRepeatedWords:
    """Test removal of words that repeat 4+ times in the same segment."""

    def test_remove_word_repeated_4_times(self):
        """Remove word repeated exactly 4 times, keep first occurrence"""
        whisper_output = {
            "text": "Trump Trump Trump Trump test.",
            "segments": [
                {"id": 870, "start": 1407.14, "end": 1440.4,
                 "text": "Trump Trump Trump Trump test."}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Trump test."
        assert result["statistics"]["repeated_words_removed"] == 3

    def test_remove_word_repeated_more_than_4_times(self):
        """Remove word repeated more than 4 times, keep first occurrence"""
        whisper_output = {
            "text": "It's great Trump Trump Trump Trump Trump Trump problem.",
            "segments": [
                {"id": 870, "start": 1407.14, "end": 1440.4,
                 "text": "It's great Trump Trump Trump Trump Trump Trump problem."}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "It's great Trump problem."
        assert result["statistics"]["repeated_words_removed"] == 5

    def test_preserve_words_repeated_less_than_4_times(self):
        """Keep words that repeat less than 4 times"""
        whisper_output = {
            "text": "Hello hello world.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello hello world."}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Hello hello world."
        assert result["statistics"]["repeated_words_removed"] == 0

    def test_handle_punctuation_around_repeated_words(self):
        """Handle punctuation around repeated words correctly - strips punctuation for comparison"""
        whisper_output = {
            "text": "Word, word, word, word, word.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Word, word, word, word, word."}
            ]
        }
        result = refine_segments(whisper_output)
        # Current implementation strips punctuation and treats "Word," and "word" as the same word
        # So it removes duplicates and keeps only the first occurrence
        assert result["segments"][0]["text"] == "Word,"  # Only first occurrence kept
        assert result["statistics"]["repeated_words_removed"] == 4

    def test_multiple_different_repeated_words(self):
        """Handle multiple different words repeated 4+ times"""
        whisper_output = {
            "text": "Trump Trump Trump Trump word word word word test.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 5.0,
                 "text": "Trump Trump Trump Trump word word word word test."}
            ]
        }
        result = refine_segments(whisper_output)
        assert result["segments"][0]["text"] == "Trump word test."
        assert result["statistics"]["repeated_words_removed"] == 6  # 3 Trump + 3 word


class TestRemovePunctuationOnlySegments:
    """Test removal of segments containing only punctuation/signs."""

    def test_remove_exclamation_mark_only_segment(self):
        """Remove segment containing only '!'"""
        whisper_output = {
            "text": "!",
            "segments": [
                {"index": 1489, "fr": "!"}  # Using the exact structure from the example
            ]
        }
        # Convert to standard format for processing
        standard_format = {
            "text": "!",
            "segments": [
                {"id": 1489, "start": 0.0, "end": 1.0, "text": "!"}
            ]
        }
        result = refine_segments(standard_format)
        assert len(result["segments"]) == 0
        assert result["statistics"]["punctuation_only_segments_removed"] == 1

    def test_remove_multiple_punctuation_only_segments(self):
        """Remove segments with various punctuation-only content"""
        whisper_output = {
            "text": "! ? . , ; :",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "!"},
                {"id": 1, "start": 1.0, "end": 2.0, "text": "?"},
                {"id": 2, "start": 2.0, "end": 3.0, "text": "..."},
                {"id": 3, "start": 3.0, "end": 4.0, "text": "@#$"}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 0
        assert result["statistics"]["punctuation_only_segments_removed"] == 4

    def test_preserve_segments_with_alphanumeric_content(self):
        """Keep segments that contain letters or numbers"""
        whisper_output = {
            "text": "Hello! 123.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello!"},
                {"id": 1, "start": 2.0, "end": 4.0, "text": "123."},
                {"id": 2, "start": 4.0, "end": 6.0, "text": "A!"}
            ]
        }
        result = refine_segments(whisper_output)
        assert len(result["segments"]) == 3
        assert result["statistics"]["punctuation_only_segments_removed"] == 0

    def test_handle_mixed_content_segments(self):
        """Keep segments with mixed punctuation and alphanumeric content"""
        whisper_output = {
            "text": "!Hello? World.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "!Hello?"},
                {"id": 1, "start": 2.0, "end": 4.0, "text": " World."}
            ]
        }
        result = refine_segments(whisper_output)
        # Both segments should be preserved since they contain alphanumeric content
        # and the first ends with sentence-ending punctuation
        assert len(result["segments"]) == 2
        assert result["statistics"]["punctuation_only_segments_removed"] == 0


class TestSequentialIDRegeneration:
    """Test ID regeneration after segment refinement to prevent index mismatch in translation."""

    def test_regenerate_ids_after_merge_creates_continuous_sequence(self):
        """When segments 68 and 69 are merged, regenerate IDs to create continuous sequence."""
        # Simulate refined segments where 68 and 69 were merged into one segment
        refined_segments = [
            {"id": 67, "text": "Sentence 67.", "start": 67.0, "end": 68.0},
            {"id": 68, "text": "Merged sentence from original 68 and 69.", "start": 68.0, "end": 70.0},  # This was 68+69 merged
            {"id": 70, "text": "Sentence 70.", "start": 70.0, "end": 71.0},  # Note the gap: no id 69
            {"id": 71, "text": "Sentence 71.", "start": 71.0, "end": 72.0}
        ]

        # Apply ID regeneration
        regenerated = regenerate_sequential_ids(refined_segments)

        # Verify continuous sequential IDs starting from 1
        assert len(regenerated) == 4
        assert regenerated[0]["id"] == 1
        assert regenerated[1]["id"] == 2
        assert regenerated[2]["id"] == 3
        assert regenerated[3]["id"] == 4

        # Verify text content is preserved
        assert regenerated[0]["text"] == "Sentence 67."
        assert regenerated[1]["text"] == "Merged sentence from original 68 and 69."
        assert regenerated[2]["text"] == "Sentence 70."
        assert regenerated[3]["text"] == "Sentence 71."

        # Verify timing is preserved
        assert regenerated[0]["start"] == 67.0
        assert regenerated[0]["end"] == 68.0
        assert regenerated[1]["start"] == 68.0
        assert regenerated[1]["end"] == 70.0

    def test_regenerate_ids_with_complex_merge_scenario(self):
        """Test ID regeneration with multiple merge operations creating gaps."""
        # Simulate a complex refinement scenario with multiple gaps
        refined_segments = [
            {"id": 1, "text": "First sentence.", "start": 0.0, "end": 2.0},
            {"id": 3, "text": "Third sentence (2 was merged with 1).", "start": 2.0, "end": 4.0},
            {"id": 5, "text": "Fifth sentence (4 was empty and removed).", "start": 4.0, "end": 6.0},
            {"id": 8, "text": "Eighth sentence (6-7 were merged into this).", "start": 6.0, "end": 9.0},
            {"id": 9, "text": "Ninth sentence.", "start": 9.0, "end": 10.0}
        ]

        regenerated = regenerate_sequential_ids(refined_segments)

        # Verify continuous sequential IDs
        assert len(regenerated) == 5
        for i, segment in enumerate(regenerated, start=1):
            assert segment["id"] == i

        # Verify order and content are preserved
        assert regenerated[1]["text"] == "Third sentence (2 was merged with 1)."
        assert regenerated[2]["text"] == "Fifth sentence (4 was empty and removed)."
        assert regenerated[3]["text"] == "Eighth sentence (6-7 were merged into this)."

    def test_no_gaps_preserves_relative_ordering(self):
        """When no gaps exist, IDs should still be regenerated starting from 1."""
        segments_without_gaps = [
            {"id": 10, "text": "First", "start": 0.0, "end": 1.0},
            {"id": 11, "text": "Second", "start": 1.0, "end": 2.0},
            {"id": 12, "text": "Third", "start": 2.0, "end": 3.0}
        ]

        regenerated = regenerate_sequential_ids(segments_without_gaps)

        # Should still regenerate to start from 1
        assert len(regenerated) == 3
        assert regenerated[0]["id"] == 1
        assert regenerated[1]["id"] == 2
        assert regenerated[2]["id"] == 3

        # Content and order preserved
        assert regenerated[0]["text"] == "First"
        assert regenerated[1]["text"] == "Second"
        assert regenerated[2]["text"] == "Third"

    def test_empty_list_handling(self):
        """Empty segment list should return empty list."""
        empty_segments = []
        regenerated = regenerate_sequential_ids(empty_segments)

        assert regenerated == []

    def test_single_segment(self):
        """Single segment should get ID 1."""
        single_segment = [
            {"id": 42, "text": "Only segment", "start": 0.0, "end": 5.0}
        ]

        regenerated = regenerate_sequential_ids(single_segment)

        assert len(regenerated) == 1
        assert regenerated[0]["id"] == 1
        assert regenerated[0]["text"] == "Only segment"
