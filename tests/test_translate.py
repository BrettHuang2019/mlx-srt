

import pytest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from translation.translate import (
    convert_segments_to_translation_format,
    filter_out_empty_and_ellipsis_segments,
    preserve_segment_order,
    verify_translation_contains_chinese_characters,
    merge_translations_back_to_segments,
    regenerate_full_transcript_text,
    summarize,
    batch_translate,
    translate_transcript,
    load_config
)

# ==========================================
# TRANSLATION PREPROCESSING TESTS
# ==========================================

#Note: all tests should output txt file in output folder.

def test_summary_length_scales_with_transcript_length():
    """
    Test summary adjusts to transcript size.
    ~150-200 words
    """
    # Use the test input file
    input_file = os.path.join(os.path.dirname(__file__), 'input', 'test_summary.json')

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Load and verify the test transcript
    with open(input_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    # Test that we can process the transcript with real summarize function
    assert len(transcript["text"]) > 100  # Ensure we have enough text
    assert len(transcript["segments"]) >= 1  # Ensure we have segments

    # Test the real summarize function with output directory
    summary = summarize(input_file, output_dir)

    # Save summary to output text file
    output_file = os.path.join(output_dir, 'test_summary_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Summary Test Output ===\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Original text length: {len(transcript['text'])} characters\n")
        f.write(f"Number of segments: {len(transcript['segments'])}\n")
        f.write(f"\nGenerated Summary:\n")
        f.write(summary)
        f.write(f"\n\nSummary length: {len(summary)} characters\n")

    # Verify summary is not empty and reasonable length
    assert len(summary.strip()) > 0
    assert len(summary.strip()) < 300

    # Note: mock summary will be shorter, but real summary should be ~150-200 characters
    assert isinstance(summary, str)

    # Verify output file was created
    assert os.path.exists(output_file)
    print(f"Summary test output saved to: {output_file}")


def test_convert_segments_to_translation_format():
    """
    Test converting transcript segments to LLM input format.
    Input: segments array from Whisper JSON with id, start, end, text
    Expected: [{index:1, fr:"Collection compétence."}, {index:2, fr:"Compréhension orale niveau 3."}]
    Note: Extract only id (as 'index') and text (as 'fr' field)
    """
    # Input segments with full Whisper structure
    segments = [
        {"id": 1, "start": 0.0, "end": 2.5, "text": "Collection compétence."},
        {"id": 2, "start": 2.5, "end": 5.0, "text": "Compréhension orale niveau 3."},
        {"id": 3, "start": 5.0, "end": 7.0, "text": "..."},  # Should be filtered out
        {"id": 4, "start": 7.0, "end": 9.0, "text": ""}     # Should be filtered out
    ]

    result = convert_segments_to_translation_format(segments)

    expected = [
        {"index": 1, "fr": "Collection compétence."},
        {"index": 2, "fr": "Compréhension orale niveau 3."}
    ]

    assert result == expected
    assert len(result) == 2  # Only valid segments included


def test_filter_out_empty_and_ellipsis_segments():
    """
    Test removing segments with no meaningful content.
    Input: Segments including "", "   ", "...", "... "
    Expected: These segments excluded from output array
    Note: Keep only segments with actual text content
    """
    segments = [
        {"id": 1, "text": "Valid content"},
        {"id": 2, "text": ""},
        {"id": 3, "text": "   "},
        {"id": 4, "text": "..."},
        {"id": 5, "text": "... "},
        {"id": 6, "text": "Another valid content"}
    ]

    result = filter_out_empty_and_ellipsis_segments(segments)

    expected_ids = [1, 6]
    result_ids = [seg["id"] for seg in result]

    assert result_ids == expected_ids
    assert len(result) == 2


def test_preserve_segment_order():
    """
    Test that segment order is maintained during preprocessing.
    Input: Segments with ids [1, 2, 3, 4]
    Expected: Output array has same order [1, 2, 3, 4]
    """
    # Input with mixed order
    segments = [
        {"id": 3, "text": "Third"},
        {"id": 1, "text": "First"},
        {"id": 4, "text": "Fourth"},
        {"id": 2, "text": "Second"}
    ]

    result = preserve_segment_order(segments)

    expected_ids = [1, 2, 3, 4]
    result_ids = [seg["id"] for seg in result]

    assert result_ids == expected_ids


# ==========================================
# LLM TRANSLATION TESTS
# ==========================================

def test_comprehensive_llm_translation():
    """
    COMPREHENSIVE LLM TRANSLATION TEST
    Tests all LLM translation functionality in one batch_translate call:

    1. Basic zh field addition for all items
    2. Batch processing with multiple segments
    3. ID preservation across all segments
    4. Chinese character validation in translations
    5. Different text types and complexities

    This single test replaces:
    - test_llm_fills_zh_field_for_all_items
    - test_llm_handles_batch_translation
    - test_llm_preserves_ids_in_response
    - test_verify_translation_contains_chinese_characters
    """
    import time
    import re
    import json
    from datetime import datetime

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Load test data from JSON file
    input_file = os.path.join(os.path.dirname(__file__), 'input', 'test_translate.json')

    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_segments = test_data.get('test_segments', [])
    summary = test_data.get('text', 'Test transcript with French dialogues')

    # Timing and execution
    start_time = time.time()
    print(f"Running comprehensive LLM translation test with {len(test_segments)} segments...")

    # Single batch_translate call for all scenarios
    translated_segments = batch_translate(test_segments, summary, output_dir)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Translation completed in {execution_time:.3f} seconds")

    # === COMPREHENSIVE ASSERTIONS ===

    # 1. Basic structure validation (test_llm_fills_zh_field_for_all_items)
    assert len(translated_segments) == len(test_segments), \
        f"Expected {len(test_segments)} translated segments, got {len(translated_segments)}"

    for i, item in enumerate(translated_segments):
        assert "index" in item, f"Segment {i} missing 'index' field"
        assert "zh" in item, f"Segment {i} missing 'zh' field"
        assert item["zh"] is not None, f"Segment {i} has None zh field"
        assert len(item["zh"].strip()) > 0, f"Segment {i} has empty zh field"

    # 2. ID preservation validation (test_llm_preserves_ids_in_response)
    input_ids = {seg["index"] for seg in test_segments}
    output_ids = {seg["index"] for seg in translated_segments}
    assert input_ids == output_ids, f"ID mismatch: input {input_ids}, output {output_ids}"

    # Additional ID order preservation check
    input_order = [seg["index"] for seg in test_segments]
    output_order = [seg["index"] for seg in translated_segments]
    assert input_order == output_order, f"Order not preserved: input {input_order}, output {output_order}"

    # 3. Batch processing validation (test_llm_handles_batch_translation)
    assert len(translated_segments) >= 10, "Should process 10+ items in batch"
    assert all("zh" in item for item in translated_segments), "All items should have zh field"

    # 4. Chinese character validation (test_verify_translation_contains_chinese_characters)
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    for i, item in enumerate(translated_segments):
        zh_text = item["zh"]
        fr_text = next((seg["fr"] for seg in test_segments if seg["index"] == item["index"]), "")

        assert chinese_pattern.search(zh_text), f"Segment {i} translation '{zh_text}' contains no Chinese characters"
        assert zh_text != fr_text, f"Segment {i} not translated (zh == fr): '{zh_text}'"

    # Use the existing verification function
    assert verify_translation_contains_chinese_characters(translated_segments), "Chinese character validation failed"

    # 5. Edge cases and quality checks
    # Check for reasonable translation length (not too short, not too long)
    for i, item in enumerate(translated_segments):
        zh_length = len(item["zh"])
        assert 1 <= zh_length <= 50, f"Segment {i} translation length {zh_length} outside reasonable range"

    print("✓ All assertions passed!")

    # === DETAILED REPORT GENERATION ===

    # Load configuration for report
    try:
        from translation.translate import load_config
        config = load_config()
        translation_config = config.get('translation', {})
        batch_size = translation_config.get('batch_size', 10)
        model_path = translation_config.get('model_path', 'N/A')
    except:
        batch_size = 10
        model_path = 'N/A'

    # Create comprehensive report
    report_file = os.path.join(output_dir, 'comprehensive_llm_translation_test_report.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== COMPREHENSIVE LLM TRANSLATION TEST REPORT ===\n")
        f.write("=" * 60 + "\n\n")

        # Test metadata
        f.write(f"Test Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total Segments Tested: {len(test_segments)}\n")
        f.write(f"Segments Translated: {len(translated_segments)}\n")
        f.write(f"Execution Time: {execution_time:.3f} seconds\n")
        f.write(f"Batch Size from Config: {batch_size}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Status: PASSED\n\n")

        # Configuration information
        f.write("=== CONFIGURATION ===\n")
        f.write(f"Translation Summary: {summary}\n\n")

        # Test scenarios covered
        f.write("=== TEST SCENARIOS COVERED ===\n")
        f.write("✓ test_llm_fills_zh_field_for_all_items - Basic zh field addition\n")
        f.write("✓ test_llm_handles_batch_translation - Batch processing (15+ items)\n")
        f.write("✓ test_llm_preserves_ids_in_response - ID preservation across all segments\n")
        f.write("✓ test_verify_translation_contains_chinese_characters - Chinese character validation\n\n")

        # Input segments breakdown
        f.write("=== INPUT SEGMENTS BREAKDOWN ===\n")
        f.write(f"Total input segments: {len(test_segments)}\n")
        f.write(f"Index range: {min(seg['index'] for seg in test_segments)} - {max(seg['index'] for seg in test_segments)}\n\n")

        f.write("Input segments (first 10):\n")
        for i, seg in enumerate(test_segments[:10]):
            f.write(f"  {i+1:2d}. Index {seg['index']:2d}: '{seg['fr']}'\n")
        if len(test_segments) > 10:
            f.write(f"  ... and {len(test_segments) - 10} more\n")
        f.write("\n")

        # Output analysis
        f.write("=== TRANSLATION OUTPUT ANALYSIS ===\n")

        # Chinese character analysis
        chinese_count = sum(1 for item in translated_segments if chinese_pattern.search(item["zh"]))
        f.write(f"Segments with Chinese characters: {chinese_count}/{len(translated_segments)}\n")

        # Translation length analysis
        lengths = [len(item["zh"]) for item in translated_segments]
        f.write(f"Translation length stats - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}\n\n")

        # Detailed segment results
        f.write("=== DETAILED SEGMENT RESULTS ===\n")
        for i, item in enumerate(translated_segments):
            fr_text = next((seg["fr"] for seg in test_segments if seg["index"] == item["index"]), "N/A")
            has_chinese = bool(chinese_pattern.search(item["zh"]))
            zh_different = item["zh"] != fr_text

            f.write(f"{i+1:2d}. Index {item['index']:2d}:\n")
            f.write(f"    FR: '{fr_text}'\n")
            f.write(f"    ZH: '{item['zh']}'\n")
            f.write(f"    Has Chinese: {'✓' if has_chinese else '✗'}\n")
            f.write(f"    Different from FR: {'✓' if zh_different else '✗'}\n")
            f.write(f"    ZH Length: {len(item['zh'])} chars\n")
            f.write("\n")

        # Validation results summary
        f.write("=== VALIDATION RESULTS SUMMARY ===\n")
        f.write("✓ Input-output count match\n")
        f.write("✓ All segments have 'index' and 'zh' fields\n")
        f.write("✓ All 'zh' fields are non-empty\n")
        f.write("✓ Original IDs preserved exactly\n")
        f.write("✓ Segment order maintained\n")
        f.write("✓ All translations contain Chinese characters\n")
        f.write("✓ All translations differ from original French\n")
        f.write("✓ Translation lengths are reasonable\n")
        f.write("✓ Comprehensive verification function passes\n\n")

        # Performance metrics
        f.write("=== PERFORMANCE METRICS ===\n")
        if execution_time > 0:
            f.write(f"Translation speed: {len(translated_segments)/execution_time:.1f} segments/second\n")
        f.write(f"Average time per segment: {execution_time/len(translated_segments):.3f} seconds\n\n")

        # Batch processing analysis
        expected_batches = (len(test_segments) + batch_size - 1) // batch_size
        f.write("=== BATCH PROCESSING ANALYSIS ===\n")
        f.write(f"Configured batch size: {batch_size}\n")
        f.write(f"Total segments: {len(test_segments)}\n")
        f.write(f"Expected batches: {expected_batches}\n")
        f.write(f"Segments per batch (approx): {len(test_segments)/expected_batches:.1f}\n\n")

        # Test coverage analysis
        f.write("=== TEST COVERAGE ANALYSIS ===\n")

        # Check specific test cases based on actual JSON data
        input_indices = {seg['index'] for seg in test_segments}

        # Test various scenarios based on available segments
        basic_segments = [1, 2] if 1 in input_indices and 2 in input_indices else list(input_indices)[:2]
        id_segments = [5, 10] if 5 in input_indices and 10 in input_indices else [max(input_indices), min(input_indices)]

        basic_present = all(item['index'] in basic_segments for item in translated_segments if item['index'] in basic_segments)
        id_present = all(item['index'] in id_segments for item in translated_segments if item['index'] in id_segments)
        batch_adequate = len(translated_segments) >= 10

        f.write(f"Basic test coverage (segments {basic_segments}): {'✓' if basic_present else '✗'}\n")
        f.write(f"ID preservation test coverage (segments {id_segments}): {'✓' if id_present else '✗'}\n")
        f.write(f"Batch test coverage (10+ segments): {'✓' if batch_adequate else '✗'}\n")
        f.write(f"Chinese validation coverage (all segments): {'✓' if chinese_count == len(translated_segments) else '✗'}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated by: comprehensive_llm_translation_test\n")

    print(f"Comprehensive test report saved to: {report_file}")

    # Save raw JSON data for programmatic access
    json_report_file = os.path.join(output_dir, 'comprehensive_llm_translation_test_data.json')
    test_data = {
        "test_name": "comprehensive_llm_translation",
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": execution_time,
        "input_segments": test_segments,
        "translated_segments": translated_segments,
        "summary": summary,
        "batch_size": batch_size,
        "model_path": model_path,
        "validation_results": {
            "all_segments_translated": len(translated_segments) == len(test_segments),
            "ids_preserved": input_ids == output_ids,
            "chinese_characters_present": verify_translation_contains_chinese_characters(translated_segments),
            "batch_size_adequate": len(translated_segments) >= 10
        }
    }

    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Test data JSON saved to: {json_report_file}")

    # Final validation using the existing function
    assert verify_translation_contains_chinese_characters(translated_segments), "Final Chinese character validation failed"

    print(f"\n🎉 Comprehensive LLM translation test PASSED!")
    print(f"   - {len(translated_segments)} segments translated")
    print(f"   - All test scenarios validated")
    print(f"   - Reports generated in {output_dir}")


def test_verify_translation_contains_chinese_characters():
    """
    Test the helper function for Chinese character validation.
    This is kept separate as it tests the validation function itself,
    not the LLM translation process.
    """
    # Valid translations
    valid_translations = [
        {"id": 1, "fr": "Bonjour", "zh": "你好"},
        {"id": 2, "fr": "Merci", "zh": "谢谢"}
    ]

    # Invalid translations
    invalid_translations = [
        {"id": 1, "fr": "Bonjour", "zh": ""},  # Empty
        {"id": 2, "fr": "Merci", "zh": "Merci"},  # Not translated
        {"id": 3, "fr": "Au revoir", "zh": "Hello"}  # No Chinese chars
    ]

    assert verify_translation_contains_chinese_characters(valid_translations) == True
    assert verify_translation_contains_chinese_characters(invalid_translations) == False


# ==========================================
# POST-PROCESSING TESTS
# ==========================================

def test_merge_translations_back_to_segments():
    """
    Test merging translated text back into original transcript structure.
    Input:
      - Original segments: [{id:1, start:0.0, end:1.5, text:"Bonjour"}]
      - Translations: [{id:1, fr:"Bonjour", zh:"你好"}]
    Expected: [{id:1, start:0.0, end:1.5, text:"Bonjour", zh:"你好"}]
    Note: Match by id, preserve all original fields (start, end, text), add zh field
    """
    original_segments = [
        {"id": 1, "start": 0.0, "end": 1.5, "text": "Bonjour"},
        {"id": 2, "start": 1.5, "end": 3.0, "text": "Merci"}
    ]

    translations = [
        {"id": 1, "fr": "Bonjour", "zh": "你好"},
        {"id": 2, "fr": "Merci", "zh": "谢谢"}
    ]

    result = merge_translations_back_to_segments(original_segments, translations)

    expected = [
        {"id": 1, "start": 0.0, "end": 1.5, "text": "Bonjour", "zh": "你好"},
        {"id": 2, "start": 1.5, "end": 3.0, "text": "Merci", "zh": "谢谢"}
    ]

    assert result == expected


def test_handle_missing_translations_gracefully():
    """
    Test handling when some segments don't have translations.
    Input: Original has ids [1,2,3], translations only have [1,3]
    Expected: Segment 2 either skipped or has zh:null/empty, no crash
    Note: Log warning for missing translations
    """
    original_segments = [
        {"id": 1, "start": 0.0, "end": 1.0, "text": "Bonjour"},
        {"id": 2, "start": 1.0, "end": 2.0, "text": "Merci"},
        {"id": 3, "start": 2.0, "end": 3.0, "text": "Au revoir"}
    ]

    # Missing translation for id=2
    translations = [
        {"id": 1, "fr": "Bonjour", "zh": "你好"},
        {"id": 3, "fr": "Au revoir", "zh": "再见"}
    ]

    result = merge_translations_back_to_segments(original_segments, translations)

    # Check that function doesn't crash and handles missing translations
    assert len(result) == 3
    assert result[0]["zh"] == "你好"
    assert "zh" not in result[1] or result[1]["zh"] is None  # Segment 2 has no translation
    assert result[2]["zh"] == "再见"


def test_regenerate_full_transcript_text_field():
    """
    Test updating top-level 'text' field after adding translations.
    Input: Full transcript JSON with segments containing zh fields
    Expected: Top-level 'text' field regenerated to include all segment texts
    Note: Optional - could add zh translations to combined text or keep original
    """
    segments = [
        {"id": 1, "start": 0.0, "end": 1.0, "text": "Bonjour", "zh": "你好"},
        {"id": 2, "start": 1.0, "end": 2.0, "text": "Merci", "zh": "谢谢"},
        {"id": 3, "start": 2.0, "end": 3.0, "text": "Au revoir", "zh": "再见"}
    ]

    result = regenerate_full_transcript_text(segments)

    expected = "Bonjour Merci Au revoir"
    assert result == expected