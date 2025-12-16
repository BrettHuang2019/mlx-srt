"""Tests for subtitle generation functionality."""

import json
import tempfile
from pathlib import Path
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from subtitle.generate_srt import generate_srt_from_segments, format_timestamp


def test_format_timestamp():
    """Test timestamp formatting."""
    # Test basic conversion
    assert format_timestamp(5.627) == "00:00:05,627"
    assert format_timestamp(65.123) == "00:01:05,123"

    # Test with hours
    assert format_timestamp(3661.999) == "01:01:01,999"

    # Test edge cases
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(0.001) == "00:00:00,001"


def test_generate_srt_from_segments():
    """Test SRT generation from segments."""
    segments = [
        {
            "id": 1,
            "start": 5.627,
            "end": 8.827,
            "text": "Le pique-nique royal de Fée.",
            "zh": "仙女们的皇家野餐。"
        },
        {
            "id": 2,
            "start": 11.327,
            "end": 14.407,
            "text": "Je crois que c'est le jour idéal pour faire notre pique-nique royal de Fée.",
            "zh": "我想今天是举行我们仙女皇家野餐的理想日子。"
        }
    ]

    expected_srt = """1
00:00:05,627 --> 00:00:08,827
Le pique-nique royal de Fée.
仙女们的皇家野餐。

2
00:00:11,327 --> 00:00:14,407
Je crois que c'est le jour idéal pour faire notre pique-nique royal de Fée.
我想今天是举行我们仙女皇家野餐的理想日子。
"""

    result = generate_srt_from_segments(segments)
    assert result == expected_srt


def test_generate_srt_fallback_to_original():
    """Test SRT generation falls back to original text when translation missing."""
    segments = [
        {
            "id": 1,
            "start": 5.627,
            "end": 8.827,
            "text": "Original text without translation"
        }
    ]

    result = generate_srt_from_segments(segments)
    lines = result.split('\n')
    assert "Original text without translation" in lines[2]
    # Should only have one line of text when no translation available
    assert lines[3] == ""  # Empty line after text


def test_end_to_end_srt_generation():
    """Test complete SRT generation workflow using test input file."""
    # Load test data
    test_input_path = Path(__file__).parent / "input" / "test_merge_translations.json"

    if not test_input_path.exists():
        # Create test data if it doesn't exist
        test_data = {
            "text": "Le pique-nique royal de Fée. Je crois que c'est le jour idéal pour faire notre pique-nique royal de Fée.",
            "segments": [
                {
                    "id": 1,
                    "start": 5.627,
                    "end": 8.827,
                    "text": "Le pique-nique royal de Fée.",
                    "zh": "仙女们的皇家野餐。"
                },
                {
                    "id": 2,
                    "start": 11.327,
                    "end": 14.407,
                    "text": "Je crois que c'est le jour idéal pour faire notre pique-nique royal de Fée.",
                    "zh": "我想今天是举行我们仙女皇家野餐的理想日子。"
                }
            ]
        }

        test_input_path.parent.mkdir(exist_ok=True)
        with open(test_input_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    # Load and test
    with open(test_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    srt_content = generate_srt_from_segments(data["segments"])

    # Verify SRT content structure
    lines = srt_content.split('\n')

    # Should have entry numbers
    assert "1" in lines
    assert "2" in lines

    # Should have timestamps
    assert "00:00:05,627 --> 00:00:08,827" in lines
    assert "00:00:11,327 --> 00:00:14,407" in lines

    # Should have translated text
    assert "仙女们的皇家野餐。" in lines
    assert "我想今天是举行我们仙女皇家野餐的理想日子。" in lines


if __name__ == "__main__":
    print("Running subtitle generation tests...")

    try:
        test_format_timestamp()
        print("✓ test_format_timestamp passed")

        test_generate_srt_from_segments()
        print("✓ test_generate_srt_from_segments passed")

        test_generate_srt_fallback_to_original()
        print("✓ test_generate_srt_fallback_to_original passed")

        test_end_to_end_srt_generation()
        print("✓ test_end_to_end_srt_generation passed")

        print("\nAll tests passed! 🎉")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)