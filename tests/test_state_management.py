#!/usr/bin/env python3
"""
Tests for state management functionality in translation pipeline.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from translation.translate import (
    load_state, save_state, create_initial_state,
    update_step_status, update_batch_status,
    get_resume_point, validate_completed_files,
    prepare_state_for_resume, load_cached_batch_translation,
    process_batch_recursive
)


def test_state_management():
    """Test basic state management functions."""
    print("Testing state management functions...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test creating initial state
        state = create_initial_state("test_transcript.json", temp_dir)

        assert state is not None
        assert state["pipeline_info"]["status"] == "running"
        assert state["pipeline_info"]["pipeline_id"].startswith("pipeline_")
        assert "refinement" in state["steps"]
        assert "summary" in state["steps"]
        assert "translation" in state["steps"]

        print("✓ Initial state creation works")

        # Test saving and loading state
        save_state(state, temp_dir)
        state_file = Path(temp_dir) / "state.json"
        assert state_file.exists()

        loaded_state = load_state(temp_dir)
        assert loaded_state is not None
        assert loaded_state["pipeline_info"]["pipeline_id"] == state["pipeline_info"]["pipeline_id"]

        print("✓ State save/load works")

        # Test updating step status
        update_step_status(loaded_state, "refinement", "completed",
                          total_input_segments=100, total_output_segments=90)

        assert loaded_state["steps"]["refinement"]["status"] == "completed"
        assert loaded_state["steps"]["refinement"]["total_input_segments"] == 100
        assert loaded_state["steps"]["refinement"]["total_output_segments"] == 90
        assert "refinement" in loaded_state["completed_steps"]

        print("✓ Step status updates work")

        # Test batch status updates
        update_batch_status(loaded_state, "batch_01_segments_001_010", "running",
                          batch_num=1, segments_count=10)

        assert loaded_state["steps"]["translation"]["current_batch"] == "batch_01_segments_001_010"
        assert "batch_01_segments_001_010" in loaded_state["steps"]["translation"]["batch_details"]

        update_batch_status(loaded_state, "batch_01_segments_001_010", "completed",
                          generation_time_seconds=15.5)

        assert "batch_01_segments_001_010" in loaded_state["steps"]["translation"]["completed_batches"]
        assert loaded_state["steps"]["translation"]["batch_details"]["batch_01_segments_001_010"]["generation_time_seconds"] == 15.5

        update_batch_status(loaded_state, "batch_01_segments_001_010", "failed",
                          error="cached response invalid")

        assert "batch_01_segments_001_010" not in loaded_state["steps"]["translation"]["completed_batches"]
        assert "batch_01_segments_001_010" in loaded_state["steps"]["translation"]["failed_batches"]

        print("✓ Batch status updates work")

        # Test resume point detection
        resume_point = get_resume_point(loaded_state, temp_dir)
        # Should return "refinement" because refinement file is missing
        assert resume_point == "refinement"

        print("✓ Resume point detection works")

        # Save updated state
        save_state(loaded_state, temp_dir)

        # Test validation with missing files
        is_valid = validate_completed_files(loaded_state, temp_dir)
        # Should be False because refinement file doesn't exist
        assert is_valid == False

        print("✓ File validation works with missing files")

        # Create the missing file
        refinement_file = Path(temp_dir) / "01_refined_transcript.json"
        refinement_file.write_text(json.dumps({"text": "test", "segments": []}))

        # Now resume point should be "summary" since refinement file exists
        resume_point = get_resume_point(loaded_state, temp_dir)
        assert resume_point == "summary"

        # Test validation with existing file for refinement step
        # Since only refinement is completed and its file exists, validation should pass
        is_valid = validate_completed_files(loaded_state, temp_dir)
        assert is_valid == True

        print("✓ File validation works with existing files")


def test_state_json_format():
    """Test that state file saves as valid JSON."""
    print("\nTesting state JSON format...")

    with tempfile.TemporaryDirectory() as temp_dir:
        state = create_initial_state("test.json", temp_dir)
        update_step_status(state, "refinement", "completed")
        update_batch_status(state, "batch_01", "completed", segments_count=5)

        save_state(state, temp_dir)

        # Load and validate JSON structure
        with open(Path(temp_dir) / "state.json", 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)

        assert "pipeline_info" in loaded_json
        assert "input_files" in loaded_json
        assert "steps" in loaded_json
        assert loaded_json["pipeline_info"]["status"] == "running"

        print("✓ State JSON format is valid")


def test_resume_from_failed_state():
    """Test that failed state.json is converted back to running on resume."""
    print("\nTesting resume from failed state...")

    with tempfile.TemporaryDirectory() as temp_dir:
        state = create_initial_state("test.json", temp_dir)
        update_step_status(state, "refinement", "completed")
        update_step_status(state, "summary", "failed", error="summary failed")
        state["pipeline_info"]["status"] = "failed"
        state["pipeline_info"]["error"] = "summary failed"
        state["pipeline_info"]["end_time"] = datetime.now().isoformat()

        refinement_file = Path(temp_dir) / "01_refined_transcript.json"
        refinement_file.write_text(json.dumps({"text": "test", "segments": []}))

        save_state(state, temp_dir)

        loaded_state = load_state(temp_dir)
        assert get_resume_point(loaded_state, temp_dir) == "summary"

        resumed_state = prepare_state_for_resume(loaded_state)
        assert resumed_state["pipeline_info"]["status"] == "running"
        assert "error" not in resumed_state["pipeline_info"]
        assert "end_time" not in resumed_state["pipeline_info"]
        assert resumed_state["steps"]["summary"]["status"] == "failed"

        print("✓ Failed state resumes from the failing step")


def test_load_cached_batch_translation_prefers_latest_valid_retry():
    """Test that resume ignores an older failed response when a later retry succeeded."""
    print("\nTesting cached batch loading prefers latest valid retry...")

    with tempfile.TemporaryDirectory() as temp_dir:
        batch_id = "batch_01_segments_001_002"
        batch_segments = [
            {"index": 1, "fr": "Bonjour"},
            {"index": 2, "fr": "Merci"},
        ]

        failed_response = Path(temp_dir) / f"07_llm_response_{batch_id}.json"
        failed_response.write_text(json.dumps({
            "batch_id": batch_id,
            "raw_response": '[{"index": 1, "zh": "你好"}]'
        }), encoding='utf-8')

        retry_response = Path(temp_dir) / f"07_llm_response_{batch_id}_retry_1.json"
        retry_response.write_text(json.dumps({
            "batch_id": batch_id,
            "raw_response": json.dumps([
                {"index": 1, "zh": "你好"},
                {"index": 2, "zh": "谢谢"},
            ], ensure_ascii=False)
        }), encoding='utf-8')

        cached = load_cached_batch_translation(temp_dir, batch_id, batch_segments)
        assert cached == [
            {"index": 1, "zh": "你好"},
            {"index": 2, "zh": "谢谢"},
        ]

        print("✓ Latest valid retry is used instead of the first failed response")


def test_process_batch_recursive_resumes_from_split_children(monkeypatch):
    """Test that recursive resume reuses successful children and retries only the missing branch."""
    print("\nTesting recursive batch resume from split children...")

    with tempfile.TemporaryDirectory() as temp_dir:
        batch_id = "batch_01_segments_001_004"
        segments = [
            {"index": 1, "fr": "Bonjour"},
            {"index": 2, "fr": "Merci"},
            {"index": 3, "fr": "Au revoir"},
            {"index": 4, "fr": "A bientot"},
        ]

        cached_first_half = Path(temp_dir) / f"07_llm_response_{batch_id}_a.json"
        cached_first_half.write_text(json.dumps({
            "batch_id": f"{batch_id}_a",
            "raw_response": json.dumps([
                {"index": 1, "zh": "你好"},
                {"index": 2, "zh": "谢谢"},
            ], ensure_ascii=False)
        }), encoding='utf-8')

        calls = []

        def fake_process_single_batch_with_retries(batch_segments, summary, context_text,
                                                   model, tokenizer, translation_prompt, max_tokens,
                                                   temperature, verbose, max_retries, retry_delay,
                                                   output_dir, current_batch_id, depth, indent):
            calls.append((current_batch_id, [item["index"] for item in batch_segments]))
            if current_batch_id == batch_id:
                raise RuntimeError("parent batch previously failed")
            if current_batch_id == f"{batch_id}_b":
                return [
                    {"index": 3, "zh": "再见"},
                    {"index": 4, "zh": "回头见"},
                ]
            raise AssertionError(f"Unexpected retry for {current_batch_id}")

        monkeypatch.setattr(
            "translation.translate.process_single_batch_with_retries",
            fake_process_single_batch_with_retries
        )

        result = process_batch_recursive(
            segments, "summary", "", None, None, "prompt",
            0, 0.0, False, 0, 0.0, temp_dir, batch_id
        )

        assert result == [
            {"index": 1, "zh": "你好"},
            {"index": 2, "zh": "谢谢"},
            {"index": 3, "zh": "再见"},
            {"index": 4, "zh": "回头见"},
        ]
        assert calls == [(f"{batch_id}_b", [3, 4])]

        print("✓ Recursive resume reuses completed split children")


if __name__ == "__main__":
    try:
        test_state_management()
        test_state_json_format()
        test_resume_from_failed_state()
        test_load_cached_batch_translation_prefers_latest_valid_retry()
        print("\n🎉 All state management tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
