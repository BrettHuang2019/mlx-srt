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
    get_resume_point, validate_completed_files
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


if __name__ == "__main__":
    try:
        test_state_management()
        test_state_json_format()
        print("\n🎉 All state management tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise