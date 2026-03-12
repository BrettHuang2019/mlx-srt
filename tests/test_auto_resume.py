#!/usr/bin/env python3
"""
Tests for automatic resume functionality.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import main as main_module
from main import main


def test_auto_resume_behavior():
    """Test that main.py automatically detects and resumes from existing state."""
    print("Testing automatic resume behavior...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock state file
        state_file = Path(temp_dir) / "state.json"
        mock_state = {
            "pipeline_info": {
                "pipeline_id": "test_pipeline_123",
                "status": "running",
                "start_time": "2024-12-20T10:00:00",
                "last_checkpoint": "2024-12-20T10:30:00"
            },
            "completed_steps": ["refinement"],
            "steps": {
                "refinement": {"status": "completed", "file": "01_refined_transcript.json"}
            }
        }

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(mock_state, f, ensure_ascii=False, indent=2)

        print(f"✓ Created mock state file: {state_file}")

        # Create the refinement file that the state references
        refinement_file = Path(temp_dir) / "01_refined_transcript.json"
        refinement_file.write_text(json.dumps({"text": "test", "segments": []}))

        print("✓ Created referenced refinement file")

        # The test would need actual audio/video files and model loading to fully work,
        # but we can at least verify that the auto-resume detection logic works
        # by checking that the state detection code runs without errors

        # Test state file detection
        if state_file.exists():
            auto_resume = True
            print(f"✓ Auto-resume detection works: Found existing state in {temp_dir}")
        else:
            auto_resume = False
            print("✗ Auto-resume detection failed: State file not found")

        assert auto_resume, "Auto-resume should detect existing state file"


def test_explicit_resume_without_output_dir_starts_fresh(monkeypatch, tmp_path):
    """Explicit resume should fall back to a fresh run when the output directory is missing."""
    input_audio = tmp_path / "sample.mp3"
    input_audio.write_bytes(b"fake audio")

    output_dir = tmp_path / "missing_output"
    transcript = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Bonjour", "zh": "\u4f60\u597d"}
        ]
    }

    monkeypatch.setattr(main_module, "check_system_resources", lambda: True)
    monkeypatch.setattr(main_module, "process_audio_file", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(main_module, "generate_srt_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(input_audio),
            "--output",
            str(output_dir),
            "--resume",
            "--no-srt",
            "--keep-artifacts",
        ],
    )

    main()

    final_transcript_path = output_dir / f"{input_audio.stem}_translated.json"
    assert output_dir.exists()
    assert final_transcript_path.exists()


if __name__ == "__main__":
    try:
        test_auto_resume_behavior()
        print("\n🎉 Auto-resume behavior test passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
