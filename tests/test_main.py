"""Tests for main orchestration pipeline - end-to-end workflow testing."""

import pytest
import json
import tempfile
import os
import sys
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import main


class TestMainOrchestration:
    """Test suite for the main orchestration pipeline using main.py."""

    @pytest.fixture(autouse=True)
    def setup_output_directory(self):
        """Create output directory for test artifacts."""
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def test_audio_file_complete_pipeline(self):
        """
        Test complete pipeline with audio file: Test1.mp3
        Uses main.py orchestration script to test end-to-end workflow.

        Expected: Final translated JSON and SRT file with proper structure, French text, and Chinese translations
        """
        print("\n=== TESTING AUDIO FILE PIPELINE (via main.py) ===")

        # Step 1: Setup paths
        input_audio_path = os.path.join(os.path.dirname(__file__), 'audio', 'Test1.mp3')
        assert os.path.exists(input_audio_path), f"Test audio file not found: {input_audio_path}"
        print(f"Input audio: {input_audio_path}")

        # Expected output files
        expected_json_path = os.path.join(os.path.dirname(__file__), 'audio', 'Test1_translated.json')
        expected_srt_path = os.path.join(self.output_dir, 'Test1_test.srt')
        processing_output_dir = os.path.join(self.output_dir, 'audio_main_test')

        # Clean up any existing outputs
        for path in [expected_json_path, expected_srt_path]:
            if os.path.exists(path):
                os.remove(path)

        # Step 2: Run main.py orchestration via subprocess
        print("\nStep 1: Running main.py orchestration...")
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py'),
            input_audio_path,
            '--output', processing_output_dir,
            '--srt-output', expected_srt_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Check if command succeeded
            if result.returncode != 0:
                print(f"Main.py failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                pytest.fail(f"Main.py execution failed: {result.stderr}")

            print("✅ Main.py orchestration completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout[-500:]}")  # Show last 500 chars

        except subprocess.TimeoutExpired:
            pytest.fail("Main.py execution timed out after 300 seconds")
        except Exception as e:
            pytest.fail(f"Error running main.py: {str(e)}")

        # Step 3: Validate output files exist
        assert os.path.exists(expected_json_path), f"Expected JSON output not found: {expected_json_path}"
        assert os.path.exists(expected_srt_path), f"Expected SRT output not found: {expected_srt_path}"
        assert os.path.exists(processing_output_dir), f"Processing output directory not found: {processing_output_dir}"

        print(f"✅ Output files created:")
        print(f"  - JSON transcript: {expected_json_path}")
        print(f"  - SRT subtitles: {expected_srt_path}")
        print(f"  - Processing dir: {processing_output_dir}")

        # Step 4: Validate JSON transcript structure
        with open(expected_json_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)

        assert 'segments' in transcript, "Transcript missing 'segments' field"
        assert len(transcript['segments']) > 0, "Transcript has no segments"

        # Check that segments have Chinese translations
        segments_with_zh = [s for s in transcript['segments'] if 'zh' in s and s['zh']]
        assert len(segments_with_zh) > 0, "No segments have Chinese translations"

        print(f"✅ JSON transcript validated:")
        print(f"  - Total segments: {len(transcript['segments'])}")
        print(f"  - Segments with Chinese: {len(segments_with_zh)}")
        print(f"  - Translation coverage: {len(segments_with_zh)/len(transcript['segments'])*100:.1f}%")

        # Step 5: Validate SRT structure and content
        with open(expected_srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        assert len(srt_content.strip()) > 0, "SRT file is empty"
        assert ' --> ' in srt_content, "SRT missing timestamp separators"

        # Parse and validate SRT structure
        srt_lines = srt_content.strip().split('\n')
        segment_numbers = [line for line in srt_lines if line.isdigit()]
        timestamp_lines = [line for line in srt_lines if ' --> ' in line]

        assert len(segment_numbers) > 0, "No segment numbers found in SRT"
        assert len(timestamp_lines) > 0, "No timestamps found in SRT"
        assert len(timestamp_lines) == len(segment_numbers), "Timestamp count doesn't match segment count"

        # Validate timestamp format
        for timestamp in timestamp_lines:
            parts = timestamp.split(' --> ')
            assert len(parts) == 2, f"Invalid timestamp structure: {timestamp}"
            for part in parts:
                assert ',' in part, f"Missing milliseconds in timestamp: {part}"

        # Check for bilingual content
        non_empty_lines = [line.strip() for line in srt_lines if line.strip() and not line.isdigit() and ' --> ' not in line]
        french_text_found = any(line and not any(ord(c) > 127 for c in line) for line in non_empty_lines)
        chinese_text_found = any(line and any(ord(c) > 127 for c in line) for line in non_empty_lines)

        print(f"✅ SRT file validated:")
        print(f"  - Segments: {len(segment_numbers)}")
        print(f"  - Timestamps: {len(timestamp_lines)}")
        print(f"  - French text found: {french_text_found}")
        print(f"  - Chinese text found: {chinese_text_found}")

        # Final assertions
        assert french_text_found, "No French text found in SRT"
        assert len(segments_with_zh) > 0, "No Chinese translations found in transcript"

        print("✅ Audio file pipeline test PASSED")

    def test_video_file_complete_pipeline(self):
        """
        Test complete pipeline with video file: Test_video_1.mp4
        Uses main.py orchestration script to test video processing workflow.

        Expected: Final translated JSON file with proper structure and processing artifacts
        """
        print("\n=== TESTING VIDEO FILE PIPELINE (via main.py) ===")

        # Step 1: Setup paths
        input_video_path = os.path.join(os.path.dirname(__file__), 'video', 'Test_video_1.mp4')
        assert os.path.exists(input_video_path), f"Test video file not found: {input_video_path}"
        print(f"Input video: {input_video_path}")

        # Expected output files
        expected_json_path = os.path.join(os.path.dirname(__file__), 'video', 'Test_video_1_translated.json')
        processing_output_dir = os.path.join(self.output_dir, 'video_main_test')

        # Clean up any existing outputs
        if os.path.exists(expected_json_path):
            os.remove(expected_json_path)

        # Step 2: Run main.py orchestration (no SRT generation to save time)
        print("\nStep 1: Running main.py orchestration for video...")
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py'),
            input_video_path,
            '--output', processing_output_dir,
            '--no-srt'  # Skip SRT generation for faster test
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Check if command succeeded
            if result.returncode != 0:
                print(f"Main.py failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                pytest.fail(f"Main.py execution failed for video: {result.stderr}")

            print("✅ Main.py video orchestration completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout[-500:]}")  # Show last 500 chars

        except subprocess.TimeoutExpired:
            pytest.fail("Main.py video execution timed out after 300 seconds")
        except Exception as e:
            pytest.fail(f"Error running main.py for video: {str(e)}")

        # Step 3: Validate output files exist
        assert os.path.exists(expected_json_path), f"Expected JSON output not found: {expected_json_path}"
        assert os.path.exists(processing_output_dir), f"Processing output directory not found: {processing_output_dir}"

        # Check for extracted audio file
        extracted_audio_path = os.path.join(processing_output_dir, 'extracted_audio.wav')
        assert os.path.exists(extracted_audio_path), "Audio extraction failed - no extracted audio file"

        print(f"✅ Video processing outputs created:")
        print(f"  - JSON transcript: {expected_json_path}")
        print(f"  - Processing dir: {processing_output_dir}")
        print(f"  - Extracted audio: {extracted_audio_path}")

        # Step 4: Validate JSON transcript structure
        with open(expected_json_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)

        assert 'segments' in transcript, "Video transcript missing 'segments' field"
        assert len(transcript['segments']) > 0, "Video transcript has no segments"

        # Check that segments have Chinese translations
        segments_with_zh = [s for s in transcript['segments'] if 'zh' in s and s['zh']]
        assert len(segments_with_zh) > 0, "No video segments have Chinese translations"

        print(f"✅ Video JSON transcript validated:")
        print(f"  - Total segments: {len(transcript['segments'])}")
        print(f"  - Segments with Chinese: {len(segments_with_zh)}")
        print(f"  - Translation coverage: {len(segments_with_zh)/len(transcript['segments'])*100:.1f}%")

        print("✅ Video file pipeline test PASSED")

    def test_main_orchestration_consistency(self):
        """
        Test that both audio and video pipelines produce consistent output structure
        when using main.py orchestration script.
        """
        print("\n=== TESTING MAIN ORCHESTRATION CONSISTENCY ===")

        # Expected output files from previous tests
        audio_json_path = os.path.join(os.path.dirname(__file__), 'audio', 'Test1_translated.json')
        video_json_path = os.path.join(os.path.dirname(__file__), 'video', 'Test_video_1_translated.json')

        # Skip test if previous tests haven't run
        if not os.path.exists(audio_json_path):
            pytest.skip("Audio pipeline output not found - run test_audio_file_complete_pipeline first")
        if not os.path.exists(video_json_path):
            pytest.skip("Video pipeline output not found - run test_video_file_complete_pipeline first")

        # Load both transcripts
        with open(audio_json_path, 'r', encoding='utf-8') as f:
            audio_transcript = json.load(f)
        with open(video_json_path, 'r', encoding='utf-8') as f:
            video_transcript = json.load(f)

        # Validate consistent structure
        for name, transcript in [("audio", audio_transcript), ("video", video_transcript)]:
            assert 'segments' in transcript, f"{name} transcript missing 'segments' field"
            assert 'text' in transcript, f"{name} transcript missing 'text' field"
            assert len(transcript['segments']) > 0, f"{name} transcript has no segments"

            # Check that segments have consistent structure
            for segment in transcript['segments'][:5]:  # Check first 5 segments
                assert 'id' in segment, f"{name} segment missing 'id' field"
                assert 'start' in segment, f"{name} segment missing 'start' field"
                assert 'end' in segment, f"{name} segment missing 'end' field"
                assert 'text' in segment, f"{name} segment missing 'text' field"

        print("✅ Structure consistency validated for both audio and video outputs")

        # Check translation coverage
        audio_zh_count = len([s for s in audio_transcript['segments'] if 'zh' in s and s['zh']])
        video_zh_count = len([s for s in video_transcript['segments'] if 'zh' in s and s['zh']])

        assert audio_zh_count > 0, "Audio transcript has no Chinese translations"
        assert video_zh_count > 0, "Video transcript has no Chinese translations"

        audio_coverage = audio_zh_count / len(audio_transcript['segments']) * 100
        video_coverage = video_zh_count / len(video_transcript['segments']) * 100

        print(f"✅ Translation coverage validated:")
        print(f"  - Audio: {audio_zh_count}/{len(audio_transcript['segments'])} ({audio_coverage:.1f}%)")
        print(f"  - Video: {video_zh_count}/{len(video_transcript['segments'])} ({video_coverage:.1f}%)")

        print("✅ Main orchestration consistency test PASSED")