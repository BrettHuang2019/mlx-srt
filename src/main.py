#!/usr/bin/env python3
"""
MLX-SRT Main Orchestration Script

This script orchestrates the complete pipeline for:
1. Audio/video ingestion (download from URL or extract audio from video if needed)
2. Speech transcription using Whisper
3. Segment refinement and cleanup
4. Translation from French to Chinese using MLX-LM
5. SRT subtitle generation

Features:
- Support for local files and URLs (YouTube, etc.)
- Automatic video download (1080p max, 720p minimum)
- Chrome cookies integration for authentication
- Automatic state management and resumption
- Robust error recovery
- Batch-level progress tracking
- JSON format for LLM responses

Usage:
    python src/main.py input_file.mp3
    python src/main.py input_file.mp4 [--output output_dir]
    python src/main.py "https://www.youtube.com/watch?v=VIDEO_ID" [--output output_dir]
    python src/main.py --help

Auto-Resume:
The pipeline automatically detects and resumes from existing state files.
If you run the same command with the same output directory, it will:
1. Check for existing state.json
2. Validate completed steps and files
3. Resume from the first incomplete step
4. Skip already processed batches

Use --resume flag to explicitly resume when existing state is found.
If the output directory does not exist yet, the pipeline starts a fresh run instead.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ingestion.extract_audio import extract_audio
from ingestion.download_from_url import is_url, download_video_from_url
from transcription.whisper_transcriber import transcribe_audio, TranscriptionPipelineError
from translation.translate import translation_pipeline
from subtitle.generate_srt import generate_srt_from_segments
from task_manager import check_system_resources


def save_transcription_report(
    output_dir: str,
    transcription_metadata: Optional[Dict[str, Any]],
    error_message: Optional[str] = None,
) -> None:
    """Persist transcription details and failures alongside Whisper artifacts."""
    output_path = Path(output_dir)
    metadata_file = output_path / "00_transcription_metadata.json"
    report_file = output_path / "00_transcription_report.txt"

    output_path.mkdir(parents=True, exist_ok=True)
    transcription_metadata = transcription_metadata or {}

    metadata_to_save = dict(transcription_metadata)
    transformed_attempts = []
    for index, attempt in enumerate(metadata_to_save.get("attempts", []), start=1):
        attempt_to_save = dict(attempt)
        if attempt_to_save.get("response") is not None:
            response_file_name = (
                f"00_transcription_attempt_{index:02d}_{attempt_to_save.get('type', 'unknown')}.json"
            )
            response_file = output_path / response_file_name
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(attempt_to_save["response"], f, ensure_ascii=False, indent=2)
            attempt_to_save["response_file"] = response_file_name
            attempt_to_save.pop("response", None)
        transformed_attempts.append(attempt_to_save)
    if transformed_attempts:
        metadata_to_save["attempts"] = transformed_attempts
    if error_message:
        metadata_to_save["status"] = "failed"
        metadata_to_save["error"] = error_message

    if metadata_to_save:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== TRANSCRIPTION REPORT ===\n\n")
        f.write(f"Status: {metadata_to_save.get('status', 'completed')}\n")
        if error_message:
            f.write(f"Error: {error_message}\n")
        f.write(f"Selected strategy: {metadata_to_save.get('selected_strategy', 'unknown')}\n")
        f.write(f"Selected model: {metadata_to_save.get('selected_model_path', 'unknown')}\n")
        f.write(
            f"Final punctuation ratio: {metadata_to_save.get('final_punctuation_ratio', 0.0):.4f}\n"
        )
        f.write(
            f"Minimum punctuation ratio: {metadata_to_save.get('min_punctuation_ratio', 0.0):.4f}\n"
        )
        f.write(
            f"Punctuation pass applied: {metadata_to_save.get('punctuation_pass_applied', False)}\n"
        )
        if metadata_to_save.get("punctuation_model"):
            f.write(f"Punctuation model: {metadata_to_save['punctuation_model']}\n")
        if metadata_to_save.get("punctuation_chunk_count") is not None:
            f.write(f"Punctuation chunks: {metadata_to_save['punctuation_chunk_count']}\n")
        if metadata_to_save.get("punctuation_chunk_words") is not None:
            f.write(f"Punctuation chunk words: {metadata_to_save['punctuation_chunk_words']}\n")
        if metadata_to_save.get("punctuation_mapping_stats"):
            mapping_stats = metadata_to_save["punctuation_mapping_stats"]
            f.write("Punctuation mapping stats:\n")
            for key, value in mapping_stats.items():
                f.write(f"  {key}: {value}\n")
        if metadata_to_save.get("error_details"):
            f.write("\nError details:\n")
            for key, value in metadata_to_save["error_details"].items():
                f.write(f"{key}: {value}\n")

        f.write("\nAttempts:\n")
        attempts = metadata_to_save.get("attempts", [])
        if not attempts:
            f.write("none\n")
        else:
            for index, attempt in enumerate(attempts, start=1):
                f.write(f"{index}. type={attempt.get('type', 'unknown')}\n")
                f.write(f"   model={attempt.get('model_path', 'unknown')}\n")
                f.write(f"   status={attempt.get('status', 'unknown')}\n")
                if "punctuation_ratio" in attempt:
                    f.write(f"   punctuation_ratio={attempt['punctuation_ratio']:.4f}\n")
                if "input_punctuation_ratio" in attempt:
                    f.write(f"   input_punctuation_ratio={attempt['input_punctuation_ratio']:.4f}\n")
                if "output_punctuation_ratio" in attempt:
                    f.write(f"   output_punctuation_ratio={attempt['output_punctuation_ratio']:.4f}\n")
                if "chunk_count" in attempt:
                    f.write(f"   chunk_count={attempt['chunk_count']}\n")
                if "chunk_words" in attempt:
                    f.write(f"   chunk_words={attempt['chunk_words']}\n")
                if attempt.get("mapping_stats"):
                    for detail_key, detail_value in attempt["mapping_stats"].items():
                        f.write(f"   mapping_{detail_key}={detail_value}\n")
                if attempt.get("error"):
                    f.write(f"   error={attempt['error']}\n")
                if attempt.get("error_details"):
                    for detail_key, detail_value in attempt["error_details"].items():
                        f.write(f"   {detail_key}={detail_value}\n")
                if attempt.get("response_file"):
                    f.write(f"   response_file={attempt['response_file']}\n")
                f.write("\n")

    if metadata_to_save:
        print(f"Transcription metadata saved: {metadata_file}")
    print(f"Transcription report saved: {report_file}")


def process_audio_file(audio_path: str, output_dir: str, resume: bool = False, url: str = None, downloaded_file: str = None, video_title: str = None) -> Dict[str, Any]:
    """Process audio file through transcription and translation pipeline.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save intermediate and final files
        resume: Whether to resume from last checkpoint
        url: Original URL (optional, for state tracking)
        downloaded_file: Path to downloaded file (optional, for state tracking)
        video_title: Video title (optional, for state tracking)

    Returns:
        Translated transcript with Chinese translations
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING AUDIO FILE")
    print(f"{'='*50}")
    print(f"Input: {audio_path}")
    print(f"Output: {output_dir}")

    # For resume mode, check if transcription already exists
    transcription_file = Path(output_dir) / "00_whisper_transcription.json"
    transcription_metadata_file = Path(output_dir) / "00_transcription_metadata.json"
    whisper_result = None
    transcription_metadata = None

    if resume and transcription_file.exists():
        print(f"\nStep 1: Loading existing transcription...")
        with open(transcription_file, 'r', encoding='utf-8') as f:
            whisper_result = json.load(f)
        if transcription_metadata_file.exists():
            with open(transcription_metadata_file, 'r', encoding='utf-8') as f:
                transcription_metadata = json.load(f)
        print(f"Transcription loaded: {transcription_file}")
    else:
        print(f"\nStep 1: Transcribing audio...")
        try:
            whisper_result, transcription_metadata = transcribe_audio(audio_path, return_metadata=True)

            # Save transcription
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(whisper_result, f, ensure_ascii=False, indent=2)
            print(f"Transcription saved: {transcription_file}")
            save_transcription_report(output_dir, transcription_metadata)
        except Exception as e:
            failure_metadata = None
            partial_result = None
            if isinstance(e, TranscriptionPipelineError):
                failure_metadata = e.metadata
                partial_result = e.partial_result
            if partial_result:
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    json.dump(partial_result, f, ensure_ascii=False, indent=2)
                print(f"💾 Raw Whisper transcription saved despite failure: {transcription_file}")
            save_transcription_report(output_dir, failure_metadata, error_message=str(e))
            # Mark pipeline as failed
            state_file = Path(output_dir) / "state.json"
            if state_file.exists():
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    state["pipeline_info"]["status"] = "failed"
                    state["pipeline_info"]["end_time"] = datetime.now().isoformat()
                    state["pipeline_info"]["error"] = f"Transcription failed: {e}"
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state, f, ensure_ascii=False, indent=2)
                    print(f"💾 Pipeline status updated to 'failed' in: {state_file}")
                except:
                    pass
            raise RuntimeError(f"Transcription failed: {e}")

    # Step 2: Translation pipeline
    print(f"\nStep 2: Running translation pipeline...")
    try:
        translated_transcript = translation_pipeline(
            whisper_result,
            output_dir,
            resume,
            url,
            downloaded_file,
            video_title,
            transcription_metadata=transcription_metadata,
        )
    except Exception as e:
        # Mark pipeline as failed
        state_file = Path(output_dir) / "state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                state["pipeline_info"]["status"] = "failed"
                state["pipeline_info"]["end_time"] = datetime.now().isoformat()
                state["pipeline_info"]["error"] = str(e)
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                print(f"💾 Pipeline status updated to 'failed' in: {state_file}")
            except:
                pass
        raise RuntimeError(f"Translation pipeline failed: {e}")

    return translated_transcript


def process_url(url: str, output_dir: str, resume: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Process URL by downloading video then processing through pipeline.

    Args:
        url: URL to download video from
        output_dir: Directory to save intermediate and final files
        resume: Whether to resume from last checkpoint

    Returns:
        Tuple of (downloaded_video_path, translated_transcript)
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING URL")
    print(f"{'='*50}")
    print(f"URL: {url}")
    print(f"Output: {output_dir}")

    # Step 1: Check if download was already completed in previous run
    downloaded_video_path = None
    video_title = None

    if resume:
        state_file = Path(output_dir) / "state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # Check if download was completed
                if (state.get("download_info", {}).get("download_completed") and
                    state.get("download_info", {}).get("url") == url):
                    downloaded_file = state.get("download_info", {}).get("downloaded_file")
                    video_title = state.get("download_info", {}).get("video_title")

                    if downloaded_file and Path(downloaded_file).exists():
                        downloaded_video_path = downloaded_file
                        print(f"\nStep 1: Resuming from completed download...")
                        print(f"URL: {url}")
                        print(f"Video file: {downloaded_video_path}")
                        print(f"Video title: {video_title}")
                    else:
                        print(f"\n⚠️  Download state found but file missing: {downloaded_file}")
                        print(f"Will re-download the video...")
            except Exception as e:
                print(f"\n⚠️  Could not load download state: {e}")
                print(f"Will proceed with fresh download...")

    # If no completed download found, download the video
    if not downloaded_video_path:
        print(f"\nStep 1: Downloading video from URL...")
        try:
            # Download to downloads folder and save state to output_dir
            downloaded_video_path, video_title = download_video_from_url(url, output_dir=output_dir)
            print(f"Video downloaded to: {downloaded_video_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download video from URL: {e}")

    # Step 2: Process downloaded video through the same pipeline
    translated_transcript = process_video_file(downloaded_video_path, output_dir, resume, url, downloaded_video_path, video_title)

    return downloaded_video_path, translated_transcript


def process_video_file(video_path: str, output_dir: str, resume: bool = False, url: str = None, downloaded_file: str = None, video_title: str = None) -> Dict[str, Any]:
    """Process video file through audio extraction, transcription, and translation.

    Args:
        video_path: Path to video file
        output_dir: Directory to save intermediate and final files
        resume: Whether to resume from last checkpoint
        url: Original URL (optional, for state tracking)
        downloaded_file: Path to downloaded file (optional, for state tracking)
        video_title: Video title (optional, for state tracking)

    Returns:
        Translated transcript with Chinese translations
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING VIDEO FILE")
    print(f"{'='*50}")
    print(f"Input: {video_path}")
    print(f"Output: {output_dir}")

    # Step 1: Extract audio from video (skip if resuming and audio already exists)
    extracted_audio_path = Path(output_dir) / "extracted_audio.wav"

    if resume and extracted_audio_path.exists():
        print(f"\nStep 1: Using existing extracted audio...")
        print(f"Audio file: {extracted_audio_path}")
    else:
        print(f"\nStep 1: Extracting audio from video...")
        try:
            extract_audio(video_path, str(extracted_audio_path))

            # Verify audio extraction
            if not extracted_audio_path.exists():
                raise RuntimeError(f"Audio extraction failed: {extracted_audio_path}")

            print(f"Audio extracted: {extracted_audio_path}")
        except Exception as e:
            # Mark pipeline as failed
            state_file = Path(output_dir) / "state.json"
            if state_file.exists():
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    state["pipeline_info"]["status"] = "failed"
                    state["pipeline_info"]["end_time"] = datetime.now().isoformat()
                    state["pipeline_info"]["error"] = f"Audio extraction failed: {e}"
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state, f, ensure_ascii=False, indent=2)
                    print(f"💾 Pipeline status updated to 'failed' in: {state_file}")
                except:
                    pass
            raise RuntimeError(f"Audio extraction failed: {e}")

    # Step 2: Process extracted audio through the same pipeline
    return process_audio_file(str(extracted_audio_path), output_dir, resume, url, downloaded_file, video_title)


def generate_srt_file(translated_transcript: Dict[str, Any], output_path: str) -> None:
    """Generate SRT subtitle file from translated transcript.

    Args:
        translated_transcript: Transcript with French and Chinese text
        output_path: Path to save SRT file
    """
    print(f"\nStep 3: Generating SRT subtitles...")

    if 'segments' not in translated_transcript:
        raise ValueError("Translated transcript missing 'segments' field")

    segments = translated_transcript['segments']
    print(f"Generating SRT from {len(segments)} segments...")

    # Generate SRT content
    srt_content = generate_srt_from_segments(segments)

    # Save SRT file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    print(f"SRT file generated: {output_path}")

    # Print summary
    segments_with_zh = len([s for s in segments if 'zh' in s and s['zh']])
    print(f"SRT Summary:")
    print(f"  - Total segments: {len(segments)}")
    print(f"  - Segments with Chinese translation: {segments_with_zh}")
    print(f"  - Translation coverage: {segments_with_zh/len(segments)*100:.1f}%")


def main():
    """Main orchestration function for MLX-SRT pipeline."""
    parser = argparse.ArgumentParser(
        description="MLX-SRT: Complete pipeline for speech transcription, translation, and subtitle generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process audio file (SRT saved alongside input, artifacts cleaned up)
  python src/main.py audio.mp3

  # Process video file with custom output directory
  python src/main.py video.mp4 --output /path/to/output

  # Process video from URL (YouTube, etc.)
  python src/main.py "https://www.youtube.com/watch?v=VIDEO_ID"

  # Process URL with custom output directory
  python src/main.py "https://www.youtube.com/watch?v=VIDEO_ID" --output /path/to/output

  # Process and keep intermediate artifacts for debugging
  python src/main.py video.mp4 --keep-artifacts

  # Process with SRT output in specific location
  python src/main.py audio.mp3 --srt-output /path/to/subtitles.srt

  # Automatically resume from last checkpoint (if state exists)
  python src/main.py video.mp4 --keep-artifacts

  # Explicitly require resuming from existing state
  python src/main.py video.mp4 --resume --keep-artifacts
        """
    )

    parser.add_argument(
        "input_file",
        help="Input audio/video file path or URL (YouTube, etc.)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output directory for intermediate files and results (default: ./<input_filename>_output)",
        default=None
    )

    parser.add_argument(
        "--srt-output", "-s",
        help="SRT output file path (default: same directory as input file)",
        default=None
    )

    parser.add_argument(
        "--no-srt",
        action="store_true",
        help="Skip SRT generation, only process transcription and translation"
    )

    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the output folder with intermediate artifacts (default: clean up on success)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing state when available. If the output directory does not exist yet, starts a fresh run. Without this flag, will auto-resume if state exists."
    )

    args = parser.parse_args()

    # System resource checks (RAM and running tasks)
    if not check_system_resources():
        print("\n❌ Cannot start task due to insufficient resources or conflicting tasks.")
        sys.exit(1)

    # Check if input is a URL
    input_is_url = is_url(args.input_file)

    # Determine output directory and check for existing state
    if args.output:
        output_dir = Path(args.output)
    else:
        if input_is_url:
            # For URLs, create a safe output directory name inside downloads folder
            from ingestion.download_from_url import get_video_info
            try:
                # Get video info to create a meaningful directory name
                video_info = get_video_info(args.input_file)
                safe_title = "".join(c for c in video_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
                # Place output folder inside downloads directory
                project_root = Path(__file__).parent.parent
                output_dir = project_root / "downloads" / f"{safe_title}_output"
            except:
                # Fallback if we can't get video info
                project_root = Path(__file__).parent.parent
                output_dir = project_root / "downloads" / "downloaded_video_output"
        else:
            # Default to input filename + "_output"
            input_path = Path(args.input_file)
            output_dir = input_path.parent / f"{input_path.stem}_output"

    # Check for existing state and auto-resume if available
    auto_resume = False
    if output_dir.exists():
        state_file = output_dir / "state.json"
        if state_file.exists():
            auto_resume = True
            print(f"📋 Found existing state in: {output_dir}")
            print(f"🔄 Will automatically resume from last checkpoint")

    resume_flag = auto_resume

    # Validate explicit resume flag
    if args.resume and not auto_resume:
        if not output_dir.exists():
            print(f"⚠️  Resume requested but output directory does not exist: {output_dir}")
            print(f"Starting a fresh run and creating the output directory.")

        else:
            state_file = output_dir / "state.json"
            if not state_file.exists():
                print(f"Error: Cannot resume - no state file found: {state_file}")
                print(f"Please run without --resume first to create the initial state.")
                sys.exit(1)

            print(f"📋 Explicitly resuming from existing state in: {output_dir}")
            resume_flag = True
    elif args.resume and auto_resume:
        print(f"📋 Resume flag specified - using existing state in: {output_dir}")
        resume_flag = True

    # Validate input
    if not input_is_url:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file does not exist: {input_path}")
            sys.exit(1)
    else:
        print(f"🔗 Processing URL: {args.input_file}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine file type
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

    file_extension = ""
    if not input_is_url:
        file_extension = input_path.suffix.lower()

    try:
        # Process file based on type
        downloaded_video_path = None

        if input_is_url:
            # Process URL
            downloaded_video_path, translated_transcript = process_url(args.input_file, str(output_dir), resume_flag)
            input_path = Path(downloaded_video_path)  # Use downloaded video for SRT naming
        else:
            # Process local file based on extension
            if file_extension in audio_extensions:
                translated_transcript = process_audio_file(str(input_path), str(output_dir), resume_flag)
            elif file_extension in video_extensions:
                translated_transcript = process_video_file(str(input_path), str(output_dir), resume_flag)
            else:
                print(f"Error: Unsupported file format: {file_extension}")
                print(f"Supported audio formats: {', '.join(sorted(audio_extensions))}")
                print(f"Supported video formats: {', '.join(sorted(video_extensions))}")
                sys.exit(1)

        # Generate SRT if requested
        if not args.no_srt:
            if args.srt_output:
                srt_path = args.srt_output
            else:
                if input_is_url:
                    # For URLs, save SRT in downloads folder with same name as video file
                    project_root = Path(__file__).parent.parent  # Go up from src/ to project root
                    downloads_dir = project_root / "downloads"

                    # Find the downloaded video file
                    video_files = [
                        f for f in downloads_dir.glob("*")
                        if f.is_file() and f.suffix.lower() in {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
                    ]

                    if video_files:
                        # Use the most recently modified video file
                        video_file = max(video_files, key=lambda x: x.stat().st_mtime)
                        srt_path = video_file.with_suffix('.srt')
                    else:
                        # Fallback if no video file found
                        srt_path = downloads_dir / "downloaded_video.srt"
                else:
                    # Default to same folder as input file
                    srt_path = input_path.parent / f"{input_path.stem}.srt"

            generate_srt_file(translated_transcript, str(srt_path))

        # Save final transcript JSON in output directory
        if input_is_url:
            final_transcript_path = output_dir / "translated_transcript.json"
        else:
            final_transcript_path = output_dir / f"{input_path.stem}_translated.json"
        with open(final_transcript_path, 'w', encoding='utf-8') as f:
            json.dump(translated_transcript, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*50}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*50}")
        if input_is_url:
            print(f"Input URL: {args.input_file}")
            if downloaded_video_path:
                print(f"Downloaded video: {downloaded_video_path}")
        else:
            print(f"Input file: {input_path}")
        print(f"Processing output: {output_dir}")
        print(f"Final transcript: {final_transcript_path}")
        if not args.no_srt:
            print(f"SRT file: {srt_path}")

        # Print final statistics
        if 'segments' in translated_transcript:
            segments = translated_transcript['segments']
            segments_with_zh = len([s for s in segments if 'zh' in s and s['zh']])
            print(f"\nFinal Statistics:")
            print(f"  - Total segments: {len(segments)}")
            print(f"  - Segments with Chinese translation: {segments_with_zh}")
            print(f"  - Translation coverage: {segments_with_zh/len(segments)*100:.1f}%")

        # Clean up output directory unless --keep-artifacts is specified
        should_keep_artifacts = args.keep_artifacts

        if not should_keep_artifacts and output_dir.exists():
            try:
                import shutil
                shutil.rmtree(output_dir)
                print(f"\n🧹 Cleaned up output directory: {output_dir}")
            except Exception as cleanup_error:
                print(f"\n⚠️  Warning: Could not clean up output directory: {cleanup_error}")
        elif should_keep_artifacts:
            print(f"\n📁 Kept artifacts in output directory: {output_dir}")
            print(f"   Use --keep-artifacts option to preserve them for debugging")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
