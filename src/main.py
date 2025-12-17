#!/usr/bin/env python3
"""
MLX-SRT Main Orchestration Script

This script orchestrates the complete pipeline for:
1. Audio/video ingestion (extract audio from video if needed)
2. Speech transcription using Whisper
3. Segment refinement and cleanup
4. Translation from French to Chinese using MLX-LM
5. SRT subtitle generation

Usage:
    python src/main.py input_file.mp3
    python src/main.py input_file.mp4 [--output output_dir]
    python src/main.py --help
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ingestion.extract_audio import extract_audio
from transcription.whisper_transcriber import transcribe_audio
from translation.translate import translation_pipeline
from subtitle.generate_srt import generate_srt_from_segments


def process_audio_file(audio_path: str, output_dir: str) -> Dict[str, Any]:
    """Process audio file through transcription and translation pipeline.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save intermediate and final files

    Returns:
        Translated transcript with Chinese translations
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING AUDIO FILE")
    print(f"{'='*50}")
    print(f"Input: {audio_path}")
    print(f"Output: {output_dir}")

    # Step 1: Transcription
    print(f"\nStep 1: Transcribing audio...")
    whisper_result = transcribe_audio(audio_path)

    # Save transcription
    transcription_file = Path(output_dir) / "00_whisper_transcription.json"
    with open(transcription_file, 'w', encoding='utf-8') as f:
        json.dump(whisper_result, f, ensure_ascii=False, indent=2)
    print(f"Transcription saved: {transcription_file}")

    # Step 2: Translation pipeline
    print(f"\nStep 2: Running translation pipeline...")
    translated_transcript = translation_pipeline(whisper_result, output_dir)

    return translated_transcript


def process_video_file(video_path: str, output_dir: str) -> Dict[str, Any]:
    """Process video file through audio extraction, transcription, and translation.

    Args:
        video_path: Path to video file
        output_dir: Directory to save intermediate and final files

    Returns:
        Translated transcript with Chinese translations
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING VIDEO FILE")
    print(f"{'='*50}")
    print(f"Input: {video_path}")
    print(f"Output: {output_dir}")

    # Step 1: Extract audio from video
    print(f"\nStep 1: Extracting audio from video...")
    extracted_audio_path = Path(output_dir) / "extracted_audio.wav"
    extract_audio(video_path, str(extracted_audio_path))

    # Verify audio extraction
    if not extracted_audio_path.exists():
        raise RuntimeError(f"Audio extraction failed: {extracted_audio_path}")

    print(f"Audio extracted: {extracted_audio_path}")

    # Step 2: Process extracted audio through the same pipeline
    return process_audio_file(str(extracted_audio_path), output_dir)


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

  # Process and keep intermediate artifacts for debugging
  python src/main.py video.mp4 --keep-artifacts

  # Process with SRT output in specific location
  python src/main.py audio.mp3 --srt-output /path/to/subtitles.srt
        """
    )

    parser.add_argument(
        "input_file",
        help="Input audio or video file path"
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

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default to input filename + "_output"
        output_dir = input_path.parent / f"{input_path.stem}_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine file type
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

    file_extension = input_path.suffix.lower()

    try:
        # Process file based on type
        if file_extension in audio_extensions:
            translated_transcript = process_audio_file(str(input_path), str(output_dir))
        elif file_extension in video_extensions:
            translated_transcript = process_video_file(str(input_path), str(output_dir))
        else:
            print(f"Error: Unsupported file format: {file_extension}")
            print(f"Supported audio formats: {', '.join(sorted(audio_extensions))}")
            print(f"Supported video formats: {', '.join(sorted(video_extensions))}")
            sys.exit(1)

        # Generate SRT if requested - place at same folder level as input file
        if not args.no_srt:
            if args.srt_output:
                srt_path = args.srt_output
            else:
                # Default to same folder as input file
                srt_path = input_path.parent / f"{input_path.stem}.srt"

            generate_srt_file(translated_transcript, str(srt_path))

        # Save final transcript JSON in output directory
        final_transcript_path = output_dir / f"{input_path.stem}_translated.json"
        with open(final_transcript_path, 'w', encoding='utf-8') as f:
            json.dump(translated_transcript, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*50}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*50}")
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
        if not args.keep_artifacts and output_dir.exists():
            try:
                import shutil
                shutil.rmtree(output_dir)
                print(f"\n🧹 Cleaned up output directory: {output_dir}")
            except Exception as cleanup_error:
                print(f"\n⚠️  Warning: Could not clean up output directory: {cleanup_error}")
        elif args.keep_artifacts:
            print(f"\n📁 Kept artifacts in output directory: {output_dir}")
            print(f"   Use --keep-artifacts option to preserve them for debugging")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()