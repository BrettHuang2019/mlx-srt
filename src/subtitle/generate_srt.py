"""Generate SRT files from translated segments."""

import json
from pathlib import Path
from typing import Dict, Any, List


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def generate_srt_from_segments(segments: List[Dict[str, Any]]) -> str:
    """Generate SRT content from translated segments with dual language support.

    Args:
        segments: List of segment dictionaries with translation fields

    Returns:
        Formatted SRT content as string with original and translated text
    """
    srt_lines = []

    for i, segment in enumerate(segments, 1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])

        original_text = segment.get("text", "").strip()
        translated_text = segment.get("zh", "").strip()

        if translated_text:
            # Include both original and translated text
            text_lines = [original_text, translated_text]
        else:
            # Only original text if no translation available
            text_lines = [original_text]

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.extend(text_lines)
        srt_lines.append("")  # Empty line between entries

    return "\n".join(srt_lines)


def main():
    """Main function to generate SRT file from JSON input."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate SRT file from translated segments")
    parser.add_argument("input_json", help="Path to JSON file with translated segments")
    parser.add_argument("-o", "--output", help="Output SRT file path (default: input filename with .srt extension)")

    args = parser.parse_args()

    # Read input JSON
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Generate SRT content
    srt_content = generate_srt_from_segments(data["segments"])

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.input_json).with_suffix('.srt')

    # Write SRT file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    print(f"SRT file generated: {output_path}")


if __name__ == "__main__":
    main()