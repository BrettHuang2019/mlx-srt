"""Small SRT parser and renderer shared by merge and translation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TIMESTAMP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})"
)


@dataclass
class Segment:
    id: int
    start: str
    end: str
    text: str


def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt(content: str) -> list[Segment]:
    segments = []
    for block in re.split(r"\r?\n(?:\s*\r?\n)+", content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            segment_id = int(lines[0].strip())
        except ValueError:
            continue
        match = _TIMESTAMP_RE.fullmatch(lines[1].strip())
        if match:
            segments.append(Segment(segment_id, match.group(1), match.group(2), "\n".join(lines[2:]).strip()))
    return segments


def render_srt(segments: list[Segment]) -> str:
    return "\n\n".join(
        f"{segment.id}\n{segment.start} --> {segment.end}\n{segment.text}"
        for segment in segments
    ) + ("\n" if segments else "")


def format_bilingual_subtitles(segments: list[Segment], translations: dict[int, str]) -> str:
    bilingual = [
        Segment(segment.id, segment.start, segment.end,
                segment.text + (f"\n{translations[segment.id]}" if translations.get(segment.id) else ""))
        for segment in segments
    ]
    return render_srt(bilingual)
