from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class Segment:
    """Represents a timestamped text segment from transcription"""
    id: int
    text: str
    start_time: float  # seconds
    end_time: float    # seconds

    @property
    def duration(self) -> float:
        """Calculate segment duration in seconds"""
        return self.end_time - self.start_time

    def __post_init__(self):
        """Validate segment data"""
        if self.id < 0:
            raise ValueError("id must be non-negative")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not self.text.strip():
            raise ValueError("text cannot be empty")


@dataclass
class TranscriptionResult:
    """Represents the full transcription result from MLX Whisper"""
    text: str  # Full transcribed text
    segments: List[Segment]  # List of segments with timestamps
    language: Optional[str] = None  # Detected language

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "segments": [
                {
                    "id": seg.id,
                    "text": seg.text,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time
                }
                for seg in self.segments
            ],
            "language": self.language
        }


@dataclass
class TranslatedSegment:
    """Represents a translated segment with preserved timestamps"""
    original_text: str
    translated_text: str
    start_time: float
    end_time: float
    source_language: str
    target_language: str
    original_confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        """Calculate segment duration in seconds"""
        return self.end_time - self.start_time

    def __post_init__(self):
        """Validate translated segment data"""
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not self.original_text.strip():
            raise ValueError("original_text cannot be empty")
        if not self.translated_text.strip():
            raise ValueError("translated_text cannot be empty")
        if not self.source_language.strip():
            raise ValueError("source_language cannot be empty")
        if not self.target_language.strip():
            raise ValueError("target_language cannot be empty")