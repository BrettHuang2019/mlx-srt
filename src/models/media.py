from dataclasses import dataclass
from typing import Optional
from enum import Enum


class MediaType(Enum):
    """Enum for media types"""
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class MediaInfo:
    """Represents metadata about a media file"""
    file_path: str
    media_type: MediaType
    duration: float  # seconds
    codec: str
    sample_rate: Optional[int] = None  # Hz, for audio
    channels: Optional[int] = None     # number of audio channels
    fps: Optional[float] = None       # frames per second, for video
    width: Optional[int] = None       # pixels, for video
    height: Optional[int] = None      # pixels, for video

    def __post_init__(self):
        """Validate media info"""
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if not self.codec.strip():
            raise ValueError("codec cannot be empty")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels is not None and self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive")


@dataclass
class AudioFile:
    """Represents an extracted audio file"""
    file_path: str
    sample_rate: int    # Hz
    channels: int       # number of audio channels
    duration: float     # seconds
    format: str         # wav, mp3, etc.

    def __post_init__(self):
        """Validate audio file info"""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if not self.format.strip():
            raise ValueError("format cannot be empty")