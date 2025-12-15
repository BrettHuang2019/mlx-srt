"""Transcription module for speech-to-text processing"""

from .whisper_transcriber import transcribe_audio

__all__ = [
    "transcribe_audio"
]