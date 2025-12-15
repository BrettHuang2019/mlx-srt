from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from enum import Enum
import json
from datetime import datetime


class ProcessingStage(Enum):
    """Enum for processing stages"""
    INGESTION = "ingestion"
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
    SUBTITLE_GENERATION = "subtitle_generation"


@dataclass
class CheckpointState:
    """Represents a checkpoint state for resuming processing"""
    stage: ProcessingStage
    completed_items: int
    total_items: int
    data: Dict[str, Any]
    timestamp: str
    input_file: str
    output_directory: str

    def __post_init__(self):
        """Validate checkpoint state"""
        if self.completed_items < 0:
            raise ValueError("completed_items must be non-negative")
        if self.total_items <= 0:
            raise ValueError("total_items must be positive")
        if self.completed_items > self.total_items:
            raise ValueError("completed_items cannot exceed total_items")
        if not self.input_file.strip():
            raise ValueError("input_file cannot be empty")
        if not self.output_directory.strip():
            raise ValueError("output_directory cannot be empty")

    def progress_percentage(self) -> float:
        """Calculate progress as percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary"""
        data = asdict(self)
        # Convert ProcessingStage enum to string
        if isinstance(data.get("stage"), ProcessingStage):
            data["stage"] = data["stage"].value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create checkpoint from dictionary"""
        # Convert stage string to enum
        if isinstance(data.get("stage"), str):
            data["stage"] = ProcessingStage(data["stage"])
        return cls(**data)

    def to_json(self) -> str:
        """Convert checkpoint to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "CheckpointState":
        """Create checkpoint from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class ResumePoint:
    """Represents a point where processing can be resumed"""
    stage: ProcessingStage
    checkpoint_file: str
    completed_items: int
    total_items: int
    timestamp: str

    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.completed_items >= self.total_items

    def progress_percentage(self) -> float:
        """Calculate progress as percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100