"""
Checkpoint Manager Module

This module handles saving and loading of checkpoint states to enable
resumable processing across all pipeline stages.
"""

import json
import os
from typing import Optional, Dict, Any
from datetime import datetime

from src.models import CheckpointState, ProcessingStage


class CheckpointManager:
    """Manages checkpoint save/load operations"""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        stage: ProcessingStage,
        data: Dict[str, Any],
        completed_items: int,
        total_items: int,
        input_file: str,
        output_directory: str,
        custom_path: Optional[str] = None
    ) -> str:
        """
        Save checkpoint state

        Args:
            stage: Current processing stage
            data: Arbitrary checkpoint data
            completed_items: Number of items completed
            total_items: Total number of items
            input_file: Original input file path
            output_directory: Output directory path
            custom_path: Optional custom checkpoint file path

        Returns:
            str: Path to saved checkpoint file
        """
        timestamp = datetime.now().isoformat()
        checkpoint = CheckpointState(
            stage=stage,
            completed_items=completed_items,
            total_items=total_items,
            data=data,
            timestamp=timestamp,
            input_file=input_file,
            output_directory=output_directory
        )

        if custom_path:
            checkpoint_path = custom_path
        else:
            # Generate filename based on input file and stage
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{input_name}_{stage.value}_{timestamp_str}.json"
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Save to file
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            f.write(checkpoint.to_json())

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Optional[CheckpointState]:
        """
        Load checkpoint from file

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            CheckpointState: Loaded checkpoint state or None if failed
        """
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                json_str = f.read()
            return CheckpointState.from_json(json_str)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def find_latest_checkpoint(
        self,
        input_file: str,
        stage: Optional[ProcessingStage] = None
    ) -> Optional[str]:
        """
        Find the latest checkpoint for a given input file and optionally stage

        Args:
            input_file: Input file path
            stage: Optional processing stage filter

        Returns:
            str: Path to latest checkpoint file or None
        """
        input_name = os.path.splitext(os.path.basename(input_file))[0]

        # List checkpoint files
        checkpoint_files = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json') and input_name in filename:
                if stage is None or stage.value in filename:
                    checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                    checkpoint_files.append(checkpoint_path)

        if not checkpoint_files:
            return None

        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoint_files[0]

    def cleanup_old_checkpoints(self, max_to_keep: int = 10, max_age_days: int = 7):
        """
        Clean up old checkpoint files

        Args:
            max_to_keep: Maximum number of checkpoints to keep
            max_age_days: Maximum age in days for checkpoints
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        checkpoint_files = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                mtime = os.path.getmtime(checkpoint_path)
                age_seconds = current_time - mtime
                checkpoint_files.append((checkpoint_path, mtime, age_seconds))

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)

        # Keep the most recent max_to_keep files
        files_to_keep = set()
        for i, (path, _, _) in enumerate(checkpoint_files):
            if i < max_to_keep:
                files_to_keep.add(path)

        # Remove files that are too old or exceed keep limit
        for path, _, age in checkpoint_files:
            if path not in files_to_keep and age > max_age_seconds:
                try:
                    os.remove(path)
                    print(f"Removed old checkpoint: {path}")
                except OSError as e:
                    print(f"Failed to remove checkpoint {path}: {e}")