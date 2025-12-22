#!/usr/bin/env python3
"""
Task Manager for MLX-SRT

Handles resource checking and concurrent task management:
- RAM availability checking (requires minimum 7GB free)
- Running task detection via state files
- Wait mechanism with countdown display
"""

import json
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_min_ram_gb() -> int:
    """Get minimum RAM requirement from config"""
    config = load_config()
    return config.get('system', {}).get('min_ram_gb', 7)


def get_task_check_interval() -> int:
    """Get task check interval from config (in seconds)"""
    config = load_config()
    return config.get('system', {}).get('task_check_interval', 5)


def get_max_wait_time() -> int:
    """Get maximum wait time from config (in seconds)"""
    config = load_config()
    max_wait_minutes = config.get('system', {}).get('max_wait_time_minutes', 30)
    return max_wait_minutes * 60  # Convert to seconds


def check_available_memory() -> tuple[bool, float]:
    """Check if system has at least the minimum required RAM.

    Returns:
        Tuple of (has_enough_ram, available_ram_gb)
    """
    try:
        import psutil
        min_ram_gb = get_min_ram_gb()
        available_gb = psutil.virtual_memory().available / (1024**3)
        return available_gb >= min_ram_gb, available_gb
    except ImportError:
        print("Warning: psutil not installed. Skipping RAM check.")
        print("Install with: pip install psutil")
        return True, 0.0


def find_running_tasks() -> Optional[Dict[str, Any]]:
    """Search for state.json files with status='running' across output directories.

    Scans the downloads folder and its subdirectories for any state.json files
    that indicate a running pipeline.

    Returns:
        Dictionary with task info if running task found, None otherwise
    """
    project_root = Path(__file__).parent.parent
    downloads_dir = project_root / "downloads"

    search_dirs = [downloads_dir]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for state_file in search_dir.rglob("state.json"):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                pipeline_info = state.get("pipeline_info", {})
                if pipeline_info.get("status") == "running":
                    return {
                        "state_file": str(state_file),
                        "pipeline_id": pipeline_info.get("pipeline_id", "unknown"),
                        "start_time": pipeline_info.get("start_time", "unknown"),
                        "output_dir": str(state_file.parent)
                    }
            except (json.JSONDecodeError, KeyError, IOError):
                continue

    return None


def wait_for_task_completion(task_info: Dict[str, Any]) -> bool:
    """Wait for running task to complete with countdown display.

    Polls the task's state file at configured interval and displays a countdown.
    User can press Ctrl+C to abort the wait.
    Will timeout after max_wait_time from config.

    Args:
        task_info: Dictionary with task information from find_running_tasks()

    Returns:
        True if task completed, False if user aborted or timeout
    """
    state_file = Path(task_info["state_file"])
    pipeline_id = task_info["pipeline_id"]
    start_time = task_info["start_time"]

    check_interval = get_task_check_interval()
    max_wait_seconds = get_max_wait_time()

    print(f"\n{'='*50}")
    print(f"⚠️  ANOTHER TASK IS RUNNING")
    print(f"{'='*50}")
    print(f"Pipeline ID: {pipeline_id}")
    print(f"Started: {start_time}")
    print(f"Output directory: {task_info['output_dir']}")
    print(f"\nWaiting for the task to complete...")
    print(f"Press Ctrl+C to abort the wait and exit.")
    print(f"Maximum wait time: {max_wait_seconds // 60} minutes")
    print(f"{'='*50}\n")

    wait_seconds = 0

    try:
        while wait_seconds < max_wait_seconds:
            time.sleep(check_interval)
            wait_seconds += check_interval

            # Check if task completed
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                current_status = state.get("pipeline_info", {}).get("status")

                if current_status != "running":
                    print(f"\n✓ Task completed! Status: {current_status}")
                    return True

            except (json.JSONDecodeError, IOError):
                print(f"\n⚠️  Could not read state file, assuming task completed")
                return True

            # Display countdown with remaining time
            minutes = wait_seconds // 60
            seconds = wait_seconds % 60
            remaining_minutes = (max_wait_seconds - wait_seconds) // 60
            remaining_seconds = (max_wait_seconds - wait_seconds) % 60
            print(f"⏳ Waiting... {minutes}m {seconds}s elapsed | "
                  f"Time remaining: {remaining_minutes}m {remaining_seconds}s (checking every {check_interval}s)")

        # Timeout reached
        print(f"\n\n⏱️  TIMEOUT: Maximum wait time of {max_wait_seconds // 60} minutes reached")
        print(f"The task is still running: {pipeline_id}")
        print(f"Exiting without starting a new task.")
        print(f"\n💡 Tip: You can increase 'max_wait_time_minutes' in config.yaml if needed.")
        return False

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Wait aborted by user (Ctrl+C)")
        print(f"The other task is still running.")
        print(f"Exiting without starting a new task.")
        return False


def check_system_resources() -> bool:
    """Perform all system resource checks before starting a new task.

    Checks:
    1. No other running tasks (will wait with timeout if found)
    2. Available RAM (must have ≥ minimum configured GB free)

    Returns:
        True if all checks pass, False otherwise
    """
    print(f"\n{'='*50}")
    print(f"SYSTEM RESOURCE CHECK")
    print(f"{'='*50}")

    # Check for running tasks FIRST (before RAM check)
    running_task = find_running_tasks()
    if running_task:
        print(f"⚠️  Running task detected: {running_task['pipeline_id']}")
        completed = wait_for_task_completion(running_task)
        if not completed:
            return False
    else:
        print(f"✓ No other running tasks detected")

    # Check RAM AFTER waiting for running tasks to complete
    min_ram_gb = get_min_ram_gb()
    has_enough_ram, available_gb = check_available_memory()
    if available_gb > 0:
        status = "✓" if has_enough_ram else "✗"
        print(f"{status} Available RAM: {available_gb:.2f} GB (minimum: {min_ram_gb} GB)")

        if not has_enough_ram:
            print(f"\n❌ ERROR: Insufficient RAM available")
            print(f"   Required: {min_ram_gb} GB")
            print(f"   Available: {available_gb:.2f} GB")
            print(f"\n   Please free up memory or wait for other tasks to complete.")
            return False

    print(f"{'='*50}")
    print(f"✓ All system checks passed")
    print(f"{'='*50}\n")

    return True
