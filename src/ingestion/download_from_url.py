#!/usr/bin/env python3
"""
URL Video Download Module using yt-dlp

This module handles downloading videos from URLs using yt-dlp with support for:
- Best quality 1080p, minimum 720p
- Chrome cookies for authentication
- Progress tracking
- Error handling and logging
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import urllib.parse
import subprocess
import json

try:
    import yt_dlp
except ImportError:
    # Try to install yt-dlp dynamically
    import subprocess
    print("Installing yt-dlp dependency...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        import yt_dlp
        print("yt-dlp installed successfully")
    except Exception as e:
        print(f"Error: Could not install yt-dlp: {e}")
        print("Please install manually with: pip install yt-dlp")
        sys.exit(1)


def is_url(input_path: str) -> bool:
    """Check if input is a URL.

    Args:
        input_path: Input string to check

    Returns:
        True if input is a URL, False otherwise
    """
    parsed = urllib.parse.urlparse(input_path)
    return bool(parsed.scheme and parsed.netloc)


def find_chrome_cookies() -> Optional[str]:
    """Find Chrome cookies database on macOS.

    Returns:
        Path to Chrome cookies database, or None if not found
    """
    home = Path.home()
    chrome_paths = [
        home / "Library/Application Support/Google/Chrome/Default/Cookies",
        home / "Library/Application Support/Google/Chrome/Profile 1/Cookies",
        home / "Library/Application Support/Google/Chrome/Profile 2/Cookies",
        home / "Library/Application Support/Google/Chrome/Profile 3/Cookies",
    ]

    for path in chrome_paths:
        if path.exists():
            return str(path)

    return None


def download_video_from_url(
    url: str,
    max_height: int = 1080,
    min_height: int = 720,
    use_cookies: bool = True,
    output_dir: str = None
) -> Tuple[str, str]:
    """Download video from URL using yt-dlp.

    Args:
        url: URL to download from
        max_height: Maximum video height (default: 1080)
        min_height: Minimum video height (default: 720)
        use_cookies: Whether to use browser cookies (default: True)
        output_dir: Output directory for state saving (optional)

    Returns:
        Tuple of (downloaded_file_path, video_title)

    Raises:
        RuntimeError: If download fails
    """
    # Create initial state before download starts
    if output_dir:
        pipeline_id = create_initial_download_state(url, output_dir)
        if pipeline_id:
            print(f"🎥 Downloading video from URL: {url}")
            print(f"   Pipeline ID: {pipeline_id}")
        else:
            print(f"🎥 Downloading video from URL: {url}")
    else:
        print(f"🎥 Downloading video from URL: {url}")

    # Always download to the downloads folder
    project_root = Path(__file__).parent.parent.parent  # Go up from src/ingestion/ to project root
    downloads_dir = project_root / "downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    output_path = str(downloads_dir)

    # Configure yt-dlp options
    ydl_opts = {
        'format': f'best[height<={max_height}][height>={min_height}]/best[height>={min_height}]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'restrictfilenames': True,
        'no_warnings': False,
        'progress_hooks': [progress_hook],
    }

    # Note: For authentication, yt-dlp can use browser cookies automatically
    # You can export browser cookies using a browser extension like "Get cookies.txt LOCALLY"
    # For now, we'll proceed without explicit cookie handling to avoid SQLite encoding issues
    chrome_cookies = find_chrome_cookies()
    if chrome_cookies:
        print("🍪 Chrome cookies found - some content may require manual cookie export for authentication")
    else:
        print("ℹ️  No Chrome cookies found - proceeding without authentication")

    # Download video
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get title
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'downloaded_video')

            print(f"📹 Video title: {video_title}")
            print(f"📺 Duration: {info.get('duration', 'unknown')} seconds")
            print(f"🎬 Uploader: {info.get('uploader', 'unknown')}")

            # Download the video
            ydl.download([url])

            # Find the downloaded file
            downloaded_files = list(Path(output_path).glob(f"{video_title}*"))
            if not downloaded_files:
                # Try alternative matching pattern
                downloaded_files = list(Path(output_path).glob("*"))
                downloaded_files = [f for f in downloaded_files if f.is_file() and f.suffix in ['.mp4', '.mkv', '.webm', '.avi']]

            if not downloaded_files:
                raise RuntimeError("No downloaded video file found")

            # Use the most recently modified file
            downloaded_file = max(downloaded_files, key=lambda x: x.stat().st_mtime)

            print(f"✅ Video downloaded successfully: {downloaded_file}")

            # Save download state if output_dir is provided
            if output_dir:
                save_download_state(url, str(downloaded_file), video_title, output_dir)

            return str(downloaded_file), video_title

    except yt_dlp.DownloadError as e:
        # Save failed download state if output_dir is provided
        if output_dir:
            save_download_state(url, None, None, output_dir, "failed", str(e))
        raise RuntimeError(f"Download failed: {e}")
    except Exception as e:
        # Save failed download state if output_dir is provided
        if output_dir:
            save_download_state(url, None, None, output_dir, "failed", str(e))
        raise RuntimeError(f"Unexpected error during download: {e}")


def progress_hook(d):
    """Progress hook for yt-dlp download progress.

    Args:
        d: Download progress dictionary from yt-dlp
    """
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A').strip()
        speed = d.get('_speed_str', 'N/A').strip()
        eta = d.get('_eta_str', 'N/A').strip()

        # Clean up the strings
        percent = percent.replace('%', '')

        print(f"⬇️  Download progress: {percent}% | Speed: {speed} | ETA: {eta}")

    elif d['status'] == 'finished':
        total_size = d.get('_total_bytes_str', 'unknown').strip()
        print(f"✅ Download completed! Size: {total_size}")

    elif d['status'] == 'error':
        print(f"❌ Download error: {d.get('error', 'Unknown error')}")


def get_video_info(url: str) -> dict:
    """Get video information without downloading.

    Args:
        url: URL to get info for

    Returns:
        Dictionary with video information
    """
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)

            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', 'Unknown'),
                'description': info.get('description', '')[:200] + '...' if info.get('description') else '',
                'formats': [
                    {
                        'format_id': f.get('format_id'),
                        'ext': f.get('ext'),
                        'height': f.get('height'),
                        'width': f.get('width'),
                        'fps': f.get('fps'),
                        'filesize': f.get('filesize')
                    }
                    for f in info.get('formats', []) if f.get('vcodec') != 'none'
                ]
            }
    except Exception as e:
        raise RuntimeError(f"Failed to get video info: {e}")


def create_initial_download_state(url: str, output_dir: str) -> str:
    """Create initial state file before download starts.

    Args:
        url: URL to be downloaded
        output_dir: Directory where state.json should be saved

    Returns:
        Pipeline ID of the created state
    """
    from datetime import datetime
    import uuid

    if not output_dir:
        return None

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    state = {
        "pipeline_info": {
            "pipeline_id": pipeline_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "last_checkpoint": datetime.now().isoformat()
        },
        "download_info": {
            "url": url,
            "downloaded_file": None,
            "video_title": None,
            "download_completed": False,
            "download_time": None
        },
        "completed_steps": [],
        "current_step": "download",
        "steps": {
            "download": {
                "status": "running",
                "url": url,
                "start_time": datetime.now().isoformat()
            }
        }
    }

    state_file = Path(output_dir) / "state.json"

    # Save initial state
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"💾 Initial state created: {state_file}")
        print(f"   Pipeline ID: {pipeline_id}")
        return pipeline_id
    except Exception as e:
        print(f"⚠️  Warning: Could not create initial state: {e}")
        return None


def save_download_state(url: str, downloaded_file: str = None, video_title: str = None,
                       output_dir: str = None, status: str = "completed", error: str = None):
    """Save download state to state.json file.

    Args:
        url: URL that was downloaded
        downloaded_file: Path to downloaded file (None if failed)
        video_title: Title of the video (None if failed)
        output_dir: Directory where state.json should be saved
        status: Download status ("completed" or "failed")
        error: Error message if download failed
    """
    if not output_dir:
        return

    from datetime import datetime

    state_file = Path(output_dir) / "state.json"

    # Load existing state or create new one
    state = {}
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except:
            state = {}

    # Initialize state structure if needed
    if "download_info" not in state:
        state["download_info"] = {}

    if "pipeline_info" not in state:
        state["pipeline_info"] = {
            "status": "running",
            "last_checkpoint": datetime.now().isoformat()
        }

    if "steps" not in state:
        state["steps"] = {}

    if "completed_steps" not in state:
        state["completed_steps"] = []

    # Update download info
    state["download_info"].update({
        "url": url,
        "downloaded_file": downloaded_file,
        "video_title": video_title,
        "download_completed": status == "completed",
        "download_time": datetime.now().isoformat() if status == "completed" else None,
        "error": error
    })

    # Update step status
    state["steps"]["download"] = {
        "status": status,
        "url": url,
        "end_time": datetime.now().isoformat()
    }

    if downloaded_file:
        state["steps"]["download"]["downloaded_file"] = downloaded_file
        state["steps"]["download"]["video_title"] = video_title

        # Add to completed steps if not already there
        if "download" not in state["completed_steps"]:
            state["completed_steps"].append("download")
    elif status == "failed":
        state["steps"]["download"]["error"] = error

    state["pipeline_info"]["last_checkpoint"] = datetime.now().isoformat()

    # Save state
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"💾 Download state saved to: {state_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save download state: {e}")