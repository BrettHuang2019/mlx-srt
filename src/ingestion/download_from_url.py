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
    use_cookies: bool = True
) -> Tuple[str, str]:
    """Download video from URL using yt-dlp.

    Args:
        url: URL to download from
        max_height: Maximum video height (default: 1080)
        min_height: Minimum video height (default: 720)

    Returns:
        Tuple of (downloaded_file_path, video_title)

    Raises:
        RuntimeError: If download fails
    """
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
            return str(downloaded_file), video_title

    except yt_dlp.DownloadError as e:
        raise RuntimeError(f"Download failed: {e}")
    except Exception as e:
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