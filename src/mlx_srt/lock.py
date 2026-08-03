"""Single-process gate using a kernel-owned advisory lock."""

from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


class LockTimeoutError(TimeoutError):
    pass


class RunLock:
    def __init__(
        self,
        input_file: str | Path,
        *,
        lock_path: str | Path | None = None,
        interval: float = 60,
        timeout: float = 1800,
        on_wait=None,
    ) -> None:
        self.input_file = str(Path(input_file).resolve())
        self.lock_path = Path(lock_path or Path.home() / ".mlx-srt" / "run.lock")
        self.interval = interval
        self.timeout = timeout
        self.on_wait = on_wait
        self.fd: int | None = None

    def _holder(self) -> dict:
        if self.fd is None:
            return {}
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            return json.loads(os.read(self.fd, 16_384).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def acquire(self) -> "RunLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                holder = self._holder()
                if time.monotonic() >= deadline:
                    message = (
                        f"Timed out waiting for pipeline lock held by pid={holder.get('pid', 'unknown')} "
                        f"input={holder.get('input', 'unknown')}"
                    )
                    os.close(self.fd)
                    self.fd = None
                    raise LockTimeoutError(message)
                if self.on_wait:
                    self.on_wait(holder)
                time.sleep(min(self.interval, max(0, deadline - time.monotonic())))
        metadata = json.dumps({
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": self.input_file,
        }).encode()
        os.ftruncate(self.fd, 0)
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.write(self.fd, metadata)
        os.fsync(self.fd)
        return self

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def check_available_memory(min_ram_gb: float) -> tuple[bool, float]:
    try:
        import psutil
    except ImportError:
        return True, 0.0
    available = psutil.virtual_memory().available / (1024 ** 3)
    return available >= min_ram_gb, available
