import multiprocessing
import os
import signal
import time

import pytest

from mlx_srt.lock import LockTimeoutError, RunLock


def _hold_lock(path, ready):
    with RunLock("holder.mp4", lock_path=path, interval=0.01, timeout=1):
        ready.set()
        time.sleep(10)


def test_contender_times_out_and_killed_holder_releases(tmp_path):
    lock_path = tmp_path / "run.lock"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    holder = context.Process(target=_hold_lock, args=(lock_path, ready))
    holder.start()
    assert ready.wait(5)
    try:
        with pytest.raises(LockTimeoutError, match=r"holder\.mp4"):
            RunLock("waiter.mp4", lock_path=lock_path, interval=0.01, timeout=0.03).acquire()
    finally:
        os.kill(holder.pid, signal.SIGKILL)
        holder.join(5)
    with RunLock("next.mp4", lock_path=lock_path, interval=0.01, timeout=0.2):
        assert lock_path.exists()
