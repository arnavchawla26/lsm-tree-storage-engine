"""The most important test in this repo: does WAL replay actually survive a
real, unclean process death, not just a simulated one?

Closing file handles cleanly (as most "crash recovery" tests do) never
exercises the code path that matters -- an in-flight write, an OS write
buffer, a process that never got to run its `finally`/`atexit` cleanup. This
test instead launches a real subprocess, sends it SIGKILL (unblockable,
untrappable) at an arbitrary moment while it is actively writing, and then
verifies a *fresh* LSMTree process can open the same data directory and
recover exactly the prefix of writes that were durably acknowledged before
the kill.
"""

import os
import re
import signal
import subprocess
import sys
import time

import pytest

from lsmtree.engine import LSMTree
from lsmtree.skiplist import NOT_FOUND

_KEY_RE = re.compile(r"^key-\d{6}$")


def _read_acked_keys(progress_path: str):
    if not os.path.exists(progress_path):
        return []
    with open(progress_path, "r") as fh:
        lines = fh.read().splitlines()
    # A line could be partially written if the kill landed mid-write() to
    # this bookkeeping file; keep only well-formed lines, which is a lower
    # bound on what was truly acknowledged (never an over-count).
    return [line for line in lines if _KEY_RE.match(line)]


def _wait_for_ready(progress_path: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    ready_path = progress_path + ".ready"
    while not os.path.exists(ready_path):
        if time.monotonic() > deadline:
            raise TimeoutError("crash_writer subprocess never became ready")
        time.sleep(0.005)


@pytest.mark.parametrize("kill_delay_seconds", [0.01, 0.05, 0.2])
def test_sigkill_mid_write_recovers_exactly_the_acked_prefix(tmp_path, kill_delay_seconds):
    data_dir = str(tmp_path / "data")
    progress_path = str(tmp_path / "progress.txt")
    num_keys = 4000  # enough puts that the process is very likely still writing at kill time

    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.helpers.crash_writer", data_dir, progress_path, str(num_keys)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    _wait_for_ready(progress_path)  # wait past process-launch/import overhead
    time.sleep(kill_delay_seconds)  # then let it get partway through the write loop
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)
    assert proc.returncode != 0  # confirm it really was killed, not a clean finish

    acked_keys = _read_acked_keys(progress_path)
    assert 0 < len(acked_keys) < num_keys, (
        "test is only meaningful if the kill landed strictly mid-run; "
        f"got {len(acked_keys)} acked keys out of {num_keys}"
    )

    # A fresh LSMTree opening the same directory must replay the WAL and
    # recover, without raising, every key that was acknowledged.
    db = LSMTree(data_dir)
    try:
        for key in acked_keys:
            i = int(key.split("-")[1])
            expected_value = f"value-{i:06d}".encode()
            assert db.get(key.encode()) == expected_value, f"lost acknowledged write for {key}"

        # The writer loop is strictly sequential (put() fully completes,
        # including fsync, before the next iteration's progress line is even
        # written), so at most ONE key beyond the last acked line -- the one
        # whose put() may have finished just as the kill landed, before its
        # progress line was written -- can legitimately exist. Anything
        # further out was never even attempted and must be absent.
        last_acked_index = max(int(k.split("-")[1]) for k in acked_keys)
        for i in range(last_acked_index + 2, min(last_acked_index + 50, num_keys)):
            assert db.get(f"key-{i:06d}".encode()) is NOT_FOUND
    finally:
        db.close()


def test_recovery_after_kill_then_further_writes_still_works(tmp_path):
    """After recovering from a kill, the engine must be fully usable again --
    new writes, further crash-safe WAL entries, all of it."""
    data_dir = str(tmp_path / "data")
    progress_path = str(tmp_path / "progress.txt")

    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.helpers.crash_writer", data_dir, progress_path, "2000"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    time.sleep(0.05)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)

    db = LSMTree(data_dir)
    db.put(b"post-recovery-key", b"post-recovery-value")
    assert db.get(b"post-recovery-key") == b"post-recovery-value"
    db.flush()
    db.close()

    # And a second reopen (now with an SSTable present, not just a WAL) also works.
    db2 = LSMTree(data_dir)
    assert db2.get(b"post-recovery-key") == b"post-recovery-value"
    db2.close()
