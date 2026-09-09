"""Subprocess helper for the crash-recovery test.

Run as: python -m tests.helpers.crash_writer <data_dir> <progress_file> <num_keys>

Writes `num_keys` puts into an LSMTree at `data_dir`, one per iteration,
appending the key it just successfully wrote to `progress_file` (flushed)
right after each `db.put()` call returns. Because `LSMTree.put()` itself
fsyncs the WAL record before returning, any key recorded in the progress
file is guaranteed durable -- so the parent test process can kill this
process at an arbitrary moment and know a reliable lower bound on which
writes must survive. (The true durable set can be at most one key ahead of
the last acked line, since `db.put()` fully completes, fsync included,
before the next line's `progress.write()` even starts -- iterations are
strictly sequential in this single-threaded loop.)

`progress_file + ".ready"` is created immediately after the LSMTree opens
and before any puts start, so the parent can wait for real work to begin
rather than guessing a fixed startup delay (process-launch and import time
is not reliably sub-millisecond).

A large memtable threshold is used so this test exercises WAL replay
specifically (the interesting/fragile path for a kill mid-write), not a
flush that happened to already run.
"""

import sys

from lsmtree.engine import LSMTree


def main() -> None:
    data_dir, progress_path, num_keys_str = sys.argv[1], sys.argv[2], sys.argv[3]
    num_keys = int(num_keys_str)

    db = LSMTree(data_dir, memtable_threshold_bytes=1 << 30)
    with open(progress_path + ".ready", "w") as ready:
        ready.write("ready\n")

    with open(progress_path, "a", buffering=1) as progress:
        for i in range(num_keys):
            key = f"key-{i:06d}"
            db.put(key.encode(), f"value-{i:06d}".encode())
            progress.write(key + "\n")
            progress.flush()


if __name__ == "__main__":
    main()
