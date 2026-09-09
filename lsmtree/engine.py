"""LSMTree: ties memtable + WAL + SSTables + compaction into one key-value store.

Write path:  put/delete -> WAL.append (fsync'd) -> MemTable -> maybe flush
Read path:   get -> MemTable (if present, done) -> SSTables newest-to-oldest
             (bloom filter + sparse index skip most of the work)
Recovery:    on startup, if a WAL file already exists (prior process died
             before flushing), replay it into a fresh MemTable before
             accepting new writes.

SSTable files are named `{seq:010d}.sst` in the data directory; seq is
monotonically increasing, so "newest" is just "highest seq number".
"""

from __future__ import annotations

import glob
import os
import re
from typing import Iterator, List, Optional, Tuple

from .compaction import compact_all
from .memtable import MemTable, TOMBSTONE
from .skiplist import NOT_FOUND
from .sstable import SSTableReader, SSTableWriter
from .wal import WAL, OP_DELETE, OP_PUT

# NOT_FOUND is re-exported here (via the `from .skiplist import NOT_FOUND`
# above) so callers can `from lsmtree.engine import NOT_FOUND` or, more
# commonly, `from lsmtree import NOT_FOUND`.

DEFAULT_MEMTABLE_THRESHOLD_BYTES = 1 << 20  # 1 MiB
DEFAULT_LEVEL0_COMPACTION_THRESHOLD = 4

_SSTABLE_RE = re.compile(r"^(\d{10})\.sst$")


def _sstable_name(seq: int) -> str:
    return f"{seq:010d}.sst"


class LSMTree:
    def __init__(
        self,
        data_dir: str,
        memtable_threshold_bytes: int = DEFAULT_MEMTABLE_THRESHOLD_BYTES,
        level0_compaction_threshold: int = DEFAULT_LEVEL0_COMPACTION_THRESHOLD,
    ):
        self.data_dir = data_dir
        self.memtable_threshold_bytes = memtable_threshold_bytes
        self.level0_compaction_threshold = level0_compaction_threshold
        os.makedirs(data_dir, exist_ok=True)

        self._wal_path = os.path.join(data_dir, "wal.log")
        self.memtable = MemTable()
        self._next_seq = self._discover_next_seq()

        # Crash recovery: replay any existing WAL into the fresh memtable
        # BEFORE opening it for further appends.
        for op, key, value in WAL.replay(self._wal_path):
            if op == OP_PUT:
                self.memtable.put(key, value)
            elif op == OP_DELETE:
                self.memtable.delete(key)
        self.wal = WAL(self._wal_path)

        self._closed = False

    # -- internal helpers --------------------------------------------------

    def _sstable_paths_newest_first(self) -> List[str]:
        paths = []
        for path in glob.glob(os.path.join(self.data_dir, "*.sst")):
            m = _SSTABLE_RE.match(os.path.basename(path))
            if m:
                paths.append((int(m.group(1)), path))
        paths.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in paths]

    def _discover_next_seq(self) -> int:
        max_seq = 0
        for path in glob.glob(os.path.join(self.data_dir, "*.sst")):
            m = _SSTABLE_RE.match(os.path.basename(path))
            if m:
                max_seq = max(max_seq, int(m.group(1)))
        return max_seq + 1

    # -- writes --------------------------------------------------------

    def put(self, key: bytes, value: bytes) -> None:
        self._check_open()
        self.wal.append(OP_PUT, key, value)
        self.memtable.put(key, value)
        self._maybe_flush()

    def delete(self, key: bytes) -> None:
        self._check_open()
        self.wal.append(OP_DELETE, key)
        self.memtable.delete(key)
        self._maybe_flush()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("LSMTree is closed")

    # -- reads -----------------------------------------------------------

    def get(self, key: bytes):
        """Returns the value bytes, or NOT_FOUND if the key is absent or deleted."""
        val = self.memtable.get(key)
        if val is TOMBSTONE:
            return NOT_FOUND
        if val is not NOT_FOUND:
            return val

        for path in self._sstable_paths_newest_first():
            with SSTableReader(path) as reader:
                found, value = reader.get(key)
                if found:
                    return NOT_FOUND if value is None else value
        return NOT_FOUND

    def scan(self, start: Optional[bytes] = None, end: Optional[bytes] = None) -> Iterator[Tuple[bytes, bytes]]:
        """Yield (key, value) pairs for start <= key <= end in ascending key order,
        merging the memtable and all SSTables with newest-wins semantics and
        tombstones filtered out.
        """
        sources: List[Iterator[Tuple[bytes, object]]] = [self.memtable.range(start, end)]
        readers = [SSTableReader(p) for p in self._sstable_paths_newest_first()]
        try:
            for reader in readers:
                sources.append(reader.scan(start, end))

            # Simple merge: since ranges are typically small compared to a full
            # table scan, materialize each source's remaining items and merge
            # with a min-heap, newest source (lowest index) wins on ties.
            import heapq

            heap = []
            iters = [iter(s) for s in sources]

            def push(i):
                try:
                    k, v = next(iters[i])
                    heapq.heappush(heap, (k, i, v))
                except StopIteration:
                    pass

            for i in range(len(iters)):
                push(i)

            while heap:
                key, i, value = heapq.heappop(heap)
                push(i)
                while heap and heap[0][0] == key:
                    _, dup_i, _ = heapq.heappop(heap)
                    push(dup_i)
                is_tombstone = value is TOMBSTONE or value is None
                if not is_tombstone:
                    yield key, value
        finally:
            for reader in readers:
                reader.close()

    # -- flush / compaction ------------------------------------------------

    def _maybe_flush(self) -> None:
        if self.memtable.approx_size_bytes >= self.memtable_threshold_bytes and not self.memtable.is_empty():
            self.flush()

    def flush(self) -> Optional[dict]:
        """Write the current memtable to a new SSTable, then clear the WAL.

        If the process crashes between the SSTable write (atomic, via
        os.replace) and the WAL clear, replay on next startup simply
        reconstructs the same memtable again -- redundant but harmless,
        since memtable entries always take precedence over SSTable entries
        for the same key.
        """
        if self.memtable.is_empty():
            return None

        seq = self._next_seq
        self._next_seq += 1
        path = os.path.join(self.data_dir, _sstable_name(seq))

        items = (
            (key, None if value is TOMBSTONE else value)
            for key, value in self.memtable.items()
        )
        meta = SSTableWriter.write(path, items, expected_items=len(self.memtable))

        self.memtable.clear()
        self.wal.clear()

        self._maybe_compact()
        return meta

    def _maybe_compact(self) -> Optional[dict]:
        paths = self._sstable_paths_newest_first()
        if len(paths) < self.level0_compaction_threshold:
            return None
        return self.compact()

    def compact(self) -> Optional[dict]:
        """Merge every current SSTable into one. Safe to call any time, including
        when there is 0 or 1 SSTables (a no-op then).
        """
        paths = self._sstable_paths_newest_first()
        if len(paths) < 2:
            return None

        seq = self._next_seq
        self._next_seq += 1
        output_path = os.path.join(self.data_dir, _sstable_name(seq))
        meta = compact_all(paths, output_path)

        for path in paths:
            os.remove(path)
        return meta

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self.wal.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def stats(self) -> dict:
        paths = self._sstable_paths_newest_first()
        return {
            "memtable_entries": len(self.memtable),
            "memtable_size_bytes": self.memtable.approx_size_bytes,
            "num_sstables": len(paths),
            "sstable_files": [os.path.basename(p) for p in paths],
        }
