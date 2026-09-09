"""MemTable: the mutable, in-memory sorted layer of the LSM-tree.

Backed by the from-scratch SkipList so it stays sorted at all times, which
lets both point lookups and range scans hit the active memtable directly
without any extra sort step. A deleted key is stored as a TOMBSTONE marker
rather than removed outright, so that on flush the tombstone is written to
the SSTable and can correctly shadow an older value for the same key that
already lives in an on-disk SSTable.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

from .skiplist import SkipList, NOT_FOUND

TOMBSTONE = object()


class MemTable:
    def __init__(self):
        self._skiplist = SkipList()
        self._approx_size_bytes = 0

    def put(self, key: bytes, value: bytes) -> None:
        old = self._skiplist.search(key)
        self._skiplist.insert(key, value)
        self._approx_size_bytes += len(key) + len(value)
        if old is not NOT_FOUND and old is not TOMBSTONE:
            self._approx_size_bytes -= len(key) + len(old)
        elif old is TOMBSTONE:
            pass  # tombstone had no value bytes counted

    def delete(self, key: bytes) -> None:
        old = self._skiplist.search(key)
        self._skiplist.insert(key, TOMBSTONE)
        if old is NOT_FOUND:
            self._approx_size_bytes += len(key)
        elif old is not TOMBSTONE:
            self._approx_size_bytes -= len(old)

    def get(self, key: bytes):
        """Returns the value bytes, TOMBSTONE (deleted), or NOT_FOUND (absent)."""
        return self._skiplist.search(key)

    def __len__(self) -> int:
        return len(self._skiplist)

    @property
    def approx_size_bytes(self) -> int:
        return self._approx_size_bytes

    def is_empty(self) -> bool:
        return len(self._skiplist) == 0

    def items(self) -> Iterator[Tuple[bytes, object]]:
        """Sorted (key, value-or-TOMBSTONE) pairs, for flushing to an SSTable."""
        return iter(self._skiplist)

    def range(self, start: Optional[bytes], end: Optional[bytes]):
        return self._skiplist.range(start, end)

    def clear(self) -> None:
        self._skiplist = SkipList()
        self._approx_size_bytes = 0
