"""SSTable: an immutable, sorted on-disk table of key-value pairs.

File layout, all little-endian:

    MAGIC (4 bytes: b"LSST")
    -- data block --
    for each entry, in sorted key order:
        [key_len: 4][key][value_len: 4][value]
        value_len == 0xFFFFFFFF marks a tombstone (deleted key), no value bytes follow
    -- sparse index block --
    for every INDEX_INTERVAL-th entry:
        [key_len: 4][key][offset: 8]     offset = byte offset of that entry in the data block
    -- bloom filter block --
    [bloom_filter bytes, length given in footer]
    -- min/max key bytes --
    [min_key][max_key]                    (lengths recorded in the fixed footer below)
    -- footer (FIXED size, at a fixed offset from end-of-file) --
    [data_block_len: 8][index_offset: 8][index_len: 8][bloom_offset: 8][bloom_len: 8]
    [num_entries: 8][min_key_len: 4][max_key_len: 4]
    [MAGIC: 4]                            (again, for corruption sanity-check at the tail)

The footer is fixed-size and sits at a fixed distance from end-of-file, so
opening a reader can seek straight to it without scanning the file -- the
variable-length min/max keys are then found by walking backward from the
footer using the lengths it just gave us, rather than trying to parse
variable-length fields back-to-front (which is not generally possible: you
can't know where a length-prefixed field starts, reading backward, without
already knowing its length).

The sparse index means a point lookup does a binary search over ~num_entries
/ INDEX_INTERVAL index entries to find the containing block, then a short
linear scan of the data block from there -- O(log n) seeks, not O(n).
"""

from __future__ import annotations

import os
import struct
from typing import BinaryIO, Iterator, List, Optional, Tuple

from .bloom_filter import BloomFilter

MAGIC = b"LSST"
TOMBSTONE_MARKER = 0xFFFFFFFF
INDEX_INTERVAL = 16

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
# data_len, index_offset, index_len, bloom_offset, bloom_len, num_entries, min_key_len, max_key_len
_FOOTER = struct.Struct("<QQQQQQII")


def _write_entry(fh: BinaryIO, key: bytes, value: Optional[bytes]) -> None:
    fh.write(_U32.pack(len(key)))
    fh.write(key)
    if value is None:
        fh.write(_U32.pack(TOMBSTONE_MARKER))
    else:
        fh.write(_U32.pack(len(value)))
        fh.write(value)


class SSTableWriter:
    """Writes a sorted sequence of (key, value_or_None) pairs to a new SSTable file.

    `items` MUST already be sorted by key and contain no duplicate keys --
    callers (MemTable flush, compaction merge) are responsible for that.
    value of None means a tombstone.
    """

    @staticmethod
    def write(path: str, items: Iterator[Tuple[bytes, Optional[bytes]]], expected_items: int = 0) -> dict:
        tmp_path = path + ".tmp"
        index_entries: List[Tuple[bytes, int]] = []
        bloom = BloomFilter.for_capacity(max(expected_items, 1))
        min_key: Optional[bytes] = None
        max_key: Optional[bytes] = None
        num_entries = 0

        with open(tmp_path, "wb") as fh:
            fh.write(MAGIC)
            data_start = fh.tell()
            for i, (key, value) in enumerate(items):
                offset = fh.tell() - data_start
                if i % INDEX_INTERVAL == 0:
                    index_entries.append((key, offset))
                _write_entry(fh, key, value)
                bloom.add(key)
                if min_key is None:
                    min_key = key
                max_key = key
                num_entries += 1
            data_len = fh.tell() - data_start

            index_offset = fh.tell()
            for key, offset in index_entries:
                fh.write(_U32.pack(len(key)))
                fh.write(key)
                fh.write(_U64.pack(offset))
            index_len = fh.tell() - index_offset

            bloom_offset = fh.tell()
            bloom_bytes = bloom.to_bytes()
            fh.write(bloom_bytes)
            bloom_len = len(bloom_bytes)

            min_key_b = min_key or b""
            max_key_b = max_key or b""
            fh.write(min_key_b)
            fh.write(max_key_b)
            footer = _FOOTER.pack(
                data_len, index_offset, index_len, bloom_offset, bloom_len,
                num_entries, len(min_key_b), len(max_key_b),
            )
            fh.write(footer)
            fh.write(MAGIC)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_path, path)  # atomic on POSIX: never leaves a half-written file at `path`
        return {
            "num_entries": num_entries,
            "min_key": min_key,
            "max_key": max_key,
            "size_bytes": os.path.getsize(path),
        }


class SSTableReader:
    """Opens an existing SSTable file for point lookups, range scans, and full iteration."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        self._fh.seek(0, os.SEEK_END)
        file_len = self._fh.tell()

        self._fh.seek(0)
        magic = self._fh.read(4)
        if magic != MAGIC:
            raise ValueError(f"not a valid SSTable (bad header magic): {path}")

        tail_magic_start = file_len - 4
        self._fh.seek(tail_magic_start)
        if self._fh.read(4) != MAGIC:
            raise ValueError(f"not a valid SSTable (bad/truncated footer): {path}")

        # The footer is fixed-size and sits right before the tail magic, so
        # it can be located directly -- no need to scan for it.
        footer_start = tail_magic_start - _FOOTER.size
        if footer_start < 4:
            raise ValueError(f"not a valid SSTable (file too small for footer): {path}")
        self._fh.seek(footer_start)
        (
            data_len, index_offset, index_len, bloom_offset, bloom_len,
            num_entries, min_key_len, max_key_len,
        ) = _FOOTER.unpack(self._fh.read(_FOOTER.size))

        # min_key and max_key were written immediately before the footer, in
        # that order, so max_key ends where the footer begins.
        max_key_start = footer_start - max_key_len
        min_key_start = max_key_start - min_key_len
        self._fh.seek(min_key_start)
        min_key = self._fh.read(min_key_len)
        max_key = self._fh.read(max_key_len)

        self.min_key = min_key if min_key_len else None
        self.max_key = max_key if max_key_len else None
        self.num_entries = num_entries
        self._data_start = 4  # right after header MAGIC
        self._data_len_bytes = data_len  # exact length of the data block, from the footer

        self._fh.seek(bloom_offset)
        self.bloom = BloomFilter.from_bytes(self._fh.read(bloom_len))

        self._index: List[Tuple[bytes, int]] = []
        self._fh.seek(index_offset)
        remaining = index_len
        while remaining > 0:
            key_len, = _U32.unpack(self._fh.read(4))
            key = self._fh.read(key_len)
            offset, = _U64.unpack(self._fh.read(8))
            self._index.append((key, offset))
            remaining -= 4 + key_len + 8

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _might_have(self, key: bytes) -> bool:
        if self.min_key is not None and (key < self.min_key or key > self.max_key):
            return False
        return self.bloom.might_contain(key)

    def get(self, key: bytes):
        """Returns value bytes, None (tombstone), or NOT_FOUND-style False-ish absence.

        Returns a 2-tuple (found: bool, value: Optional[bytes]) where value is
        None both for "not found" and for "tombstone" -- callers must check `found`.
        """
        if not self._might_have(key):
            return False, None

        # Binary search the sparse index for the last entry whose key <= target.
        lo, hi = 0, len(self._index) - 1
        start_offset = self._data_start
        if self._index:
            best = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if self._index[mid][0] <= key:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best >= 0:
                start_offset = self._data_start + self._index[best][1]

        self._fh.seek(start_offset)
        for scan_key, value in self._scan_from(start_offset):
            if scan_key == key:
                return True, value
            if scan_key > key:
                break
        return False, None

    def _scan_from(self, byte_offset: int) -> Iterator[Tuple[bytes, Optional[bytes]]]:
        self._fh.seek(byte_offset)
        data_end = self._data_start + self._data_len_bytes  # exact end of data block, per the footer
        while self._fh.tell() < data_end:
            key_len_b = self._fh.read(4)
            if len(key_len_b) < 4:
                return
            key_len, = _U32.unpack(key_len_b)
            key = self._fh.read(key_len)
            value_len, = _U32.unpack(self._fh.read(4))
            if value_len == TOMBSTONE_MARKER:
                yield key, None
            else:
                value = self._fh.read(value_len)
                yield key, value

    def __iter__(self) -> Iterator[Tuple[bytes, Optional[bytes]]]:
        """Full sorted iteration over every entry (used by compaction)."""
        yield from self._scan_from(self._data_start)

    def scan(self, start: Optional[bytes], end: Optional[bytes]) -> Iterator[Tuple[bytes, Optional[bytes]]]:
        for key, value in self:
            if start is not None and key < start:
                continue
            if end is not None and key > end:
                break
            yield key, value
