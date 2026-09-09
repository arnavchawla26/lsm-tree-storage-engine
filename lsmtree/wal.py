"""Write-ahead log: durability for the memtable.

Every put/delete is appended here (and fsync'd) before it is applied to the
in-memory memtable. If the process dies before the memtable is flushed to an
SSTable, replaying the WAL on the next startup reconstructs exactly the
memtable state that existed at the moment of the crash.

Record format (little-endian), one per op:
    [op: 1 byte]            1 = PUT, 2 = DELETE
    [key_len: 4 bytes]
    [key: key_len bytes]
    [value_len: 4 bytes]    0 for DELETE (no value follows)
    [value: value_len bytes]
    [crc32: 4 bytes]        CRC32 over everything above, this record only

The CRC lets replay detect a partially-written last record -- exactly what
happens when the process is killed mid-`write()` -- and stop cleanly at the
last good record instead of raising or silently reading garbage.
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import BinaryIO, Iterator, Optional, Tuple

OP_PUT = 1
OP_DELETE = 2

_HEADER = struct.Struct("<BII")  # op, key_len, value_len
_CRC = struct.Struct("<I")


class WAL:
    def __init__(self, path: str):
        self.path = path
        self._fh: Optional[BinaryIO] = open(path, "ab", buffering=0)

    def append(self, op: int, key: bytes, value: bytes = b"") -> None:
        body = _HEADER.pack(op, len(key), len(value)) + key + value
        crc = zlib.crc32(body) & 0xFFFFFFFF
        record = body + _CRC.pack(crc)
        self._fh.write(record)
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def clear(self) -> None:
        """Truncate the log to empty (called after a successful flush)."""
        self.close()
        with open(self.path, "wb"):
            pass
        self._fh = open(self.path, "ab", buffering=0)

    @staticmethod
    def replay(path: str) -> Iterator[Tuple[int, bytes, bytes]]:
        """Yield (op, key, value) records from an existing WAL file.

        Stops at the first record that is truncated or fails its CRC check
        (which is exactly what a mid-write crash leaves behind) rather than
        raising, since everything before that point is still valid,
        durable data.
        """
        if not os.path.exists(path):
            return
        with open(path, "rb") as fh:
            data = fh.read()
        offset = 0
        n = len(data)
        while offset < n:
            if offset + _HEADER.size > n:
                break  # truncated header
            op, key_len, value_len = _HEADER.unpack_from(data, offset)
            record_len = _HEADER.size + key_len + value_len + _CRC.size
            if offset + record_len > n:
                break  # truncated body/crc
            body = data[offset:offset + _HEADER.size + key_len + value_len]
            crc_stored, = _CRC.unpack_from(data, offset + _HEADER.size + key_len + value_len)
            if (zlib.crc32(body) & 0xFFFFFFFF) != crc_stored:
                break  # corrupt record
            key = body[_HEADER.size:_HEADER.size + key_len]
            value = body[_HEADER.size + key_len:]
            if op not in (OP_PUT, OP_DELETE):
                break
            yield op, key, value
            offset += record_len
