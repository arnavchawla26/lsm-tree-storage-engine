"""A from-scratch Bloom filter used per-SSTable to skip disk reads for keys
that are definitely absent.

Uses double hashing (Kirsch-Mitzenmacher): two independent base hashes
(derived from a single blake2b digest, no external deps) are combined as
h_i(x) = h1(x) + i * h2(x) to simulate k independent hash functions, which
is statistically indistinguishable from k truly independent hashes for
Bloom filter purposes and much cheaper than hashing k times.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Iterable


class BloomFilter:
    def __init__(self, num_bits: int, num_hashes: int):
        if num_bits < 8:
            num_bits = 8
        if num_hashes < 1:
            num_hashes = 1
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self._bits = bytearray((num_bits + 7) // 8)

    @classmethod
    def for_capacity(cls, expected_items: int, false_positive_rate: float = 0.01) -> "BloomFilter":
        """Size a filter for expected_items with target false_positive_rate."""
        expected_items = max(1, expected_items)
        m = -(expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2)
        num_bits = max(8, int(math.ceil(m)))
        k = max(1, round((num_bits / expected_items) * math.log(2)))
        return cls(num_bits, k)

    def _hashes(self, item: bytes):
        digest = hashlib.blake2b(item, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little")
        if h2 % 2 == 0:  # ensure h2 is odd so it's coprime with power-of-two-ish sizes
            h2 += 1
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    def add(self, item: bytes) -> None:
        for bit_index in self._hashes(item):
            self._bits[bit_index // 8] |= 1 << (bit_index % 8)

    def add_all(self, items: Iterable[bytes]) -> None:
        for item in items:
            self.add(item)

    def might_contain(self, item: bytes) -> bool:
        """False means definitely absent. True means maybe present."""
        for bit_index in self._hashes(item):
            if not (self._bits[bit_index // 8] & (1 << (bit_index % 8))):
                return False
        return True

    def to_bytes(self) -> bytes:
        header = struct.pack("<III", self.num_bits, self.num_hashes, len(self._bits))
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        num_bits, num_hashes, byte_len = struct.unpack_from("<III", data, 0)
        offset = struct.calcsize("<III")
        bf = cls(num_bits, num_hashes)
        bf._bits = bytearray(data[offset:offset + byte_len])
        return bf
