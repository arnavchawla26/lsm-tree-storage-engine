"""A from-scratch skip list: the sorted in-memory structure backing MemTable.

A skip list gives expected O(log n) insert/search/delete while staying
sorted at all times (unlike a hash map + sort-on-flush), which is what lets
the memtable answer range scans directly without a full sort first.

This is a textbook probabilistic skip list (Pugh, 1990): each node carries a
random "tower" of forward pointers, and search/insert walk from the top
level down, moving right while the next key is still less than the target.
"""

from __future__ import annotations

import random
from typing import Any, Iterator, Optional, Tuple

_MAX_LEVEL = 16
_P = 0.5

NOT_FOUND = object()  # marks "no such key" distinctly from a stored None


class _Node:
    __slots__ = ("key", "value", "forward")

    def __init__(self, key: Optional[bytes], value: Any, level: int):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)


class SkipList:
    """Sorted map of bytes -> Any, ordered by Python's natural bytes order."""

    def __init__(self, max_level: int = _MAX_LEVEL, p: float = _P, seed: Optional[int] = None):
        if max_level < 1:
            raise ValueError("max_level must be >= 1")
        if not (0 < p < 1):
            raise ValueError("p must be in (0, 1)")
        self._max_level = max_level
        self._p = p
        self._level = 0  # highest level currently in use (0-indexed)
        self._head = _Node(None, None, max_level)
        self._size = 0
        self._rand = random.Random(seed)

    def __len__(self) -> int:
        return self._size

    def _random_level(self) -> int:
        level = 0
        while self._rand.random() < self._p and level < self._max_level - 1:
            level += 1
        return level

    def _find_predecessors(self, key: bytes) -> list:
        update = [self._head] * (self._max_level)
        node = self._head
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node
        return update

    def insert(self, key: bytes, value: Any) -> None:
        """Insert key with value, overwriting if key already present."""
        update = self._find_predecessors(key)
        candidate = update[0].forward[0]
        if candidate is not None and candidate.key == key:
            candidate.value = value
            return

        new_level = self._random_level()
        if new_level > self._level:
            for i in range(self._level + 1, new_level + 1):
                update[i] = self._head
            self._level = new_level

        node = _Node(key, value, new_level)
        for i in range(new_level + 1):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
        self._size += 1

    def search(self, key: bytes) -> Any:
        """Return the stored value for key, or NOT_FOUND if absent."""
        node = self._head
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is not None and node.key == key:
            return node.value
        return NOT_FOUND

    def __contains__(self, key: bytes) -> bool:
        return self.search(key) is not NOT_FOUND

    def delete(self, key: bytes) -> bool:
        """Remove key entirely from the structure. Returns True if it existed."""
        update = self._find_predecessors(key)
        candidate = update[0].forward[0]
        if candidate is None or candidate.key != key:
            return False
        for i in range(self._level + 1):
            if update[i].forward[i] is not candidate:
                break
            update[i].forward[i] = candidate.forward[i]
        while self._level > 0 and self._head.forward[self._level] is None:
            self._level -= 1
        self._size -= 1
        return True

    def __iter__(self) -> Iterator[Tuple[bytes, Any]]:
        node = self._head.forward[0]
        while node is not None:
            yield node.key, node.value
            node = node.forward[0]

    def range(self, start: Optional[bytes], end: Optional[bytes]) -> Iterator[Tuple[bytes, Any]]:
        """Yield (key, value) for start <= key <= end, in sorted order.

        start=None means "from the beginning"; end=None means "to the end".
        """
        if start is None:
            node = self._head.forward[0]
        else:
            node = self._head
            for i in range(self._level, -1, -1):
                while node.forward[i] is not None and node.forward[i].key < start:
                    node = node.forward[i]
            node = node.forward[0]
        while node is not None and (end is None or node.key <= end):
            yield node.key, node.value
            node = node.forward[0]
