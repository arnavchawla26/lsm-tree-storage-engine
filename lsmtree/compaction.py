"""Size-tiered compaction: merge every current SSTable into one new SSTable.

This engine keeps a single tier (level 0). When the number of SSTable files
exceeds `level0_compaction_threshold`, ALL of them are merged in one pass:

  - a k-way merge (via heapq) walks every source SSTable's sorted iterator
    in lockstep by key
  - when several sources have the same key, the one from the most recently
    created SSTable (highest sequence number) wins
  - because this single compaction pass always consumes every SSTable that
    exists, it is safe to drop tombstones entirely in the output: a
    tombstone's only job is to shadow an older value for the same key in an
    older SSTable, and every older SSTable is, by construction, part of
    this same merge

This is deliberately simpler than real-world leveled compaction (which
compacts between adjacent levels incrementally instead of all-at-once) --
see the README's Roadmap section.
"""

from __future__ import annotations

import heapq
from typing import List, Tuple

from .sstable import SSTableReader, SSTableWriter


def compact_all(sstable_paths_newest_first: List[str], output_path: str) -> dict:
    """Merge all given SSTables (ordered newest-first) into one new SSTable.

    Returns the same metadata dict SSTableWriter.write returns. Does not
    delete the input files -- the caller (LSMTree) does that only after this
    call has returned successfully, so a crash mid-compaction never loses data.
    """
    readers = [SSTableReader(p) for p in sstable_paths_newest_first]
    try:
        merged = _merge_newest_wins(readers)
        total_entries = sum(r.num_entries for r in readers)
        meta = SSTableWriter.write(output_path, merged, expected_items=total_entries)
    finally:
        for r in readers:
            r.close()
    return meta


def _merge_newest_wins(readers: List[SSTableReader]):
    """Yield (key, value) pairs (tombstones dropped) merged across readers.

    readers[0] is the newest source, readers[-1] the oldest -- ties broken
    by preferring the lowest reader index (i.e. the newest source).
    """
    heap: List[Tuple[bytes, int, object]] = []
    iterators = [iter(r) for r in readers]

    def push(idx: int) -> None:
        try:
            key, value = next(iterators[idx])
        except StopIteration:
            return
        heapq.heappush(heap, (key, idx, value))

    for idx in range(len(iterators)):
        push(idx)

    while heap:
        key, idx, value = heapq.heappop(heap)
        push(idx)
        # heapq breaks ties on idx (each reader contributes at most one
        # heap entry at a time, so idx is always unique among current
        # entries), so the tuple just popped is the newest source's entry
        # for `key`. Drain and discard every other entry for the same key
        # from older sources before moving on to the next distinct key.
        while heap and heap[0][0] == key:
            _, dup_idx, _ = heapq.heappop(heap)
            push(dup_idx)
        if value is not None:  # drop tombstones -- see module docstring
            yield key, value
