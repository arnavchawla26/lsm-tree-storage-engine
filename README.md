# lsm-tree-storage-engine

A from-scratch LSM-tree (Log-Structured Merge-tree) key-value storage engine
in Python -- the write-optimized counterpart to a B-Tree: writes are cheap
(append-only, sequential I/O) and reads are made fast again with bloom
filters and a sparse index, instead of an in-place page-modifying B+Tree.

Every layer is hand-implemented, no external storage/database dependencies:
a skip list for the sorted in-memory memtable, a binary SSTable file format
with its own bloom filter and sparse index, a checksummed write-ahead log,
and a k-way-merge compactor.

## What it does

```
put/delete -> WAL.append (fsync'd) -> MemTable (skip list) -> flush when full
get        -> MemTable -> newest-to-oldest SSTables (bloom filter + sparse index skip most reads)
scan       -> merged, newest-wins, tombstone-filtered walk across MemTable + all SSTables
```

- **MemTable** -- a from-scratch skip list (`lsmtree/skiplist.py`, Pugh 1990)
  keeps writes sorted in memory with expected O(log n) insert/search, so
  range scans over the active memtable don't need a separate sort step.
- **WAL** -- every `put`/`delete` is appended and `fsync`'d to a
  write-ahead log *before* it touches the memtable. Each record carries its
  own CRC32; replay stops cleanly at the first truncated or corrupt record
  instead of raising, which is exactly what a process killed mid-write
  leaves behind.
- **SSTable** -- an immutable, sorted, binary on-disk file: a data block,
  a sparse index (every 16th key, for O(log n) seeks), a per-table bloom
  filter (double-hashed from a single blake2b digest) to skip disk reads
  for keys that are provably absent, and a fixed-size footer at a known
  offset from EOF so opening a reader never needs to scan the file.
- **Compaction** -- size-tiered: once more than `level0_compaction_threshold`
  SSTables exist, all of them are merged in one k-way heap merge
  (newest-source-wins on key conflicts, tombstones dropped since every
  older SSTable that a tombstone could shadow is, by construction, part of
  the same merge).
- **CLI** -- `lsmkv`, a small argparse-based front end: `put` / `get` /
  `delete` / `scan` / `stats` / `compact`.

## Tech stack

Python 3.9+, standard library only (`heapq`, `hashlib`, `struct`, `zlib`,
`argparse`). Tests use `pytest`.

## How to run

```bash
pip install -e ".[dev]"

lsmkv --db ./data put foo bar
lsmkv --db ./data get foo
lsmkv --db ./data scan --start a --end z
lsmkv --db ./data stats
lsmkv --db ./data compact

pytest                 # 80 tests
```

```python
from lsmtree import LSMTree

with LSMTree("./data") as db:
    db.put(b"key", b"value")
    db.get(b"key")           # b"value"
    db.delete(b"key")
    db.get(b"key")           # lsmtree.NOT_FOUND
    list(db.scan(b"a", b"z"))
```

## Crash recovery, actually tested

`tests/test_crash_recovery.py` launches a real subprocess that writes
thousands of keys, sends it an unblockable `SIGKILL` at an essentially
arbitrary moment (parametrized over several delays, run repeatedly), and
then opens a **fresh** `LSMTree` on the same directory in the parent
process. The test verifies every write the subprocess durably acknowledged
(i.e. `put()` returned, meaning its WAL record was `fsync`'d) survives the
kill and comes back with the correct value -- and that nothing beyond the
one write that could plausibly have been in flight silently appears. Closing
file handles cleanly never exercises this path; only an actual, unclean
process death does.

## Current status

Functional v1, shipped complete in one run: skip list, memtable, WAL with
CRC-checked replay, SSTable read/write with bloom filter + sparse index,
size-tiered compaction, `lsmkv` CLI, and 80 tests (including the SIGKILL
crash-recovery test above), all passing from a clean `pip install -e ".[dev]"`.

### Roadmap (deliberately out of scope for v1)

- **Leveled compaction.** This engine does single-tier, compact-everything
  compaction, which is simple and always correct but rewrites more data
  than necessary as the dataset grows. Real leveled compaction (SSTables
  organized into levels with non-overlapping key ranges within a level,
  compacted incrementally between adjacent levels) reduces write
  amplification substantially and would be the natural next step.
- **Manifest file / atomic multi-file commits.** SSTable discovery currently
  relies on filesystem globbing plus a sequence number encoded in the
  filename, which is simple and crash-safe (an `os.replace` either lands a
  complete file or doesn't) but doesn't support things like atomically
  swapping several files at once during compaction rollout.
- **Concurrent writers.** The engine is single-process, single-threaded by
  design; no locking is implemented.

## Part of a series

The complementary storage-engine design to
[b-tree-kv-store](https://github.com/arnavchawla26/b-tree-kv-store)'s
from-scratch on-disk B+Tree: same crash-recovery testing discipline
(a real SIGKILL, not a simulated one), opposite write/read tradeoff.
