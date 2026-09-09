import glob
import os

from lsmtree.engine import LSMTree
from lsmtree.skiplist import NOT_FOUND


def test_put_get_delete(data_dir):
    with LSMTree(data_dir) as db:
        db.put(b"a", b"1")
        assert db.get(b"a") == b"1"
        db.delete(b"a")
        assert db.get(b"a") is NOT_FOUND


def test_get_missing_key(data_dir):
    with LSMTree(data_dir) as db:
        assert db.get(b"nope") is NOT_FOUND


def test_overwrite(data_dir):
    with LSMTree(data_dir) as db:
        db.put(b"k", b"1")
        db.put(b"k", b"2")
        assert db.get(b"k") == b"2"


def test_flush_creates_sstable_and_clears_memtable(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1 << 30) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
        assert not db.memtable.is_empty()
        db.flush()
        assert db.memtable.is_empty()
        assert len(glob.glob(os.path.join(data_dir, "*.sst"))) == 1
        # data still reachable after flush, now served from the SSTable
        assert db.get(b"a") == b"1"
        assert db.get(b"b") == b"2"


def test_automatic_flush_when_threshold_exceeded(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=50) as db:
        for i in range(20):
            db.put(f"key{i:03d}".encode(), b"x" * 10)
        assert len(glob.glob(os.path.join(data_dir, "*.sst"))) >= 1
        for i in range(20):
            assert db.get(f"key{i:03d}".encode()) == b"x" * 10


def test_delete_after_flush_shadows_sstable_value(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1) as db:
        db.put(b"k", b"v")  # flushed immediately (tiny threshold)
        assert db.get(b"k") == b"v"
        db.delete(b"k")  # tombstone lands in the fresh memtable
        assert db.get(b"k") is NOT_FOUND


def test_reopen_recovers_flushed_data(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1 << 30) as db:
        db.put(b"a", b"1")
        db.flush()

    with LSMTree(data_dir) as db2:
        assert db2.get(b"a") == b"1"


def test_reopen_without_flush_recovers_via_wal_replay(data_dir):
    db = LSMTree(data_dir, memtable_threshold_bytes=1 << 30)
    db.put(b"a", b"1")
    db.put(b"b", b"2")
    db.delete(b"a")
    db.wal.close()  # simulate an abrupt stop: no clean flush, WAL just stops accepting writes
    db._closed = True

    db2 = LSMTree(data_dir)
    assert db2.get(b"a") is NOT_FOUND  # tombstone replayed correctly
    assert db2.get(b"b") == b"2"
    db2.close()


def test_scan_merges_memtable_and_sstables_newest_wins(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1 << 30) as db:
        db.put(b"a", b"old-a")
        db.put(b"b", b"1")
        db.flush()
        db.put(b"a", b"new-a")  # overwrite, still in memtable
        db.put(b"c", b"3")

        result = dict(db.scan())
        assert result == {b"a": b"new-a", b"b": b"1", b"c": b"3"}


def test_scan_range_bounds(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1 << 30) as db:
        for k in [b"a", b"b", b"c", b"d", b"e"]:
            db.put(k, k.upper())
        db.flush()
        db.put(b"f", b"F")  # stays in memtable

        result = [k for k, _ in db.scan(b"c", b"f")]
        assert result == [b"c", b"d", b"e", b"f"]


def test_scan_excludes_deleted_keys(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1 << 30) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
        db.flush()
        db.delete(b"a")

        result = dict(db.scan())
        assert result == {b"b": b"2"}


def test_compaction_reduces_sstable_count_and_preserves_data(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1, level0_compaction_threshold=1000) as db:
        for i in range(5):
            db.put(f"round{i}".encode(), str(i).encode())  # each put flushes (threshold=1)

        assert len(glob.glob(os.path.join(data_dir, "*.sst"))) == 5
        db.compact()
        assert len(glob.glob(os.path.join(data_dir, "*.sst"))) == 1

        for i in range(5):
            assert db.get(f"round{i}".encode()) == str(i).encode()


def test_automatic_compaction_triggers_at_threshold(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1, level0_compaction_threshold=3) as db:
        for i in range(10):
            db.put(f"k{i}".encode(), str(i).encode())
        # compaction should have kicked in at least once, keeping file count low
        assert len(glob.glob(os.path.join(data_dir, "*.sst"))) < 10
        for i in range(10):
            assert db.get(f"k{i}".encode()) == str(i).encode()


def test_compact_with_fewer_than_two_tables_is_a_noop(data_dir):
    with LSMTree(data_dir) as db:
        assert db.compact() is None
        db.put(b"a", b"1")
        db.flush()
        assert db.compact() is None  # only 1 sstable


def test_stats(data_dir):
    with LSMTree(data_dir, memtable_threshold_bytes=1 << 30) as db:
        db.put(b"a", b"1")
        stats = db.stats()
        assert stats["memtable_entries"] == 1
        assert stats["num_sstables"] == 0
        db.flush()
        stats = db.stats()
        assert stats["memtable_entries"] == 0
        assert stats["num_sstables"] == 1


def test_operations_after_close_raise(data_dir):
    db = LSMTree(data_dir)
    db.close()
    try:
        db.put(b"a", b"1")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
