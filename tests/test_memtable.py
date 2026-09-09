from lsmtree.memtable import MemTable, TOMBSTONE
from lsmtree.skiplist import NOT_FOUND


def test_put_and_get():
    mt = MemTable()
    mt.put(b"a", b"1")
    assert mt.get(b"a") == b"1"
    assert mt.get(b"missing") is NOT_FOUND


def test_put_overwrite():
    mt = MemTable()
    mt.put(b"a", b"1")
    mt.put(b"a", b"2")
    assert mt.get(b"a") == b"2"
    assert len(mt) == 1


def test_delete_sets_tombstone_not_removal():
    mt = MemTable()
    mt.put(b"a", b"1")
    mt.delete(b"a")
    assert mt.get(b"a") is TOMBSTONE
    assert len(mt) == 1  # tombstone still occupies a slot, to be flushed


def test_delete_missing_key_still_records_tombstone():
    mt = MemTable()
    mt.delete(b"never-existed")
    assert mt.get(b"never-existed") is TOMBSTONE


def test_is_empty():
    mt = MemTable()
    assert mt.is_empty() is True
    mt.put(b"a", b"1")
    assert mt.is_empty() is False


def test_items_sorted():
    mt = MemTable()
    mt.put(b"c", b"3")
    mt.put(b"a", b"1")
    mt.put(b"b", b"2")
    assert [k for k, _ in mt.items()] == [b"a", b"b", b"c"]


def test_approx_size_bytes_tracks_puts_and_overwrites():
    mt = MemTable()
    assert mt.approx_size_bytes == 0
    mt.put(b"ab", b"cde")  # 2 + 3 = 5
    assert mt.approx_size_bytes == 5
    mt.put(b"ab", b"z")  # overwrite: 2 + 1 = 3
    assert mt.approx_size_bytes == 3


def test_approx_size_bytes_tracks_delete_of_existing_key():
    mt = MemTable()
    mt.put(b"key", b"value")  # 3 + 5 = 8
    size_before = mt.approx_size_bytes
    mt.delete(b"key")
    # tombstone: value bytes no longer counted, key length remains
    assert mt.approx_size_bytes == len(b"key")
    assert mt.approx_size_bytes < size_before


def test_range_over_memtable():
    mt = MemTable()
    for k in [b"a", b"b", b"c", b"d"]:
        mt.put(k, k.upper())
    assert [k for k, _ in mt.range(b"b", b"c")] == [b"b", b"c"]


def test_clear_resets_state():
    mt = MemTable()
    mt.put(b"a", b"1")
    mt.clear()
    assert mt.is_empty()
    assert mt.approx_size_bytes == 0
    assert mt.get(b"a") is NOT_FOUND
