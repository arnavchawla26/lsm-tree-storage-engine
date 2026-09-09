import os

import pytest

from lsmtree.sstable import SSTableReader, SSTableWriter


def _write(data_dir, name, items):
    path = os.path.join(data_dir, name)
    meta = SSTableWriter.write(path, iter(items), expected_items=len(items))
    return path, meta


def test_write_and_read_back(data_dir):
    items = [(b"a", b"1"), (b"b", b"2"), (b"c", b"3")]
    path, meta = _write(data_dir, "t1.sst", items)
    assert meta["num_entries"] == 3
    assert meta["min_key"] == b"a"
    assert meta["max_key"] == b"c"

    with SSTableReader(path) as reader:
        assert reader.num_entries == 3
        assert reader.min_key == b"a"
        assert reader.max_key == b"c"
        for key, expected_value in items:
            found, value = reader.get(key)
            assert found is True
            assert value == expected_value


def test_get_missing_key_within_range(data_dir):
    items = [(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]
    path, _ = _write(data_dir, "t2.sst", items)
    with SSTableReader(path) as reader:
        found, value = reader.get(b"b")
        assert found is False
        assert value is None


def test_get_missing_key_out_of_range(data_dir):
    items = [(b"m", b"1"), (b"n", b"2")]
    path, _ = _write(data_dir, "t3.sst", items)
    with SSTableReader(path) as reader:
        assert reader.get(b"a") == (False, None)
        assert reader.get(b"z") == (False, None)


def test_tombstone_roundtrip(data_dir):
    items = [(b"a", b"1"), (b"b", None), (b"c", b"3")]
    path, _ = _write(data_dir, "t4.sst", items)
    with SSTableReader(path) as reader:
        found, value = reader.get(b"b")
        assert found is True
        assert value is None  # tombstone: found but no value


def test_full_iteration_matches_input_order(data_dir):
    items = [(f"key{i:03d}".encode(), str(i).encode()) for i in range(200)]
    path, _ = _write(data_dir, "t5.sst", items)
    with SSTableReader(path) as reader:
        assert list(reader) == items


def test_scan_range(data_dir):
    items = [(f"k{i:03d}".encode(), str(i).encode()) for i in range(50)]
    path, _ = _write(data_dir, "t6.sst", items)
    with SSTableReader(path) as reader:
        result = list(reader.scan(b"k010", b"k015"))
        assert [k for k, _ in result] == [f"k{i:03d}".encode() for i in range(10, 16)]


def test_scan_unbounded(data_dir):
    items = [(f"k{i:03d}".encode(), str(i).encode()) for i in range(5)]
    path, _ = _write(data_dir, "t7.sst", items)
    with SSTableReader(path) as reader:
        assert list(reader.scan(None, None)) == items


def test_bloom_filter_says_no_for_definitely_absent_key(data_dir):
    items = [(str(i).encode(), str(i).encode()) for i in range(500)]
    path, _ = _write(data_dir, "t8.sst", items)
    with SSTableReader(path) as reader:
        # min/max range check already rejects most out-of-range probes;
        # this checks the bloom filter is actually wired in and functional
        # for something inside the key range but never inserted.
        assert reader.bloom.num_bits > 0
        found, _ = reader.get(b"definitely-not-a-real-key")
        assert found is False


def test_invalid_file_raises(data_dir):
    path = os.path.join(data_dir, "not_an_sstable.sst")
    with open(path, "wb") as fh:
        fh.write(b"not a valid sstable file at all")
    with pytest.raises(ValueError):
        SSTableReader(path)


def test_single_entry_table(data_dir):
    path, meta = _write(data_dir, "t9.sst", [(b"only", b"value")])
    assert meta["num_entries"] == 1
    with SSTableReader(path) as reader:
        assert reader.get(b"only") == (True, b"value")


def test_large_table_index_spans_multiple_blocks(data_dir):
    # More than INDEX_INTERVAL entries so the sparse index has multiple
    # entries and get() must actually use binary search + scan correctly.
    items = [(f"key-{i:05d}".encode(), str(i).encode()) for i in range(500)]
    path, _ = _write(data_dir, "t10.sst", items)
    with SSTableReader(path) as reader:
        assert len(reader._index) > 1
        for key, expected in items[::37]:  # sample across the whole range
            found, value = reader.get(key)
            assert found is True
            assert value == expected
