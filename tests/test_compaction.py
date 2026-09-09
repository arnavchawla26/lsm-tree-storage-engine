import os

from lsmtree.compaction import compact_all
from lsmtree.sstable import SSTableReader, SSTableWriter


def _make_sstable(data_dir, name, items):
    path = os.path.join(data_dir, name)
    SSTableWriter.write(path, iter(items), expected_items=len(items))
    return path


def test_compact_merges_disjoint_tables(data_dir):
    t1 = _make_sstable(data_dir, "1.sst", [(b"a", b"1"), (b"c", b"3")])
    t2 = _make_sstable(data_dir, "2.sst", [(b"b", b"2"), (b"d", b"4")])
    output = os.path.join(data_dir, "out.sst")
    meta = compact_all([t2, t1], output)  # newest first = t2

    assert meta["num_entries"] == 4
    with SSTableReader(output) as reader:
        assert list(reader) == [(b"a", b"1"), (b"b", b"2"), (b"c", b"3"), (b"d", b"4")]


def test_compact_newest_source_wins_on_conflict(data_dir):
    old = _make_sstable(data_dir, "old.sst", [(b"x", b"old-value")])
    new = _make_sstable(data_dir, "new.sst", [(b"x", b"new-value")])
    output = os.path.join(data_dir, "out.sst")
    # newest_first order: `new` must come before `old`
    compact_all([new, old], output)

    with SSTableReader(output) as reader:
        found, value = reader.get(b"x")
        assert found is True
        assert value == b"new-value"


def test_compact_drops_tombstones(data_dir):
    old = _make_sstable(data_dir, "old.sst", [(b"gone", b"was-here"), (b"stays", b"1")])
    new = _make_sstable(data_dir, "new.sst", [(b"gone", None)])  # tombstone shadows old value
    output = os.path.join(data_dir, "out.sst")
    compact_all([new, old], output)

    with SSTableReader(output) as reader:
        assert reader.get(b"gone") == (False, None)  # gone entirely, not even a tombstone
        assert reader.get(b"stays") == (True, b"1")
        assert reader.num_entries == 1


def test_compact_three_way_merge_with_mixed_recency(data_dir):
    t1 = _make_sstable(data_dir, "1.sst", [(b"k", b"v1")])   # oldest
    t2 = _make_sstable(data_dir, "2.sst", [(b"k", b"v2")])   # middle
    t3 = _make_sstable(data_dir, "3.sst", [(b"k", b"v3")])   # newest
    output = os.path.join(data_dir, "out.sst")
    compact_all([t3, t2, t1], output)

    with SSTableReader(output) as reader:
        assert reader.get(b"k") == (True, b"v3")
        assert reader.num_entries == 1


def test_compact_single_table_is_a_clean_copy(data_dir):
    t1 = _make_sstable(data_dir, "1.sst", [(b"a", b"1"), (b"b", b"2")])
    output = os.path.join(data_dir, "out.sst")
    compact_all([t1], output)
    with SSTableReader(output) as reader:
        assert list(reader) == [(b"a", b"1"), (b"b", b"2")]


def test_compact_does_not_delete_input_files(data_dir):
    t1 = _make_sstable(data_dir, "1.sst", [(b"a", b"1")])
    t2 = _make_sstable(data_dir, "2.sst", [(b"b", b"2")])
    output = os.path.join(data_dir, "out.sst")
    compact_all([t2, t1], output)
    # compact_all is a pure merge; deleting inputs is the caller's job
    # (LSMTree.compact), so both source files must still exist here.
    assert os.path.exists(t1)
    assert os.path.exists(t2)
