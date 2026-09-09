import random

import pytest

from lsmtree.skiplist import NOT_FOUND, SkipList


def test_empty_search_returns_not_found():
    sl = SkipList()
    assert sl.search(b"missing") is NOT_FOUND
    assert b"missing" not in sl
    assert len(sl) == 0


def test_insert_and_search():
    sl = SkipList(seed=1)
    sl.insert(b"b", 2)
    sl.insert(b"a", 1)
    sl.insert(b"c", 3)
    assert sl.search(b"a") == 1
    assert sl.search(b"b") == 2
    assert sl.search(b"c") == 3
    assert len(sl) == 3


def test_insert_overwrites_existing_key():
    sl = SkipList(seed=2)
    sl.insert(b"x", 1)
    sl.insert(b"x", 2)
    assert sl.search(b"x") == 2
    assert len(sl) == 1  # overwrite, not a new node


def test_iteration_is_sorted():
    sl = SkipList(seed=3)
    keys = [b"delta", b"alpha", b"charlie", b"bravo", b"echo"]
    for i, k in enumerate(keys):
        sl.insert(k, i)
    assert [k for k, _ in sl] == sorted(keys)


def test_delete_removes_key():
    sl = SkipList(seed=4)
    sl.insert(b"a", 1)
    sl.insert(b"b", 2)
    assert sl.delete(b"a") is True
    assert sl.search(b"a") is NOT_FOUND
    assert len(sl) == 1
    assert sl.delete(b"a") is False  # already gone


def test_delete_missing_key_returns_false():
    sl = SkipList(seed=5)
    assert sl.delete(b"nope") is False


def test_range_full():
    sl = SkipList(seed=6)
    for k in [b"a", b"b", b"c", b"d"]:
        sl.insert(k, k)
    assert [k for k, _ in sl.range(None, None)] == [b"a", b"b", b"c", b"d"]


def test_range_bounded():
    sl = SkipList(seed=7)
    for k in [b"a", b"b", b"c", b"d", b"e"]:
        sl.insert(k, k)
    assert [k for k, _ in sl.range(b"b", b"d")] == [b"b", b"c", b"d"]


def test_range_start_only():
    sl = SkipList(seed=8)
    for k in [b"a", b"b", b"c"]:
        sl.insert(k, k)
    assert [k for k, _ in sl.range(b"b", None)] == [b"b", b"c"]


def test_range_end_only():
    sl = SkipList(seed=9)
    for k in [b"a", b"b", b"c"]:
        sl.insert(k, k)
    assert [k for k, _ in sl.range(None, b"b")] == [b"a", b"b"]


def test_range_no_match():
    sl = SkipList(seed=10)
    sl.insert(b"a", 1)
    sl.insert(b"z", 26)
    assert list(sl.range(b"m", b"n")) == []


def test_large_random_insert_stays_sorted_and_consistent():
    rng = random.Random(42)
    sl = SkipList(seed=42)
    reference = {}
    for _ in range(2000):
        key = str(rng.randint(0, 500)).encode()
        value = rng.randint(0, 1_000_000)
        sl.insert(key, value)
        reference[key] = value

    assert len(sl) == len(reference)
    keys_in_order = [k for k, _ in sl]
    assert keys_in_order == sorted(reference.keys())
    for key, expected in reference.items():
        assert sl.search(key) == expected


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        SkipList(max_level=0)
    with pytest.raises(ValueError):
        SkipList(p=0)
    with pytest.raises(ValueError):
        SkipList(p=1)
