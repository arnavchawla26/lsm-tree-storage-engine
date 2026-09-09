import random

from lsmtree.bloom_filter import BloomFilter


def test_no_false_negatives():
    bf = BloomFilter.for_capacity(1000, false_positive_rate=0.01)
    items = [f"key-{i}".encode() for i in range(1000)]
    for item in items:
        bf.add(item)
    for item in items:
        assert bf.might_contain(item) is True


def test_false_positive_rate_roughly_bounded():
    n = 2000
    target_fpr = 0.02
    bf = BloomFilter.for_capacity(n, false_positive_rate=target_fpr)
    present = [f"present-{i}".encode() for i in range(n)]
    for item in present:
        bf.add(item)

    rng = random.Random(123)
    absent = [f"absent-{rng.randint(0, 10**9)}".encode() for _ in range(5000)]
    false_positives = sum(1 for item in absent if bf.might_contain(item))
    observed_fpr = false_positives / len(absent)
    # Generous margin -- this is a statistical property, not exact.
    assert observed_fpr < target_fpr * 4


def test_empty_filter_rejects_everything_probabilistically():
    bf = BloomFilter.for_capacity(100)
    assert bf.might_contain(b"anything") is False


def test_serialization_roundtrip_preserves_behavior():
    bf = BloomFilter.for_capacity(50)
    items = [str(i).encode() for i in range(50)]
    for item in items:
        bf.add(item)

    data = bf.to_bytes()
    restored = BloomFilter.from_bytes(data)

    assert restored.num_bits == bf.num_bits
    assert restored.num_hashes == bf.num_hashes
    for item in items:
        assert restored.might_contain(item) is True


def test_for_capacity_produces_more_bits_for_more_items():
    small = BloomFilter.for_capacity(10)
    large = BloomFilter.for_capacity(10000)
    assert large.num_bits > small.num_bits


def test_minimum_size_guard():
    bf = BloomFilter(num_bits=0, num_hashes=0)
    assert bf.num_bits >= 8
    assert bf.num_hashes >= 1
