import os

from lsmtree.wal import WAL, OP_DELETE, OP_PUT


def test_append_and_replay_roundtrip(data_dir):
    path = os.path.join(data_dir, "wal.log")
    wal = WAL(path)
    wal.append(OP_PUT, b"a", b"1")
    wal.append(OP_PUT, b"b", b"2")
    wal.append(OP_DELETE, b"a")
    wal.close()

    records = list(WAL.replay(path))
    assert records == [
        (OP_PUT, b"a", b"1"),
        (OP_PUT, b"b", b"2"),
        (OP_DELETE, b"a", b""),
    ]


def test_replay_nonexistent_file_yields_nothing(data_dir):
    path = os.path.join(data_dir, "does_not_exist.log")
    assert list(WAL.replay(path)) == []


def test_clear_truncates_log(data_dir):
    path = os.path.join(data_dir, "wal.log")
    wal = WAL(path)
    wal.append(OP_PUT, b"a", b"1")
    wal.clear()
    assert list(WAL.replay(path)) == []
    wal.append(OP_PUT, b"b", b"2")
    wal.close()
    assert list(WAL.replay(path)) == [(OP_PUT, b"b", b"2")]


def test_replay_stops_cleanly_at_truncated_last_record(data_dir):
    """Simulates exactly what a mid-write crash leaves behind: a WAL file
    whose last record is cut off partway through. Replay must recover every
    complete record before the cut and simply stop, not raise.
    """
    path = os.path.join(data_dir, "wal.log")
    wal = WAL(path)
    wal.append(OP_PUT, b"good1", b"1")
    wal.append(OP_PUT, b"good2", b"2")
    wal.close()

    # Append a record, then chop off its last few bytes to simulate a
    # process death mid-write of the third record.
    wal = WAL(path)
    wal.append(OP_PUT, b"partial", b"this-should-be-lost")
    wal.close()
    full_size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.truncate(full_size - 5)

    records = list(WAL.replay(path))
    assert records == [(OP_PUT, b"good1", b"1"), (OP_PUT, b"good2", b"2")]


def test_replay_detects_corrupted_record_via_crc(data_dir):
    path = os.path.join(data_dir, "wal.log")
    wal = WAL(path)
    wal.append(OP_PUT, b"good", b"1")
    wal.append(OP_PUT, b"corrupt-me", b"2")
    wal.close()

    with open(path, "r+b") as fh:
        data = bytearray(fh.read())
    # Flip a byte inside the second record's key, which will fail its CRC check.
    flip_index = data.index(b"corrupt-me") + 2
    data[flip_index] ^= 0xFF
    with open(path, "wb") as fh:
        fh.write(data)

    records = list(WAL.replay(path))
    assert records == [(OP_PUT, b"good", b"1")]
