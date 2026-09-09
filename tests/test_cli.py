import subprocess
import sys


def _run(*args, db):
    return subprocess.run(
        [sys.executable, "-m", "lsmtree.cli", "--db", db, *args],
        capture_output=True, text=True, timeout=30,
    )


def test_put_and_get(tmp_path):
    db = str(tmp_path / "data")
    result = _run("put", "foo", "bar", db=db)
    assert result.returncode == 0
    assert "OK put" in result.stdout

    result = _run("get", "foo", db=db)
    assert result.returncode == 0
    assert result.stdout.strip() == "bar"


def test_get_missing_key_exits_nonzero(tmp_path):
    db = str(tmp_path / "data")
    result = _run("get", "missing", db=db)
    assert result.returncode == 1
    assert "not found" in result.stdout


def test_delete(tmp_path):
    db = str(tmp_path / "data")
    _run("put", "k", "v", db=db)
    result = _run("delete", "k", db=db)
    assert result.returncode == 0
    result = _run("get", "k", db=db)
    assert result.returncode == 1


def test_scan(tmp_path):
    db = str(tmp_path / "data")
    for k, v in [("a", "1"), ("b", "2"), ("c", "3")]:
        _run("put", k, v, db=db)
    result = _run("scan", db=db)
    assert result.returncode == 0
    lines = [l for l in result.stdout.splitlines() if l]
    assert lines == ["a\t1", "b\t2", "c\t3"]


def test_scan_with_bounds(tmp_path):
    db = str(tmp_path / "data")
    for k in "abcde":
        _run("put", k, k.upper(), db=db)
    result = _run("scan", "--start", "b", "--end", "d", db=db)
    lines = [l for l in result.stdout.splitlines() if l]
    assert lines == ["b\tB", "c\tC", "d\tD"]


def test_stats(tmp_path):
    db = str(tmp_path / "data")
    _run("put", "a", "1", db=db)
    result = _run("stats", db=db)
    assert result.returncode == 0
    assert "memtable_entries: 1" in result.stdout


def test_compact_reports_noop_with_few_tables(tmp_path):
    db = str(tmp_path / "data")
    _run("put", "a", "1", db=db)
    result = _run("compact", db=db)
    assert result.returncode == 0
    assert "Nothing to compact" in result.stdout


def test_compact_after_multiple_flushes(tmp_path):
    db = str(tmp_path / "data")
    for i in range(5):
        _run("--memtable-threshold", "1", "put", f"k{i}", str(i), db=db)
    result = _run("compact", db=db)
    assert result.returncode == 0
    assert "Compacted into 1 SSTable" in result.stdout


def test_persistence_across_invocations(tmp_path):
    db = str(tmp_path / "data")
    _run("put", "x", "1", db=db)
    _run("put", "y", "2", db=db)
    result = _run("get", "x", db=db)
    assert result.stdout.strip() == "1"
    result = _run("get", "y", db=db)
    assert result.stdout.strip() == "2"
