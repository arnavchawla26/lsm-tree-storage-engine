"""lsmkv: a small command-line front end for LSMTree.

Examples:
    lsmkv --db ./data put foo bar
    lsmkv --db ./data get foo
    lsmkv --db ./data delete foo
    lsmkv --db ./data scan --start a --end z
    lsmkv --db ./data stats
    lsmkv --db ./data compact
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .engine import LSMTree
from .skiplist import NOT_FOUND


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lsmkv", description="A from-scratch LSM-tree key-value store.")
    parser.add_argument("--db", default="./lsmkv_data", help="Data directory (default: ./lsmkv_data)")
    parser.add_argument(
        "--memtable-threshold",
        type=int,
        default=None,
        help="Flush the memtable once it holds roughly this many bytes (default: 1 MiB)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_put = sub.add_parser("put", help="Set a key's value")
    p_put.add_argument("key")
    p_put.add_argument("value")

    p_get = sub.add_parser("get", help="Look up a key")
    p_get.add_argument("key")

    p_del = sub.add_parser("delete", help="Delete a key")
    p_del.add_argument("key")

    p_scan = sub.add_parser("scan", help="Range-scan keys in sorted order")
    p_scan.add_argument("--start", default=None, help="Start key (inclusive); omit for the beginning")
    p_scan.add_argument("--end", default=None, help="End key (inclusive); omit for the end")

    sub.add_parser("stats", help="Show memtable/SSTable statistics")
    sub.add_parser("compact", help="Force a compaction of all current SSTables")

    return parser


def _open_engine(args) -> LSMTree:
    kwargs = {}
    if args.memtable_threshold is not None:
        kwargs["memtable_threshold_bytes"] = args.memtable_threshold
    return LSMTree(args.db, **kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    engine = _open_engine(args)
    try:
        if args.command == "put":
            engine.put(args.key.encode(), args.value.encode())
            print(f"OK put {args.key!r}")
        elif args.command == "get":
            value = engine.get(args.key.encode())
            if value is NOT_FOUND:
                print(f"(not found) {args.key!r}")
                return 1
            print(value.decode(errors="replace"))
        elif args.command == "delete":
            engine.delete(args.key.encode())
            print(f"OK delete {args.key!r}")
        elif args.command == "scan":
            start = args.start.encode() if args.start is not None else None
            end = args.end.encode() if args.end is not None else None
            count = 0
            for key, value in engine.scan(start, end):
                print(f"{key.decode(errors='replace')}\t{value.decode(errors='replace')}")
                count += 1
            print(f"({count} entries)", file=sys.stderr)
        elif args.command == "stats":
            stats = engine.stats()
            for k, v in stats.items():
                print(f"{k}: {v}")
        elif args.command == "compact":
            meta = engine.compact()
            if meta is None:
                print("Nothing to compact (fewer than 2 SSTables)")
            else:
                print(f"Compacted into 1 SSTable with {meta['num_entries']} entries "
                      f"({meta['size_bytes']} bytes)")
        else:  # pragma: no cover - argparse enforces valid subcommands
            parser.error(f"unknown command {args.command!r}")
            return 2
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
