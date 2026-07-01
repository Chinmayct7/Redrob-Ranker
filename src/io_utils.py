"""Streaming I/O for the candidate pool and submission CSV."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterator


def open_candidates_path(path: str | Path):
    """Return a text-mode file handle for a .jsonl or .jsonl.gz file."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield one candidate dict per non-empty line. Skips and counts
    malformed lines rather than crashing the whole run on one bad row."""
    bad = 0
    with open_candidates_path(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    print(f"[warn] skipping malformed JSON at line {line_no}")
    if bad:
        print(f"[warn] {bad} malformed lines skipped total")


def write_submission_csv(path: str | Path, rows: list[dict]) -> None:
    """rows: list of {candidate_id, rank, score, reasoning}, already sorted
    and ranked 1..N. Writes exactly the header validate_submission.py
    expects."""
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.8f}", r["reasoning"]])
