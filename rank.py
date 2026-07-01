#!/usr/bin/env python3
"""
Redrob Hackathon ranker -- single-command entry point.

Usage:
    python rank.py --candidates data/candidates.jsonl.gz --jd data/job_description.md --out submission.csv

No network access, no GPU, pure-Python + stdlib (see requirements.txt for
the few optional dev/test deps -- none are needed at rank-time). Designed
to comfortably clear the 5-minute / 16GB / CPU-only constraint on the full
100k-candidate pool.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import sys
import time

from src import config, io_utils, jd_parser, scoring, semantic
from src.honeypot import check_honeypot
from src.reasoning import build_reasoning


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against the JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--jd", default="data/job_description.md", help="Path to job_description.md")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=config.TOP_N)
    args = parser.parse_args()

    t_start = time.time()

    # --- 1. Parse the JD and sanity-check our domain-knowledge tables ---
    jd = jd_parser.load_and_parse(args.jd)
    print(f"[jd] {jd.role_title} | experience {jd.experience_min}-{jd.experience_max} "
          f"(ideal {jd.ideal_experience_min}-{jd.ideal_experience_max}) | "
          f"notice <= {jd.notice_ideal_max_days}d")
    if jd.warnings:
        print("[jd] warnings:")
        for w in jd.warnings:
            print(f"  - {w}")

    # --- 2. Pass 1: stream the pool once to compute semantic IDF weights -
    print(f"[pass 1] computing semantic IDF over {args.candidates} ...")
    t0 = time.time()

    def text_iter():
        for c in io_utils.iter_candidates(args.candidates):
            yield semantic.candidate_text(c)

    idf, n_seen = semantic.compute_idf(text_iter())
    print(f"[pass 1] done: {n_seen} candidates, {time.time() - t0:.1f}s")

    # --- 3. Pass 2: score everyone, keep a running top-N heap ------------
    print("[pass 2] scoring candidates ...")
    t0 = time.time()
    heap: list[tuple] = []
    counter = itertools.count()
    n_total = 0
    n_honeypot = 0

    for candidate in io_utils.iter_candidates(args.candidates):
        n_total += 1
        if check_honeypot(candidate):
            n_honeypot += 1
            continue
        scored = scoring.score_candidate(candidate, idf)
        entry = (scored.score, next(counter), candidate["candidate_id"], candidate, scored)
        if len(heap) < args.top_n:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)

        if n_total % 20000 == 0:
            print(f"  ... {n_total} processed ({time.time() - t0:.1f}s)")

    print(f"[pass 2] done: {n_total} processed, {n_honeypot} honeypots excluded "
          f"({n_honeypot/n_total:.2%}), {time.time() - t0:.1f}s")

    if len(heap) < args.top_n:
        print(f"[warn] only {len(heap)} non-honeypot candidates available, "
              f"fewer than requested top-{args.top_n}.")

    # --- 4. Sort: score descending, candidate_id ascending on exact ties -
    ranked = sorted(heap, key=lambda e: (-e[0], e[2]))

    rows = []
    for rank, (score, _, cid, candidate, scored) in enumerate(ranked, start=1):
        reasoning = build_reasoning(candidate, scored)
        rows.append({"candidate_id": cid, "rank": rank, "score": score, "reasoning": reasoning})

    io_utils.write_submission_csv(args.out, rows)
    print(f"[output] wrote {len(rows)} rows to {args.out}")

    print("\nTop 10:")
    for r in rows[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  {r['score']:.4f}  {r['reasoning'][:110]}")

    print(f"\n[done] total runtime {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
