"""Compare aggregate accuracy of pre-fix vs. post-fix runs on the spot-check
books (ruth, ezra, genesis). Backs the §6 Limitations claim that the cache
flags don't shift outputs in expectation.

For each book, reports:
  - per-virtue mean accuracy (n=5 seeds × 100 scenarios = 500 trials per virtue)
  - grand mean (n=20 seed-virtue cells)
  - 95% CI on grand mean
  - paired-t test on per-run means: pre-fix vs post-fix
  - decision: PASS if |Δ| < 1 pp AND paired-t p > 0.05

Run this AFTER both the main run finishes AND run_marx_9b_postfix_spotcheck.sh
has produced *_postfixcheck.json files.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
BOOKS = ["ruth", "ezra", "genesis"]


def per_run_means(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        rows = json.load(f)
    by_run = defaultdict(list)
    for r in rows:
        if r.get("accuracy") is None:
            continue
        by_run[r["run_index"]].append(r["accuracy"])
    if not by_run:
        return None
    return np.array([np.mean(v) for _, v in sorted(by_run.items())])


def per_virtue_means(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    by_v = defaultdict(list)
    for r in rows:
        if r.get("accuracy") is None:
            continue
        by_v[r["virtue"]].append(r["accuracy"])
    return {v: float(np.mean(a)) for v, a in by_v.items()}


def fmt_ci(arr):
    m = arr.mean() * 100
    sd = arr.std(ddof=1) * 100
    t_crit = 2.776 if len(arr) == 5 else 2.0  # df=4 for n=5
    ci = t_crit * sd / (len(arr) ** 0.5)
    return m, ci


def main():
    print(f"{'Book':<12} {'Virtue':<11} {'Pre-fix':>10} {'Post-fix':>10} "
          f"{'Δ pp':>7} {'paired-t p':>11} {'verdict':>10}")
    print("-" * 78)

    overall_results = []
    for book in BOOKS:
        pre_runs = per_run_means(f"qwen35_9b_thinkoff_wikictrl_{book}.json")
        post_runs = per_run_means(f"qwen35_9b_thinkoff_wikictrl_{book}_postfixcheck.json")
        if pre_runs is None or post_runs is None:
            print(f"{book:<12} MISSING — pre={pre_runs is not None} post={post_runs is not None}")
            continue

        pre_v = per_virtue_means(f"qwen35_9b_thinkoff_wikictrl_{book}.json")
        post_v = per_virtue_means(f"qwen35_9b_thinkoff_wikictrl_{book}_postfixcheck.json")
        for v in ["prudence", "justice", "courage", "temperance"]:
            pre_m = pre_v.get(v, 0) * 100
            post_m = post_v.get(v, 0) * 100
            delta = post_m - pre_m
            print(f"{book:<12} {v:<11} {pre_m:>9.1f}% {post_m:>9.1f}% "
                  f"{delta:>+7.2f}")

        # Grand mean comparison
        pre_m, pre_ci = fmt_ci(pre_runs)
        post_m, post_ci = fmt_ci(post_runs)
        delta = post_m - pre_m
        t, p = stats.ttest_rel(pre_runs, post_runs)
        verdict = "PASS" if abs(delta) < 1.0 and p > 0.05 else "FLAG"

        print(f"{book:<12} {'GRAND':<11} {pre_m:>8.1f}±{pre_ci:.1f}% "
              f"{post_m:>8.1f}±{post_ci:.1f}% {delta:>+7.2f} {p:>11.3f} "
              f"{verdict:>10}")
        print()
        overall_results.append((book, pre_m, post_m, delta, p, verdict))

    # Aggregate verdict
    print("=" * 78)
    if not overall_results:
        print("No data to compare. Run the spot-check script first.")
        return
    deltas = np.array([r[3] for r in overall_results])
    print(f"Across {len(overall_results)} books:")
    print(f"  mean |Δ| = {np.mean(np.abs(deltas)):.2f} pp")
    print(f"  max  |Δ| = {np.max(np.abs(deltas)):.2f} pp")
    n_pass = sum(1 for r in overall_results if r[5] == "PASS")
    print(f"  {n_pass}/{len(overall_results)} books PASS the |Δ|<1pp + p>0.05 criterion")
    if n_pass == len(overall_results):
        print("\n→ Cache fix does not shift outputs. Pre-fix and post-fix data can be "
              "pooled in the analysis without correction.")
    else:
        print("\n→ At least one book shows a shift. Investigate before pooling.")


if __name__ == "__main__":
    main()
