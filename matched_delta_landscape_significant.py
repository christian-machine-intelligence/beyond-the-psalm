"""Abridged matched-Δ landscape — only the 19 Bonferroni-significant books.

A trimmed companion to `matched_delta_landscape.py` that drops the 47
below-detection (Tier C) books and shows only Tier A + Tier B. Useful
as a focused visual for the headline result without the long tail of
underpowered cells.

Output: figures/matched_delta_landscape_significant.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
N_COMP = 66
ALPHA_BONF = 0.05 / N_COMP

BOOKS = [
    ("Genesis", "genesis", "torah"), ("Exodus", "exodus", "torah"),
    ("Leviticus", "leviticus", "torah"), ("Numbers", "numbers", "torah"),
    ("Deuteronomy", "deuteronomy", "torah"),
    ("Joshua", "joshua", "historical"), ("Judges", "judges", "historical"),
    ("Ruth", "ruth", "narrative"),
    ("1 Samuel", "1_samuel", "historical"), ("2 Samuel", "2_samuel", "historical"),
    ("1 Kings", "1_kings", "historical"), ("2 Kings", "2_kings", "historical"),
    ("1 Chronicles", "1_chronicles", "historical"),
    ("2 Chronicles", "2_chronicles", "historical"),
    ("Ezra", "ezra", "historical"), ("Nehemiah", "nehemiah", "historical"),
    ("Esther", "esther", "narrative"),
    ("Job", "job", "wisdom"), ("Psalms", "psalms", "wisdom"),
    ("Proverbs", "proverbs", "wisdom"), ("Ecclesiastes", "ecclesiastes", "wisdom"),
    ("Song of Solomon", "song_of_solomon", "narrative"),
    ("Isaiah", "isaiah", "prophet"), ("Jeremiah", "jeremiah", "prophet"),
    ("Lamentations", "lamentations", "prophet"), ("Ezekiel", "ezekiel", "prophet"),
    ("Daniel", "daniel", "apocalyptic"),
    ("Hosea", "hosea", "prophet"), ("Joel", "joel", "prophet"),
    ("Amos", "amos", "prophet"), ("Obadiah", "obadiah", "prophet"),
    ("Jonah", "jonah", "prophet"), ("Micah", "micah", "prophet"),
    ("Nahum", "nahum", "prophet"), ("Habakkuk", "habakkuk", "prophet"),
    ("Zephaniah", "zephaniah", "prophet"), ("Haggai", "haggai", "prophet"),
    ("Zechariah", "zechariah", "prophet"), ("Malachi", "malachi", "prophet"),
    ("Matthew", "matthew", "gospel"), ("Mark", "mark", "gospel"),
    ("Luke", "luke", "gospel"), ("John", "john", "gospel"),
    ("Acts", "acts", "gospel"),
    ("Romans", "romans", "epistle"),
    ("1 Corinthians", "1_corinthians", "epistle"),
    ("2 Corinthians", "2_corinthians", "epistle"),
    ("Galatians", "galatians", "epistle"), ("Ephesians", "ephesians", "epistle"),
    ("Philippians", "philippians", "epistle"),
    ("Colossians", "colossians", "epistle"),
    ("1 Thessalonians", "1_thessalonians", "epistle"),
    ("2 Thessalonians", "2_thessalonians", "epistle"),
    ("1 Timothy", "1_timothy", "epistle"), ("2 Timothy", "2_timothy", "epistle"),
    ("Titus", "titus", "epistle"), ("Philemon", "philemon", "epistle"),
    ("Hebrews", "hebrews", "epistle"), ("James", "james", "epistle"),
    ("1 Peter", "1_peter", "epistle"), ("2 Peter", "2_peter", "epistle"),
    ("1 John", "1_john", "epistle"),
    ("2 John", "2_john", "epistle"), ("3 John", "3_john", "epistle"),
    ("Jude", "jude", "epistle"),
    ("Revelation", "revelation", "apocalyptic"),
]

TIER_A = {"1 Peter", "2 Timothy", "1 Timothy",
          "2 Corinthians", "Romans", "1 Corinthians"}


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


def ci95(arr):
    sd = arr.std(ddof=1)
    t_crit = 2.776 if len(arr) == 5 else 2.0
    return t_crit * sd / (len(arr) ** 0.5)


def main():
    psalm = per_run_means("qwen35_9b_thinkoff_psalm.json")
    van = per_run_means("qwen35_9b_thinkoff_vanilla.json")
    psalm_delta_van = (psalm.mean() - van.mean()) * 100

    rows = []
    for canonical, slug, genre in BOOKS:
        bible = per_run_means(f"qwen35_9b_thinkoff_bible_{slug}.json")
        wikictrl = per_run_means(f"qwen35_9b_thinkoff_wikictrl_{slug}.json")
        if bible is None or wikictrl is None:
            continue
        diff_runs = (bible - wikictrl) * 100
        matched_delta = diff_runs.mean()
        diff_ci = ci95(diff_runs)
        t, p = stats.ttest_rel(bible, wikictrl)
        is_bonf = (p < ALPHA_BONF)
        is_tier_a = canonical in TIER_A
        # Filter to Tier A + Tier B (= Bonf-significant) only
        if not (is_bonf or is_tier_a):
            continue
        rows.append({
            "book": canonical, "genre": genre,
            "matched_delta": matched_delta,
            "ci95": diff_ci,
            "p_matched": p,
            "tier_a": is_tier_a,
        })

    print(f"Loaded {len(rows)}/19 Bonferroni-significant books (Tier A + Tier B)")

    rows.sort(key=lambda r: r["matched_delta"])

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    genre_colors = {
        "epistle": "#c0392b",
        "gospel": "#e67e22",
        "prophet": "#8e44ad",
        "torah": "#2980b9",
        "historical": "#16a085",
        "wisdom": "#f39c12",
        "apocalyptic": "#7f8c8d",
        "narrative": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ys = np.arange(len(rows))
    vals = [r["matched_delta"] for r in rows]
    errs = [r["ci95"] for r in rows]
    colors = [genre_colors[r["genre"]] for r in rows]
    edgecolors = ["#222" if r["tier_a"] else "black" for r in rows]
    linewidths = [1.4 if r["tier_a"] else 0.6 for r in rows]

    ax.barh(ys, vals, xerr=errs, color=colors,
            edgecolor=edgecolors, linewidth=linewidths,
            error_kw={"elinewidth": 0.7, "ecolor": "#333"})

    ax.set_yticks(ys)
    ax.set_yticklabels([r["book"] for r in rows], fontsize=9)
    ax.set_xlabel("Δ vs length-matched Wikipedia control (pp)", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(psalm_delta_van, color="#1f77b4", linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"10-psalm Δ vs vanilla ({psalm_delta_van:+.1f} pp)")

    n_a = sum(1 for r in rows if r["tier_a"])
    n_b = len(rows) - n_a
    ax.set_title("Bonferroni-Significant Books — Qwen 3.5 9B\n"
                 f"19 of 66 books cross α = 0.05/66 vs matched Wiki control "
                 f"({n_a} Tier A + {n_b} Tier B)",
                 fontsize=11)

    genre_handles = [Patch(color=c, label=g) for g, c in genre_colors.items()
                     if g in {r["genre"] for r in rows}]
    handles = [
        Patch(facecolor="white", edgecolor="#222", linewidth=1.4,
              label="Tier A (≥ 10-psalm)"),
        *genre_handles,
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=1,
              framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, max(vals) + 1.5)
    fig.tight_layout()

    out = Path(__file__).parent / "matched_delta_landscape_significant.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
