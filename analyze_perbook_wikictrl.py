"""Per-book length-matched Wikipedia control analysis — direct comparison
to the original single-control analysis from ICMI-018.

For each of the 66 books we now have:
  - bible_<slug>.json     : KJV book injection result (5 seeds)
  - wikictrl_<slug>.json  : Wikipedia control truncated to that book's exact
                            char length (5 seeds)

The per-book matched-control Δ is the apples-to-apples content effect for
that book — it controls for length whereas the original single-control did
not. This script reports:

1. New per-book Δ vs matched control, with paired-t and Bonferroni-corrected p
2. Re-tiered membership under the matched-control regime
3. Diff vs original analysis: which books moved tiers, which "below Wikipedia"
   findings survive when the control is matched
4. Summary table that ICMI-018's §6 Limitations promised
"""

from __future__ import annotations

import csv
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


# Mirror ICMI-018's published per-book "Δ Wiki" against the SINGLE control
ORIGINAL_SUB_WIKI = {"Philemon", "2 John", "3 John", "Jude", "Ecclesiastes"}

# Tier from ICMI-018 published table
ORIGINAL_TIER_A = {"1 Peter", "2 Timothy", "1 Timothy",
                   "2 Corinthians", "Romans", "1 Corinthians"}
ORIGINAL_TIER_B = {"1 Thessalonians", "Titus", "Philippians", "Hosea",
                   "Micah", "Exodus", "Zephaniah", "Ezekiel", "2 Chronicles"}


def main():
    # ----- Baselines -----
    print("=== Baselines (single-condition, identical to published) ===")
    van = per_run_means("qwen35_9b_thinkoff_vanilla.json")
    psalm = per_run_means("qwen35_9b_thinkoff_psalm.json")
    single_ctrl = per_run_means("qwen35_9b_thinkoff_control.json")
    if van is None or psalm is None or single_ctrl is None:
        print("ERROR: missing baseline files")
        return
    van_m = van.mean() * 100
    psalm_m = psalm.mean() * 100
    single_ctrl_m = single_ctrl.mean() * 100
    print(f"  vanilla:        {van_m:.2f}%")
    print(f"  10-psalm:       {psalm_m:.2f}%")
    print(f"  single Wiki:    {single_ctrl_m:.2f}%")
    print()

    # ----- Per-book analysis under MATCHED controls -----
    rows = []
    n_complete = 0
    for canonical, slug, genre in BOOKS:
        bible = per_run_means(f"qwen35_9b_thinkoff_bible_{slug}.json")
        wikictrl = per_run_means(f"qwen35_9b_thinkoff_wikictrl_{slug}.json")
        if bible is None or wikictrl is None:
            print(f"  SKIP {canonical}: bible={bible is not None} wikictrl={wikictrl is not None}")
            continue
        n_complete += 1

        bible_m = bible.mean() * 100
        wiki_m = wikictrl.mean() * 100
        single_delta = bible_m - single_ctrl_m   # ICMI-018 published Δ Wiki
        matched_delta = bible_m - wiki_m         # NEW: matched-control Δ

        # Paired-t against MATCHED control
        t_matched, p_matched = stats.ttest_rel(bible, wikictrl)
        # Paired-t against single global control (for comparison with published)
        t_single, p_single = stats.ttest_rel(bible, single_ctrl)

        # New tiering under matched-control regime
        if bible_m >= psalm_m:
            tier_new = "A"
        elif p_matched < ALPHA_BONF and matched_delta > 0:
            tier_new = "B"
        else:
            tier_new = "C"

        # Original tier
        if canonical in ORIGINAL_TIER_A:
            tier_old = "A"
        elif canonical in ORIGINAL_TIER_B:
            tier_old = "B"
        else:
            tier_old = "C"

        rows.append({
            "book": canonical, "slug": slug, "genre": genre,
            "bible": bible_m, "wikictrl_matched": wiki_m,
            "delta_vanilla": bible_m - van_m,
            "delta_single_ctrl": single_delta,
            "delta_matched_ctrl": matched_delta,
            "p_matched": p_matched,
            "p_single": p_single,
            "tier_old": tier_old,
            "tier_new": tier_new,
            "tier_changed": tier_old != tier_new,
            "below_wiki_old": canonical in ORIGINAL_SUB_WIKI,
            "below_wiki_new": matched_delta < 0,
        })

    print(f"Books in analysis: {n_complete}/66")
    print()

    # ----- Headline: which books changed below-Wikipedia status? -----
    print("=" * 90)
    print("HEADLINE — books at or below their matched control")
    print("=" * 90)
    below_new = sorted([r for r in rows if r["below_wiki_new"]],
                       key=lambda x: x["delta_matched_ctrl"])
    print(f"\nBooks with matched-Δ < 0 (matched-Wiki finding): {len(below_new)}")
    print(f"{'book':<18} {'len-matched':>11} {'old single-Δ':>13} {'new matched-Δ':>14} {'p (Bonf)':>11}")
    for r in below_new:
        bonf = min(r["p_matched"] * N_COMP, 1.0)
        print(f"{r['book']:<18} "
              f"{'YES' if r['below_wiki_old'] else 'no':>11} "
              f"{r['delta_single_ctrl']:>+12.2f}  "
              f"{r['delta_matched_ctrl']:>+13.2f}  "
              f"{bonf:>10.3f}")

    print(f"\nBooks below SINGLE control but at-or-above MATCHED control:")
    print(f"  (this is the 'short-book confound' the new control retires)")
    moved_up = [r for r in rows if r["below_wiki_old"] and not r["below_wiki_new"]]
    for r in moved_up:
        print(f"  {r['book']:<18} single-Δ={r['delta_single_ctrl']:+.2f}  "
              f"matched-Δ={r['delta_matched_ctrl']:+.2f}")

    # ----- Tier movements -----
    print()
    print("=" * 90)
    print("TIER CHANGES under matched-control regime")
    print("=" * 90)
    tier_change = [r for r in rows if r["tier_changed"]]
    if not tier_change:
        print("  No books changed tier.")
    else:
        for r in tier_change:
            print(f"  {r['book']:<18} {r['tier_old']} -> {r['tier_new']}  "
                  f"matched-Δ={r['delta_matched_ctrl']:+.2f}  "
                  f"p_matched={r['p_matched']:.4f}")

    # Tier counts
    new_a = [r for r in rows if r["tier_new"] == "A"]
    new_b = [r for r in rows if r["tier_new"] == "B"]
    new_c = [r for r in rows if r["tier_new"] == "C"]
    print(f"\nNew tier counts:")
    print(f"  Tier A (≥ 10-psalm baseline):       {len(new_a):>3} (was 6)")
    print(f"  Tier B (Bonf-sig > matched Wiki):   {len(new_b):>3} (was 9)")
    print(f"  Tier C (no detectable specificity): {len(new_c):>3} (was 51)")

    print(f"\nTier B membership under matched control:")
    for r in sorted(new_b, key=lambda x: -x["delta_matched_ctrl"]):
        print(f"  {r['book']:<18} matched-Δ={r['delta_matched_ctrl']:+.2f} "
              f"(genre={r['genre']})")

    # ----- Aggregate stats -----
    print()
    print("=" * 90)
    print("AGGREGATE")
    print("=" * 90)
    deltas_matched = np.array([r["delta_matched_ctrl"] for r in rows])
    deltas_single = np.array([r["delta_single_ctrl"] for r in rows])
    print(f"\nMatched-Δ across 66 books:")
    print(f"  mean = {deltas_matched.mean():+.2f} pp")
    print(f"  range = {deltas_matched.min():+.2f} to {deltas_matched.max():+.2f}")
    print(f"  n above 0 (book > matched Wiki): {(deltas_matched > 0).sum()}/66")
    print(f"  n Bonf-sig > matched Wiki:       "
          f"{sum(1 for r in rows if r['p_matched'] < ALPHA_BONF and r['delta_matched_ctrl'] > 0)}/66")

    print(f"\nFor reference (original published numbers):")
    print(f"  Single-Δ mean = {deltas_single.mean():+.2f} pp")
    print(f"  n above 0: {(deltas_single > 0).sum()}/66 (was 61/66 in published)")
    print(f"  n Bonf-sig: "
          f"{sum(1 for r in rows if r['p_single'] < ALPHA_BONF and r['delta_single_ctrl'] > 0)}/66 (was 47/66)")

    # ----- Genre means under matched control -----
    print()
    print("=" * 90)
    print("Genre-level Δ comparison (single vs matched)")
    print("=" * 90)
    by_genre = defaultdict(list)
    for r in rows:
        by_genre[r["genre"]].append(r)
    print(f"{'genre':<14} {'n':>3} {'single-Δ mean':>14} {'matched-Δ mean':>15} "
          f"{'shift':>8}")
    for g in sorted(by_genre.keys(), key=lambda x: -np.mean([r["delta_matched_ctrl"] for r in by_genre[x]])):
        single_mean = np.mean([r["delta_single_ctrl"] for r in by_genre[g]])
        matched_mean = np.mean([r["delta_matched_ctrl"] for r in by_genre[g]])
        print(f"{g:<14} {len(by_genre[g]):>3} {single_mean:>+13.2f}  "
              f"{matched_mean:>+14.2f}  {matched_mean - single_mean:>+7.2f}")

    # ----- Save CSV with everything -----
    csv_path = Path(__file__).parent / "matched_control_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in sorted(rows, key=lambda x: -x["bible"]):
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in r.items()})
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
