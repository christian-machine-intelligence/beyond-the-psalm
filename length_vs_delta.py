"""Scatterplot: book length (tokens) vs Δ-vs-vanilla.

Tests the §5.3 claim that length does not predict effect. Computes
token counts using the same Qwen 3.5 tokenizer (or a BPE proxy) as
inference, plots all 66 books with the top-6 (Tier A) and bottom-5
(sub-Wikipedia) outliers labeled, and reports Spearman + Pearson r.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make virtue-bench's KJV loader importable
VB2 = Path(__file__).parent.parent / "virtue-bench-2" / "src"
sys.path.insert(0, str(VB2))

from virtue_bench.core.bible import load_bible_text  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

# (canonical name, slug, virtue-bench 3-letter id)
BOOKS = [
    ("Genesis", "genesis", "GEN"), ("Exodus", "exodus", "EXO"),
    ("Leviticus", "leviticus", "LEV"), ("Numbers", "numbers", "NUM"),
    ("Deuteronomy", "deuteronomy", "DEU"),
    ("Joshua", "joshua", "JOS"), ("Judges", "judges", "JDG"),
    ("Ruth", "ruth", "RUT"),
    ("1 Samuel", "1_samuel", "1SA"), ("2 Samuel", "2_samuel", "2SA"),
    ("1 Kings", "1_kings", "1KI"), ("2 Kings", "2_kings", "2KI"),
    ("1 Chronicles", "1_chronicles", "1CH"),
    ("2 Chronicles", "2_chronicles", "2CH"),
    ("Ezra", "ezra", "EZR"), ("Nehemiah", "nehemiah", "NEH"),
    ("Esther", "esther", "EST"),
    ("Job", "job", "JOB"), ("Psalms", "psalms", "PSA"),
    ("Proverbs", "proverbs", "PRO"),
    ("Ecclesiastes", "ecclesiastes", "ECC"),
    ("Song of Solomon", "song_of_solomon", "SNG"),
    ("Isaiah", "isaiah", "ISA"), ("Jeremiah", "jeremiah", "JER"),
    ("Lamentations", "lamentations", "LAM"),
    ("Ezekiel", "ezekiel", "EZK"), ("Daniel", "daniel", "DAN"),
    ("Hosea", "hosea", "HOS"), ("Joel", "joel", "JOL"),
    ("Amos", "amos", "AMO"), ("Obadiah", "obadiah", "OBA"),
    ("Jonah", "jonah", "JON"), ("Micah", "micah", "MIC"),
    ("Nahum", "nahum", "NAM"), ("Habakkuk", "habakkuk", "HAB"),
    ("Zephaniah", "zephaniah", "ZEP"), ("Haggai", "haggai", "HAG"),
    ("Zechariah", "zechariah", "ZEC"), ("Malachi", "malachi", "MAL"),
    ("Matthew", "matthew", "MAT"), ("Mark", "mark", "MRK"),
    ("Luke", "luke", "LUK"), ("John", "john", "JHN"),
    ("Acts", "acts", "ACT"),
    ("Romans", "romans", "ROM"),
    ("1 Corinthians", "1_corinthians", "1CO"),
    ("2 Corinthians", "2_corinthians", "2CO"),
    ("Galatians", "galatians", "GAL"),
    ("Ephesians", "ephesians", "EPH"),
    ("Philippians", "philippians", "PHP"),
    ("Colossians", "colossians", "COL"),
    ("1 Thessalonians", "1_thessalonians", "1TH"),
    ("2 Thessalonians", "2_thessalonians", "2TH"),
    ("1 Timothy", "1_timothy", "1TI"), ("2 Timothy", "2_timothy", "2TI"),
    ("Titus", "titus", "TIT"), ("Philemon", "philemon", "PHM"),
    ("Hebrews", "hebrews", "HEB"), ("James", "james", "JAS"),
    ("1 Peter", "1_peter", "1PE"), ("2 Peter", "2_peter", "2PE"),
    ("1 John", "1_john", "1JN"), ("2 John", "2_john", "2JN"),
    ("3 John", "3_john", "3JN"), ("Jude", "jude", "JUD"),
    ("Revelation", "revelation", "REV"),
]

# Tier A: matches/exceeds 10-psalm baseline
TIER_A = {"1 Peter", "2 Timothy", "1 Timothy",
          "2 Corinthians", "Romans", "1 Corinthians"}
# Bottom-5 by matched-Wikipedia Δ (smallest content-specific effects)
BOTTOM_5 = {"Philemon", "Judges", "Song of Solomon", "Ruth", "Jude"}


def per_run_means(filename: str) -> np.ndarray:
    path = RESULTS_DIR / filename
    if not path.exists():
        return np.array([])
    with open(path) as f:
        rows = json.load(f)
    by_run = defaultdict(list)
    for r in rows:
        if r.get("accuracy") is None:
            continue
        by_run[r["run_index"]].append(r["accuracy"])
    return np.array([np.mean(v) for _, v in sorted(by_run.items())])


def get_tokenizer():
    """Try Qwen tokenizer, fall back to GPT-2 BPE if unavailable."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
        return tok, "Qwen 2.5 BPE (proxy for Qwen 3.5)"
    except Exception:
        pass
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base"), "cl100k_base BPE"
    except Exception:
        pass
    # crude word-count fallback
    class WordCount:
        def encode(self, s):
            return s.split()
    return WordCount(), "whitespace word count"


def main():
    # Per-book Δ vs vanilla AND vs per-book matched Wikipedia control
    van = per_run_means("qwen35_9b_thinkoff_vanilla.json").mean() * 100
    psalm = per_run_means("qwen35_9b_thinkoff_psalm.json").mean() * 100

    tok, tok_name = get_tokenizer()

    rows = []
    for canonical, slug, abbrev in BOOKS:
        bible_runs = per_run_means(f"qwen35_9b_thinkoff_bible_{slug}.json")
        wiki_runs = per_run_means(f"qwen35_9b_thinkoff_wikictrl_{slug}.json")
        if len(bible_runs) == 0:
            continue
        text = load_bible_text(books=[abbrev])
        ntoks = len(tok.encode(text))
        bible_m = bible_runs.mean() * 100
        wiki_m = wiki_runs.mean() * 100 if wiki_runs is not None else float("nan")
        rows.append({
            "book": canonical, "abbrev": abbrev,
            "tokens": ntoks, "chars": len(text),
            "delta_vanilla": bible_m - van,
            "delta_matched_wiki": bible_m - wiki_m,
        })
        print(f"{canonical:<18} tokens={ntoks:>6}  chars={len(text):>7}  "
              f"Δvan={rows[-1]['delta_vanilla']:+.1f}  "
              f"Δmatched-wiki={rows[-1]['delta_matched_wiki']:+.1f}")

    # Correlations
    from scipy import stats
    log_tokens = np.log10([r["tokens"] for r in rows])
    deltas_van = np.array([r["delta_vanilla"] for r in rows])
    deltas_matched = np.array([r["delta_matched_wiki"] for r in rows])

    pearson_van = stats.pearsonr(log_tokens, deltas_van)
    spearman_van = stats.spearmanr([r["tokens"] for r in rows], deltas_van)
    pearson_matched = stats.pearsonr(log_tokens, deltas_matched)
    spearman_matched = stats.spearmanr([r["tokens"] for r in rows], deltas_matched)

    print(f"\nTokenizer: {tok_name}")
    print(f"\n--- length vs Δ-vs-vanilla ---")
    print(f"  Pearson  (log10 tokens):  r={pearson_van.statistic:+.3f}  p={pearson_van.pvalue:.4f}")
    print(f"  Spearman (raw tokens):    ρ={spearman_van.statistic:+.3f}  p={spearman_van.pvalue:.4f}")
    print(f"\n--- length vs Δ-vs-MATCHED-Wikipedia ---")
    print(f"  Pearson  (log10 tokens):  r={pearson_matched.statistic:+.3f}  p={pearson_matched.pvalue:.4f}")
    print(f"  Spearman (raw tokens):    ρ={spearman_matched.statistic:+.3f}  p={spearman_matched.pvalue:.4f}")

    # Plot
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    xs = np.array([r["tokens"] for r in rows])
    ys = deltas_matched   # v2: y-axis is now matched-Wiki Δ (content effect)

    # Color/marker per group
    is_tier_a = np.array([r["book"] in TIER_A for r in rows])
    is_bottom = np.array([r["book"] in BOTTOM_5 for r in rows])
    is_other = ~(is_tier_a | is_bottom)

    # Plot in three layers
    ax.scatter(xs[is_other], ys[is_other], s=28, c="#7a7a7a",
               alpha=0.55, edgecolors="none", label="Other 55 books")
    ax.scatter(xs[is_bottom], ys[is_bottom], s=70, c="#c0392b",
               edgecolors="black", linewidths=0.6, marker="v",
               label="Bottom-5 by matched Δ (5)", zorder=3)
    ax.scatter(xs[is_tier_a], ys[is_tier_a], s=70, c="#27ae60",
               edgecolors="black", linewidths=0.6, marker="^",
               label="Tier A: ≥ 10-psalm baseline (6)", zorder=3)

    # Reference line at 0 — matched control floor; every book is above it
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.6,
               label="Matched-Wiki floor (0 pp)")

    # Linear regression line on log-tokens vs matched-Δ
    slope, intercept = np.polyfit(log_tokens, deltas_matched, 1)
    xfit = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 100)
    yfit = slope * np.log10(xfit) + intercept
    ax.plot(xfit, yfit, color="#444444", linewidth=1.2, alpha=0.5,
            linestyle="-", label=f"Linear fit on log₁₀ tokens (r={pearson_matched.statistic:+.2f})")

    # Label outliers
    for r in rows:
        if r["book"] in TIER_A or r["book"] in BOTTOM_5:
            ax.annotate(r["book"], xy=(r["tokens"], r["delta_matched_wiki"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("Book length (Qwen tokens, log scale)", fontsize=10)
    ax.set_ylabel("Δ accuracy vs length-matched Wikipedia control (pp)", fontsize=10)
    ax.set_title("Book length vs content-specific effect (66 books, matched-Wiki Δ)\n"
                 f"Pearson r(log₁₀ tokens, matched-Δ) = {pearson_matched.statistic:+.2f}, "
                 f"p = {pearson_matched.pvalue:.3f}",
                 fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    out = Path(__file__).parent / "length_vs_delta.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"\nWrote {out}")

    # Also save the per-book length data for the appendix
    csv_path = Path(__file__).parent / "book_length_table.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["book", "abbrev", "tokens", "chars",
                                               "delta_vanilla", "delta_matched_wiki"])
        writer.writeheader()
        for r in sorted(rows, key=lambda x: -x["delta_vanilla"]):
            writer.writerow(r)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
