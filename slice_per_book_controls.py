"""Slice per-book length-matched Wikipedia control files.

Design (consistency-preserving):
  - The Wikipedia corpus (data/wikipedia_corpus.txt) is built once and is
    invariant across books.
  - For each book, the control is corpus[:book_chars] — the FIRST book_chars
    bytes of the corpus, where book_chars is the char-length of the KJV book
    text.
  - Every book therefore sees the same Wikipedia content as a *prefix*; only
    the truncation point differs. Two books of equal length get byte-identical
    controls.
  - Books at or below the original control's length (~19,779 chars) receive a
    truncation of the EXACT same content as the single control used in
    ICMI-018's published runs, preserving comparability.

Output:
  data/wikipedia_controls/<slug>.txt   --  one file per book

Usage:
    python3 slice_per_book_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CORPUS = DATA / "wikipedia_corpus.txt"
OUT_DIR = DATA / "wikipedia_controls"

# Make virtue-bench's KJV loader importable so we can compute exact KJV
# char counts per book using the same text the inference run injects.
VB2 = ROOT.parent / "virtue-bench-2" / "src"
sys.path.insert(0, str(VB2))
from virtue_bench.core.bible import load_bible_text  # noqa: E402

BOOKS = [
    ("genesis", "GEN"), ("exodus", "EXO"), ("leviticus", "LEV"),
    ("numbers", "NUM"), ("deuteronomy", "DEU"),
    ("joshua", "JOS"), ("judges", "JDG"), ("ruth", "RUT"),
    ("1_samuel", "1SA"), ("2_samuel", "2SA"),
    ("1_kings", "1KI"), ("2_kings", "2KI"),
    ("1_chronicles", "1CH"), ("2_chronicles", "2CH"),
    ("ezra", "EZR"), ("nehemiah", "NEH"), ("esther", "EST"),
    ("job", "JOB"), ("psalms", "PSA"), ("proverbs", "PRO"),
    ("ecclesiastes", "ECC"), ("song_of_solomon", "SNG"),
    ("isaiah", "ISA"), ("jeremiah", "JER"), ("lamentations", "LAM"),
    ("ezekiel", "EZK"), ("daniel", "DAN"),
    ("hosea", "HOS"), ("joel", "JOL"), ("amos", "AMO"),
    ("obadiah", "OBA"), ("jonah", "JON"), ("micah", "MIC"),
    ("nahum", "NAM"), ("habakkuk", "HAB"), ("zephaniah", "ZEP"),
    ("haggai", "HAG"), ("zechariah", "ZEC"), ("malachi", "MAL"),
    ("matthew", "MAT"), ("mark", "MRK"), ("luke", "LUK"),
    ("john", "JHN"), ("acts", "ACT"),
    ("romans", "ROM"), ("1_corinthians", "1CO"),
    ("2_corinthians", "2CO"), ("galatians", "GAL"),
    ("ephesians", "EPH"), ("philippians", "PHP"),
    ("colossians", "COL"), ("1_thessalonians", "1TH"),
    ("2_thessalonians", "2TH"),
    ("1_timothy", "1TI"), ("2_timothy", "2TI"),
    ("titus", "TIT"), ("philemon", "PHM"),
    ("hebrews", "HEB"), ("james", "JAS"),
    ("1_peter", "1PE"), ("2_peter", "2PE"),
    ("1_john", "1JN"), ("2_john", "2JN"), ("3_john", "3JN"),
    ("jude", "JUD"), ("revelation", "REV"),
]


def main():
    if not CORPUS.exists():
        sys.exit(f"missing {CORPUS}; run build_wikipedia_corpus.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = CORPUS.read_text(encoding="utf-8")
    print(f"corpus: {len(corpus):,} chars")

    longest_needed = 0
    longest_book = ""
    rows = []
    for slug, abbrev in BOOKS:
        book_text = load_bible_text(books=[abbrev])
        n = len(book_text)
        if n > longest_needed:
            longest_needed, longest_book = n, slug
        rows.append((slug, abbrev, n))

    if longest_needed > len(corpus):
        sys.exit(
            f"corpus is {len(corpus):,} chars but longest book "
            f"({longest_book}) needs {longest_needed:,}. "
            f"Extend ARTICLES in build_wikipedia_corpus.py and re-run."
        )

    print(f"longest book: {longest_book} = {longest_needed:,} chars (fits)")

    summary = []
    for slug, abbrev, n in rows:
        sliced = corpus[:n]
        out = OUT_DIR / f"{slug}.txt"
        out.write_text(sliced, encoding="utf-8")
        summary.append((slug, abbrev, n, len(sliced)))

    print(f"\nwrote {len(summary)} per-book controls in {OUT_DIR}/\n")
    print(f"{'slug':<18} {'abbrev':<6} {'book chars':>10} {'ctrl chars':>10}")
    for slug, abbrev, n_book, n_ctrl in summary:
        print(f"{slug:<18} {abbrev:<6} {n_book:>10,} {n_ctrl:>10,}")

    # Sanity: shortest 5 controls should be byte-identical prefixes of the
    # corpus (and therefore of each other up to the shortest length)
    print("\nByte-identity sanity check (every control is corpus[:n]):")
    shortest = sorted(summary, key=lambda x: x[2])[:3]
    longest = sorted(summary, key=lambda x: -x[2])[:3]
    for slug, _ab, n, _ in shortest + longest:
        ctrl = (OUT_DIR / f"{slug}.txt").read_text(encoding="utf-8")
        ok = ctrl == corpus[:n]
        print(f"  {slug:<18} ({n:>6,} chars)  prefix-match: {ok}")


if __name__ == "__main__":
    main()
