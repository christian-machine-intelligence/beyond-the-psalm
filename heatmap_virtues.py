"""Per-book × per-virtue heatmap of injection gain (Δ vs vanilla).

Y-axis: 66 books + 1 baseline row (10-psalm), sorted by overall gain.
X-axis: prudence, justice, courage, temperance.
Cell color: Δ accuracy vs vanilla (diverging colormap centered at 0).
Cell value: signed pp gain.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

BOOKS = [
    ("Genesis", "genesis"), ("Exodus", "exodus"), ("Leviticus", "leviticus"),
    ("Numbers", "numbers"), ("Deuteronomy", "deuteronomy"),
    ("Joshua", "joshua"), ("Judges", "judges"), ("Ruth", "ruth"),
    ("1 Samuel", "1_samuel"), ("2 Samuel", "2_samuel"),
    ("1 Kings", "1_kings"), ("2 Kings", "2_kings"),
    ("1 Chronicles", "1_chronicles"), ("2 Chronicles", "2_chronicles"),
    ("Ezra", "ezra"), ("Nehemiah", "nehemiah"), ("Esther", "esther"),
    ("Job", "job"), ("Psalms", "psalms"), ("Proverbs", "proverbs"),
    ("Ecclesiastes", "ecclesiastes"), ("Song of Solomon", "song_of_solomon"),
    ("Isaiah", "isaiah"), ("Jeremiah", "jeremiah"), ("Lamentations", "lamentations"),
    ("Ezekiel", "ezekiel"), ("Daniel", "daniel"),
    ("Hosea", "hosea"), ("Joel", "joel"), ("Amos", "amos"), ("Obadiah", "obadiah"),
    ("Jonah", "jonah"), ("Micah", "micah"), ("Nahum", "nahum"),
    ("Habakkuk", "habakkuk"), ("Zephaniah", "zephaniah"), ("Haggai", "haggai"),
    ("Zechariah", "zechariah"), ("Malachi", "malachi"),
    ("Matthew", "matthew"), ("Mark", "mark"), ("Luke", "luke"),
    ("John", "john"), ("Acts", "acts"),
    ("Romans", "romans"), ("1 Corinthians", "1_corinthians"),
    ("2 Corinthians", "2_corinthians"), ("Galatians", "galatians"),
    ("Ephesians", "ephesians"), ("Philippians", "philippians"),
    ("Colossians", "colossians"), ("1 Thessalonians", "1_thessalonians"),
    ("2 Thessalonians", "2_thessalonians"),
    ("1 Timothy", "1_timothy"), ("2 Timothy", "2_timothy"),
    ("Titus", "titus"), ("Philemon", "philemon"), ("Hebrews", "hebrews"),
    ("James", "james"), ("1 Peter", "1_peter"), ("2 Peter", "2_peter"),
    ("1 John", "1_john"), ("2 John", "2_john"), ("3 John", "3_john"),
    ("Jude", "jude"), ("Revelation", "revelation"),
]

VIRTUES = ["prudence", "justice", "courage", "temperance"]


def per_virtue_means(filename: str) -> dict[str, float]:
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


def main():
    # Baselines
    van = per_virtue_means("qwen35_9b_thinkoff_vanilla.json")
    psalm = per_virtue_means("qwen35_9b_thinkoff_psalm.json")

    # Per-book
    book_data = []
    for canonical, slug in BOOKS:
        m = per_virtue_means(f"qwen35_9b_thinkoff_bible_{slug}.json")
        if not m:
            continue
        book_data.append((canonical, m))

    # Sort by overall mean accuracy descending
    book_data.sort(key=lambda kv: -np.mean(list(kv[1].values())))

    # Build matrix: rows = ten-psalm baseline + 66 books, cols = virtues
    rows = []
    rows.append(("10-psalm baseline", psalm))
    rows.extend(book_data)

    deltas = np.array([
        [m[v] - van[v] for v in VIRTUES]
        for _, m in rows
    ]) * 100  # to pp

    raw = np.array([
        [m[v] for v in VIRTUES]
        for _, m in rows
    ]) * 100

    labels = [n for n, _ in rows]

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(7.2, 16))
    # Diverging colormap centered at 0; vmin slightly negative, vmax = max gain
    vmax = float(np.max(np.abs(deltas)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(deltas, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(len(VIRTUES)))
    ax.set_xticklabels([v.capitalize() for v in VIRTUES], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate each cell with the signed pp gain
    for i, label in enumerate(labels):
        for j, v in enumerate(VIRTUES):
            val = deltas[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                    fontsize=6, color=color)

    # Highlight baseline row
    for i, n in enumerate(labels):
        if n == "10-psalm baseline":
            ax.axhline(i + 0.5, color="black", linewidth=0.8, alpha=0.5)

    ax.set_title("Per-book × per-virtue gain (pp vs vanilla)\n"
                 "Books sorted by overall accuracy (descending)\nQwen 3.5 9B, VirtueBench V2 ratio",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Δ vs vanilla (pp)", fontsize=9)

    fig.tight_layout()
    out = Path(__file__).parent / "virtue_heatmap.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
