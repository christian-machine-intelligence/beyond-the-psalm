"""Build an extended Wikipedia corpus for per-book length-matched controls.

Goal: produce data/wikipedia_corpus.txt covering the longest book (Psalms,
~234 KB). The first 19,779 chars are taken verbatim from the existing
data/wikipedia_control.txt so that books with char counts at or below
that length see exactly the same control content as ICMI-018's published
single-control runs. Beyond 19.8 KB we extend with additional Wikipedia
articles in the same secular factual register (geography, geology,
astronomy, biology, geology of named regions, climate, atmospheric
science).

Output:
  data/wikipedia_corpus.txt  --  ~280 KB of UTF-8 plaintext

Run anywhere that has internet access. No auth required (Wikipedia REST API).

Usage:
    python3 build_wikipedia_corpus.py [--target-chars 280000]
"""

from __future__ import annotations

import argparse
import json as _json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).parent
DATA = ROOT / "data"
EXISTING = DATA / "wikipedia_control.txt"
OUT = DATA / "wikipedia_corpus.txt"

# Articles in the same register as the original control (geography of Iceland +
# Earth's atmosphere). All neutral factual prose — geography, geology, astronomy,
# atmospheric science, biology — chosen for compatibility with the existing
# control rather than per-book content matching.
ARTICLES = [
    # Geography / geology — extensions of the Iceland geology theme
    "Geography_of_Norway",
    "Geography_of_New_Zealand",
    "Geography_of_Greenland",
    "Plate_tectonics",
    "Volcanism",
    "Mineralogy",
    # Atmospheric science — extensions of the atmosphere theme
    "Climate",
    "Stratosphere",
    "Cloud",
    "Precipitation",
    # Astronomy
    "Solar_System",
    "Star",
    "Galaxy",
    "Planet",
    "Asteroid_belt",
    # Biology
    "Cell_(biology)",
    "Photosynthesis",
    "Mitochondrion",
    "Ecosystem",
    "Biome",
]

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_EXTRACT = (
    "https://en.wikipedia.org/w/api.php?"
    "action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles="
)


def fetch_extract(title: str) -> str:
    """Fetch the plain-text extract of a Wikipedia article via curl.

    Uses curl rather than urllib because urllib on macOS often can't find
    the system cert chain; curl uses the keychain.
    """
    url = WIKI_EXTRACT + quote(title)
    cmd = ["curl", "-sL", "--max-time", "30",
           "-A", "ICMI-018-control-builder/1.0", url]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(f"curl failed: rc={out.returncode} {out.stderr[:200]}")
    data = _json.loads(out.stdout)
    pages = data.get("query", {}).get("pages", {})
    for _pid, page in pages.items():
        text = page.get("extract", "")
        if text:
            return text
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-chars", type=int, default=280000,
                   help="Stop appending once corpus reaches this many chars")
    p.add_argument("--no-network", action="store_true",
                   help="Skip Wikipedia fetches; just copy existing control "
                        "(useful for dry-run / testing the pipeline)")
    args = p.parse_args()

    if not EXISTING.exists():
        sys.exit(f"missing {EXISTING}")

    base = EXISTING.read_text(encoding="utf-8").rstrip() + "\n\n"
    print(f"seed: {EXISTING.name} ({len(base):,} chars)")

    if args.no_network:
        OUT.write_text(base, encoding="utf-8")
        print(f"wrote {OUT} ({len(base):,} chars) — NO NETWORK MODE")
        return

    pieces = [base]
    total = len(base)
    for title in ARTICLES:
        if total >= args.target_chars:
            break
        readable = title.replace("_", " ")
        try:
            print(f"  fetch {readable!r}...", end=" ", flush=True)
            text = fetch_extract(title)
        except Exception as e:
            print(f"FAILED ({e})")
            time.sleep(2)
            continue
        if not text:
            print("empty extract")
            continue
        block = f"{readable}\n\n{text.strip()}\n\n"
        pieces.append(block)
        total += len(block)
        print(f"{len(block):,} chars  (running: {total:,})")
        time.sleep(0.5)  # be polite to Wikipedia

    if total < args.target_chars:
        print(f"WARNING: only assembled {total:,} chars, target was "
              f"{args.target_chars:,}. Re-run after adding more articles "
              f"to ARTICLES in {Path(__file__).name}.")

    OUT.write_text("".join(pieces), encoding="utf-8")
    print(f"\nwrote {OUT}: {total:,} chars  ({len(pieces)-1} extension articles)")


if __name__ == "__main__":
    main()
