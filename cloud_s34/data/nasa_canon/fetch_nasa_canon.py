#!/usr/bin/env python3
"""Fetch NASA Five Millennium Canon catalog pages (SEcat5 / LEcat5) for the
century ranges that the local machine could not reach, verify that each page
parses into eclipse rows, and write a manifest with sha256 / size / row counts.

Usage:
    python fetch_nasa_canon.py            # fetch default ranges into ./pages
    python fetch_nasa_canon.py --verify   # only re-verify files already present

The default ranges are exactly the gap described in the project log
(S33 续 6): the local NASA canon stops at 299 BCE, so Babylonian diary records
before that year (adart1) were left unlabelled; and the local canon also stops
at 1700 CE, which bounds 《清實錄》 SE/LE extraction.  Pages:
    SE/LE  -0699..-0600, -0599..-0500, -0499..-0400, -0399..-0300  (8 pages)
    SE/LE   1701..1800,   1801..1900                                (4 pages)
Only standard-library modules are used so the script runs anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://eclipse.gsfc.nasa.gov"
HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"

# (kind, start_year, end_year) -> NASA uses astronomical year numbering with a
# leading sign and four digits; "-0699--0600" means 700 BCE .. 601 BCE.
DEFAULT_RANGES = [
    ("SE", -699, -600), ("LE", -699, -600),
    ("SE", -599, -500), ("LE", -599, -500),
    ("SE", -499, -400), ("LE", -499, -400),
    ("SE", -399, -300), ("LE", -399, -300),
    ("SE", 1701, 1800), ("LE", 1701, 1800),
    ("SE", 1801, 1900), ("LE", 1801, 1900),
]

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
# A catalog row begins with the catalog number, then the calendar date, e.g.
#   "00001  -1999 Jun 12  03:14:51 ..." (SE) / "00001 -1999 Jun 26 ..." (LE).
ROW_RE = re.compile(
    r"^\s*(\d{5})\s+(-?\d{4})\s+(" + "|".join(MONTHS) + r")\s+(\d{1,2})\s+(\d{2}:\d{2}:\d{2})"
)


def ystr(y: int) -> str:
    return f"{y:05d}" if y < 0 else f"{y:04d}"


def url_for(kind: str, y0: int, y1: int) -> str:
    cat = "SEcat5" if kind == "SE" else "LEcat5"
    return f"{BASE}/{cat}/{kind}{ystr(y0)}-{ystr(y1)}.html"


def fname_for(kind: str, y0: int, y1: int) -> str:
    return f"{kind}{ystr(y0)}-{ystr(y1)}.html"


def fetch(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "verichrono-canon-fetch/1.0 (research; contact via repo)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - we want to retry on anything
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed after {retries} tries: {url}: {last}")


def parse_rows(text: str, kind: str, y0: int, y1: int) -> dict:
    """Count catalog rows and check that every row's year lies in [y0, y1]."""
    # Catalog rows live inside <pre> blocks; strip tags so anchors inside rows don't break the regex.
    body = re.sub(r"<[^>]+>", "", text)
    body = html.unescape(body)
    rows, bad_year, years = 0, 0, []
    for line in body.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        rows += 1
        y = int(m.group(2))
        years.append(y)
        if not (y0 <= y <= y1):
            bad_year += 1
    return {
        "rows": rows,
        "rows_outside_range": bad_year,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        # A century of the canon normally holds ~220-260 solar and ~220-260 lunar eclipses.
        "plausible": 150 <= rows <= 320 and bad_year == 0,
    }


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="do not download; re-verify existing pages")
    args = ap.parse_args()
    PAGES.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "source": "NASA GSFC Five Millennium Canon of Solar/Lunar Eclipses (Espenak & Meeus), catalog pages",
        "note": "Pages are the publicly served HTML catalog tables; verify rows count ~200-260 per century.",
        "pages": [],
    }
    ok = True
    for kind, y0, y1 in DEFAULT_RANGES:
        url = url_for(kind, y0, y1)
        path = PAGES / fname_for(kind, y0, y1)
        if args.verify:
            if not path.exists():
                print(f"MISSING {path.name}")
                ok = False
                continue
            data = path.read_bytes()
        else:
            t0 = time.time()
            data = fetch(url)
            path.write_bytes(data)
            print(f"fetched {path.name:>22}  {len(data):>7} bytes  {time.time()-t0:5.2f}s")
        info = parse_rows(data.decode("utf-8", errors="replace"), kind, y0, y1)
        entry = {"file": f"pages/{path.name}", "url": url, "bytes": len(data), "sha256": sha256(data), **info}
        manifest["pages"].append(entry)
        flag = "OK " if info["plausible"] else "BAD"
        print(f"  {flag} rows={info['rows']:>3} years={info['year_min']}..{info['year_max']} outside={info['rows_outside_range']}")
        ok = ok and info["plausible"]
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest ->", HERE / "MANIFEST.json")
    print("ALL PLAUSIBLE" if ok else "SOME PAGES FAILED VALIDATION")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
