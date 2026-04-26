"""
Sanity-check URLs in the master CSVs.

For each launch:
  - HEAD/GET the canonical URL
  - Mark which ones return non-200 or redirect to a generic homepage
  - For misses, query the Wayback Machine and confirm a snapshot exists

Writes data/launches/url_status.csv with columns:
  launch_id, status, http_code, url_used, has_wayback, note

This does NOT modify the master CSVs. Use the report to fix them by hand or
let scrape.py fall back to Wayback automatically.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LAUNCHES_DIR = ROOT / "data" / "launches"
UA = "Mozilla/5.0 (compatible; LaunchCorpusBot/0.1)"
HEADERS = {"User-Agent": UA}


def check_live(url: str) -> tuple[int, str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
        return r.status_code, r.url
    except Exception as e:
        return 0, str(e)


def check_wayback(url: str, date: str) -> bool:
    ts = date.replace("-", "")
    api = f"https://archive.org/wayback/available?url={url}&timestamp={ts}"
    try:
        r = requests.get(api, headers=HEADERS, timeout=20)
        snap = r.json().get("archived_snapshots", {}).get("closest")
        return bool(snap and snap.get("available"))
    except Exception:
        return False


def check_one(row: dict) -> dict:
    url = row["canonical_url"].strip()
    if not url:
        return {**row, "status": "no_url", "http_code": "", "url_used": "",
                "has_wayback": "", "note": "missing url"}
    code, url_used = check_live(url)
    ok_live = code == 200 and url_used and not is_landing(url_used)
    has_wb = "" if ok_live else ("yes" if check_wayback(url, row["date"]) else "no")
    return {
        "launch_id": row["launch_id"],
        "status": "live" if ok_live else ("wayback" if has_wb == "yes" else "missing"),
        "http_code": code,
        "url_used": url_used,
        "has_wayback": has_wb,
        "note": "" if ok_live else f"live_failed code={code}",
    }


def is_landing(url: str) -> bool:
    """Heuristic: redirected to a homepage with no path."""
    return url.rstrip("/").count("/") <= 2


def main() -> int:
    rows: list[dict] = []
    for name in ["llm_launches.csv", "apple_launches.csv"]:
        with (LAUNCHES_DIR / name).open() as f:
            rows.extend(csv.DictReader(f))

    print(f"Checking {len(rows)} URLs...")
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(check_one, r): r for r in rows}
        for fut in as_completed(futs):
            results.append(fut.result())

    out = LAUNCHES_DIR / "url_status.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["launch_id", "status", "http_code",
                                          "url_used", "has_wayback", "note"])
        w.writeheader()
        w.writerows(results)

    by_status = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"Wrote {out.relative_to(ROOT)}")
    for k, v in sorted(by_status.items()):
        print(f"  {k:10s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
