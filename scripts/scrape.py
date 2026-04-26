"""
Scrape launch announcement pages for both corpora.

For each row in data/launches/{llm,apple}_launches.csv:
  1. Try the canonical URL.
  2. On 404 / 5xx / redirect to homepage / non-success, fall back to Wayback
     Machine (snapshot closest to the launch date).
  3. Extract main article text via trafilatura, store under data/raw/.

Output per launch:
  data/raw/<launch_id>.json
    {
      "launch_id": "...",
      "url_attempted": "...",
      "url_used": "...",          # actual URL that succeeded (may be archive.org)
      "source": "live" | "wayback",
      "fetched_at": "ISO-8601",
      "status": "ok" | "failed",
      "title": "...",
      "text": "...",              # main body, plain text
      "word_count": int,
      "error": "..."              # only present when status=failed
    }
  data/raw/html/<launch_id>.html  (raw HTML, for re-extraction later)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura

ROOT = Path(__file__).resolve().parents[1]
LAUNCHES_DIR = ROOT / "data" / "launches"
RAW_DIR = ROOT / "data" / "raw"
HTML_DIR = RAW_DIR / "html"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
TIMEOUT = 30


def fetch_live(url: str) -> tuple[int, str, str]:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    return r.status_code, r.text, r.url


def fetch_wayback(url: str, target_date: str) -> tuple[str, str] | None:
    """Find the snapshot closest to target_date (YYYY-MM-DD) and fetch it."""
    ts = target_date.replace("-", "")
    api = f"https://archive.org/wayback/available?url={url}&timestamp={ts}"
    try:
        r = requests.get(api, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        snap = r.json().get("archived_snapshots", {}).get("closest")
        if not snap or not snap.get("available"):
            return None
        snap_url = snap["url"]
        if snap_url.startswith("http://"):
            snap_url = "https://" + snap_url[len("http://"):]
        s = requests.get(snap_url, headers=HEADERS, timeout=TIMEOUT)
        s.raise_for_status()
        return snap_url, s.text
    except Exception:
        return None


def extract_text(html: str, url: str) -> tuple[str, str, str]:
    extracted = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=False,
        favor_recall=True, output_format="txt"
    ) or ""
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "") or ""
    # trafilatura returns ISO-ish date when found; "" otherwise
    pub = (meta.date if meta and meta.date else "") or ""
    return title, extracted.strip(), pub


def looks_like_failure(text: str, url_used: str) -> bool:
    """Heuristic: short body or redirected to a generic landing page."""
    if len(text) < 400:
        return True
    if "/newsroom/" not in url_used and "/news/" not in url_used \
       and "/blog/" not in url_used and "/index/" not in url_used \
       and "/pr/library/" not in url_used and "archive.org" not in url_used:
        # Apple homepage / OpenAI homepage / etc. would land here
        if url_used.rstrip("/").count("/") <= 2:
            return True
    return False


def scrape_one(row: dict) -> dict:
    launch_id = row["launch_id"]
    url = row["canonical_url"].strip()
    date = row["date"]
    out: dict = {
        "launch_id": launch_id,
        "url_attempted": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    html = ""
    url_used = ""
    source = ""

    # Try live first
    if url:
        try:
            code, html, url_used = fetch_live(url)
            if code == 200:
                source = "live"
        except Exception as e:
            out["live_error"] = str(e)

    # Extract & decide if we need Wayback
    title, text, pub_date = ("", "", "")
    if html:
        title, text, pub_date = extract_text(html, url_used or url)
    if not text or looks_like_failure(text, url_used or url):
        wb = fetch_wayback(url, date) if url else None
        if wb:
            url_used, html = wb
            source = "wayback"
            title, text, pub_date = extract_text(html, url_used)

    if not text:
        out.update(status="failed", error="no_text_extracted",
                   url_used=url_used, source=source or "none")
        return out

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    (HTML_DIR / f"{launch_id}.html").write_text(html, encoding="utf-8")

    out.update(
        status="ok",
        url_used=url_used,
        source=source,
        title=title,
        text=text,
        word_count=len(text.split()),
        published_date=pub_date,           # from page metadata; "" if missing
        csv_date=date,                     # the date in our master CSV
    )
    return out


def load_rows() -> list[dict]:
    rows = []
    for name, domain in [("llm_launches.csv", "llm"), ("apple_launches.csv", "apple")]:
        path = LAUNCHES_DIR / name
        with path.open() as f:
            for r in csv.DictReader(f):
                r["domain"] = domain
                rows.append(r)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="Comma-separated launch_ids to scrape (default: all)")
    p.add_argument("--domain", choices=["llm", "apple"], help="Limit to one corpus")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true", help="Re-scrape even if output exists")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if args.domain:
        rows = [r for r in rows if r["domain"] == args.domain]
    if args.only:
        wanted = set(args.only.split(","))
        rows = [r for r in rows if r["launch_id"] in wanted]

    todo = []
    for r in rows:
        out_path = RAW_DIR / f"{r['launch_id']}.json"
        if out_path.exists() and not args.force:
            continue
        todo.append(r)

    print(f"Scraping {len(todo)} / {len(rows)} launches "
          f"(skipping {len(rows) - len(todo)} already done)")

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scrape_one, r): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {"launch_id": r["launch_id"], "status": "failed", "error": str(e)}
            out_path = RAW_DIR / f"{result['launch_id']}.json"
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            if result.get("status") == "ok":
                ok += 1
                print(f"  ok    {result['launch_id']:40s} "
                      f"[{result['source']}] {result['word_count']}w")
            else:
                failed += 1
                print(f"  FAIL  {result['launch_id']:40s} "
                      f"{result.get('error', '?')}", file=sys.stderr)
            time.sleep(0.1)  # gentle rate-limit

    print(f"\nDone. ok={ok}  failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
