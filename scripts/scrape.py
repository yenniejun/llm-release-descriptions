"""
Scrape launch announcement pages for both corpora.

For each row in data/launches/{llm,apple}_launches.csv:
  1. Try the canonical URL with realistic browser headers.
  2. On failure (non-200, too-short body, redirect to homepage), fall back to
     the Wayback Machine snapshot closest to the launch date.
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
      "published_date": "...",    # from page metadata; "" if missing
      "csv_date": "...",          # the date in our master CSV
      "error": "..."              # only present when status=failed
    }
  data/raw/html/<launch_id>.html  (raw HTML, for re-extraction later)

Tips for running locally
------------------------
Most sites serve fine from a residential IP but block cloud/datacenter IPs.
Run from your own machine:

    pip install -r requirements.txt
    python scripts/scrape.py --workers 4 --delay 1.5

If a site is still blocking you, Wayback Machine is the reliable fallback.
The script will try it automatically. You can also force wayback-only mode:

    python scripts/scrape.py --wayback-only --domain apple
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura

ROOT = Path(__file__).resolve().parents[1]
LAUNCHES_DIR = ROOT / "data" / "launches"
RAW_DIR = ROOT / "data" / "raw"
HTML_DIR = RAW_DIR / "html"

TIMEOUT = 30

# Full Chrome-on-Mac browser header set — gets past most basic bot detection
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-CH-UA": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"macOS"',
    "Cache-Control": "max-age=0",
}

# Per-domain polite delay in seconds (in addition to --delay baseline)
DOMAIN_DELAYS: dict[str, float] = {
    "openai.com": 2.0,
    "anthropic.com": 1.5,
    "apple.com": 1.5,
    "archive.org": 2.0,
}


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    return s


SESSION = get_session()


def domain_delay(url: str, base: float) -> None:
    host = urlparse(url).hostname or ""
    extra = max(DOMAIN_DELAYS.get(h, 0.0) for h in DOMAIN_DELAYS if h in host)
    jitter = random.uniform(0.1, 0.5)
    time.sleep(base + extra + jitter)


def fetch_live(url: str) -> tuple[int, str, str]:
    r = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
    return r.status_code, r.text, r.url


def fetch_wayback(url: str, target_date: str) -> tuple[str, str] | None:
    """Find the Wayback snapshot closest to target_date (YYYY-MM-DD) and fetch it."""
    ts = target_date.replace("-", "")

    # Try CDX API first (more reliable than availability API)
    cdx = (
        f"https://web.archive.org/cdx/search/cdx"
        f"?url={url}&output=json&limit=1&fl=timestamp,original,statuscode"
        f"&filter=statuscode:200&closest={ts}&sort=closest"
    )
    snap_url = None
    try:
        r = SESSION.get(cdx, timeout=TIMEOUT)
        if r.status_code == 200:
            rows = r.json()
            if rows and len(rows) > 1:  # first row is header
                ts_found = rows[1][0]
                snap_url = f"https://web.archive.org/web/{ts_found}/{url}"
    except Exception:
        pass

    # Fall back to availability API
    if not snap_url:
        try:
            api = f"https://archive.org/wayback/available?url={url}&timestamp={ts}"
            r = SESSION.get(api, timeout=TIMEOUT)
            r.raise_for_status()
            snap = r.json().get("archived_snapshots", {}).get("closest")
            if snap and snap.get("available"):
                snap_url = snap["url"]
        except Exception:
            pass

    if not snap_url:
        return None

    if snap_url.startswith("http://"):
        snap_url = "https://" + snap_url[len("http://"):]

    try:
        s = SESSION.get(snap_url, timeout=TIMEOUT)
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
    pub = (meta.date if meta and meta.date else "") or ""
    return title, extracted.strip(), pub


def looks_like_failure(text: str, url_used: str) -> bool:
    """Heuristic: short body or redirected to a generic landing page."""
    if len(text) < 400:
        return True
    path_parts = urlparse(url_used).path.strip("/").split("/")
    meaningful_path = len(path_parts) >= 2 and any(len(p) > 4 for p in path_parts)
    if not meaningful_path and "archive.org" not in url_used:
        return True
    return False


def scrape_one(row: dict, base_delay: float, wayback_only: bool) -> dict:
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

    if not wayback_only and url:
        domain_delay(url, base_delay)
        try:
            code, html, url_used = fetch_live(url)
            if code == 200:
                source = "live"
        except Exception as e:
            out["live_error"] = str(e)

    title, text, pub_date = ("", "", "")
    if html:
        title, text, pub_date = extract_text(html, url_used or url)

    if not text or looks_like_failure(text, url_used or url):
        if url:
            domain_delay("archive.org", base_delay)
            wb = fetch_wayback(url, date)
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
        published_date=pub_date,
        csv_date=date,
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
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers (keep low to avoid rate limits; default 4)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Base seconds between requests per worker (default 1.0)")
    p.add_argument("--force", action="store_true", help="Re-scrape even if output exists")
    p.add_argument("--wayback-only", action="store_true",
                   help="Skip live fetch; go straight to Wayback Machine")
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
          f"(skipping {len(rows) - len(todo)} already done, "
          f"workers={args.workers}, delay={args.delay}s)")

    ok = failed = 0
    # Use workers=1 for sequential scraping when delay is large (kinder to servers)
    workers = min(args.workers, max(1, int(1.0 / max(args.delay, 0.1))))
    workers = max(workers, args.workers)  # but honour explicit --workers

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(scrape_one, r, args.delay, args.wayback_only): r
            for r in todo
        }
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
                print(f"  ok    {result['launch_id']:42s} "
                      f"[{result['source']:7s}] {result['word_count']}w")
            else:
                failed += 1
                print(f"  FAIL  {result['launch_id']:42s} "
                      f"{result.get('error', '?')}", file=sys.stderr)

    print(f"\nDone. ok={ok}  failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
