"""
Extract structured features from each scraped launch announcement using
the Claude API.

Reads:  data/raw/<launch_id>.json (output of scrape.py)
Writes: data/processed/features.jsonl  (one row per launch, append-safe)
        data/processed/features.parquet (rebuilt from jsonl on each run)

Design notes
------------
- One Claude call per launch. We use **tool use** to force a strict JSON
  schema, so every row has the same fields.
- The system prompt is **cached** (cache_control on the system block) so
  we pay full input cost once and ~10% on subsequent calls.
- Resumable: if a launch_id already exists in features.jsonl we skip it
  unless --force is passed.
- Model: claude-sonnet-4-6 by default (good cost/quality for extraction).
  Override with --model. Use claude-opus-4-7 for spot-check quality runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

ROOT = Path(__file__).resolve().parents[1]
LAUNCHES_DIR = ROOT / "data" / "launches"
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
JSONL_PATH = PROC_DIR / "features.jsonl"
PARQUET_PATH = PROC_DIR / "features.parquet"

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TEXT_CHARS = 30_000  # truncate very long posts (rare); keeps token cost predictable

SYSTEM_PROMPT = """\
You are an analyst extracting structured features from product launch
announcements. Two corpora are in play:

1. LLM/AI model launches (OpenAI, Anthropic, Google, Meta, Mistral, etc.).
2. Apple hardware launches (iPhone, iPad, Mac, iPod, Apple Silicon).

The research question is: are recent LLM launches starting to look like
recent iPhone launches? That is, are they shifting from category-creating
to incremental ("better/faster/sleeker/cheaper") rhetoric?

Your job is to read the announcement text and return a strict JSON object
via the `record_features` tool. Be a careful, literal reader: only count
phrases that ACTUALLY appear in the text. When in doubt, prefer fewer
items over speculative ones. Quote phrases exactly as they appear.

Definitions you must apply:

- novelty_signals: phrases claiming something genuinely NEW or first-of-kind.
  Examples: "first ever", "introducing", "a new category", "never before
  possible", "redefines", "reinvents", "breakthrough", "unprecedented".
  COUNT only when the phrase is applied to the product itself, not to
  general industry trends.

- incremental_signals: phrases claiming improvement over a prior version
  WITHOUT claiming a new category. Examples: "faster", "improved", "more
  efficient", "X% better", "longer battery life", "higher accuracy".

- superlatives: "best", "smartest", "most powerful", "fastest ever",
  "thinnest", "most advanced". Standalone adjectives at the top of the
  scale (no comparison target needed).

- comparatives: explicit comparisons ("X% faster than", "smarter than",
  "compared to", "vs"). Each comparative has a target.

- comparison_targets: for each explicit comparison, classify the target as:
  - "internal": prior generation of the same product line (e.g., "the
    previous iPhone", "Claude 3.5 Sonnet" when announcing Claude 4)
  - "external": a competitor or external benchmark target (e.g.,
    "GPT-4", "Snapdragon", "the leading Android phone")
  - "industry": vague references to "the industry", "the competition"

- benchmarks_mentioned: named tests/scores. For LLMs: MMLU, GPQA, HumanEval,
  SWE-bench, AIME, ARC, etc. For Apple: Geekbench, battery hours, "X hours
  of video playback", camera megapixels, display nits, etc.

- new_capability_claims: list (max 8) of specific things the announcement
  claims this product can do that the previous version could not. Quote
  the claim succinctly. Empty list if nothing qualifies.

- adjectives: list (max 15) of the most prominent adjectives describing the
  product itself. Lowercase, lemmatized (e.g., "fastest" not "the fastest").

- model_type (LLMs only; "" for Apple): one of
  ["thinking", "flash/small", "image-gen", "video-gen", "audio",
   "multimodal", "code-focused", "science-focused", "agent",
   "major-version", "incremental-version", "open-weights", "other"]

- product_category (Apple only; "" for LLMs): one of
  ["iPhone", "iPad", "Mac-laptop", "Mac-desktop", "iPod", "chip", "other"]

- quantification_density: count the integers, decimals, and percentages
  in the text, then divide by total word count, multiply by 100. Round to
  one decimal.

- use_case_vs_spec_ratio: estimate the share of text spent describing what
  USERS DO with the product (workflows, scenarios, examples) vs. what the
  product IS internally (chip names, specs, benchmark numbers).
  Return a decimal in [0, 1] where 1.0 = entirely use-case framed,
  0.0 = entirely spec-sheet framed. 0.5 = balanced.

- launch_summary: one sentence (max 30 words) describing what the launch is.

If a field doesn't apply, return an empty list/string/0 — never null.
"""

EXTRACTION_TOOL = {
    "name": "record_features",
    "description": "Record extracted features from this launch announcement.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "launch_summary", "model_type", "product_category",
            "benchmarks_mentioned", "use_cases_mentioned", "adjectives",
            "novelty_signals", "novelty_signal_count",
            "incremental_signals", "incremental_signal_count",
            "superlatives", "superlative_count",
            "comparatives", "comparative_count",
            "comparison_targets",
            "new_capability_claims",
            "quantification_density", "use_case_vs_spec_ratio",
        ],
        "properties": {
            "launch_summary": {"type": "string"},
            "model_type": {"type": "string"},
            "product_category": {"type": "string"},
            "benchmarks_mentioned": {
                "type": "array", "items": {"type": "string"}, "maxItems": 30
            },
            "use_cases_mentioned": {
                "type": "array", "items": {"type": "string"}, "maxItems": 20
            },
            "adjectives": {
                "type": "array", "items": {"type": "string"}, "maxItems": 15
            },
            "novelty_signals": {
                "type": "array", "items": {"type": "string"}, "maxItems": 30,
                "description": "Exact phrases from the text"
            },
            "novelty_signal_count": {"type": "integer", "minimum": 0},
            "incremental_signals": {
                "type": "array", "items": {"type": "string"}, "maxItems": 30
            },
            "incremental_signal_count": {"type": "integer", "minimum": 0},
            "superlatives": {
                "type": "array", "items": {"type": "string"}, "maxItems": 30
            },
            "superlative_count": {"type": "integer", "minimum": 0},
            "comparatives": {
                "type": "array", "items": {"type": "string"}, "maxItems": 30
            },
            "comparative_count": {"type": "integer", "minimum": 0},
            "comparison_targets": {
                "type": "array",
                "maxItems": 30,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target", "kind"],
                    "properties": {
                        "target": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["internal", "external", "industry"]
                        }
                    }
                }
            },
            "new_capability_claims": {
                "type": "array", "items": {"type": "string"}, "maxItems": 8
            },
            "quantification_density": {"type": "number", "minimum": 0},
            "use_case_vs_spec_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
}


def load_metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for name, domain in [("llm_launches.csv", "llm"), ("apple_launches.csv", "apple")]:
        with (LAUNCHES_DIR / name).open() as f:
            for r in csv.DictReader(f):
                r["domain"] = domain
                meta[r["launch_id"]] = r
    return meta


def already_done() -> set[str]:
    if not JSONL_PATH.exists():
        return set()
    done = set()
    with JSONL_PATH.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["launch_id"])
            except Exception:
                continue
    return done


def extract_one(client: Anthropic, model: str, launch_id: str,
                meta: dict, raw: dict) -> dict:
    text = raw.get("text", "")[:MAX_TEXT_CHARS]
    user_msg = (
        f"LAUNCH_ID: {launch_id}\n"
        f"DOMAIN: {meta['domain']}\n"
        f"COMPANY: {meta['company']}\n"
        f"PRODUCT: {meta['product_name']}\n"
        f"DATE: {meta['date']}\n"
        f"TITLE: {raw.get('title', '')}\n\n"
        f"--- ANNOUNCEMENT TEXT ---\n{text}"
    )

    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_features"},
        messages=[{"role": "user", "content": user_msg}],
    )

    features = None
    for block in resp.content:
        if block.type == "tool_use" and block.name == "record_features":
            features = block.input
            break
    if features is None:
        raise RuntimeError(f"No tool_use block returned for {launch_id}")

    return {
        "launch_id": launch_id,
        "domain": meta["domain"],
        "company": meta["company"],
        "product_family": meta.get("product_family", ""),
        "product_name": meta["product_name"],
        "date": meta["date"],
        "url_used": raw.get("url_used", ""),
        "source": raw.get("source", ""),
        "word_count": raw.get("word_count", 0),
        **features,
        "_model": model,
        "_input_tokens": resp.usage.input_tokens,
        "_cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
        "_output_tokens": resp.usage.output_tokens,
    }


def write_parquet():
    if not JSONL_PATH.exists():
        return
    rows = [json.loads(l) for l in JSONL_PATH.open()]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"  wrote {PARQUET_PATH.relative_to(ROOT)} ({len(df)} rows)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--only", help="Comma-separated launch_ids")
    p.add_argument("--domain", choices=["llm", "apple"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Stop after N extractions")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    done = set() if args.force else already_done()

    raw_files = sorted(RAW_DIR.glob("*.json"))
    todo: list[tuple[str, dict, dict]] = []
    for path in raw_files:
        launch_id = path.stem
        if launch_id in done:
            continue
        if launch_id not in meta:
            continue
        if args.domain and meta[launch_id]["domain"] != args.domain:
            continue
        if args.only and launch_id not in set(args.only.split(",")):
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        if raw.get("status") != "ok":
            continue
        todo.append((launch_id, meta[launch_id], raw))
        if args.limit and len(todo) >= args.limit:
            break

    print(f"Extracting features for {len(todo)} launches "
          f"(model={args.model}, already done={len(done)})")

    client = Anthropic()
    ok = failed = 0
    with JSONL_PATH.open("a") as out:
        for i, (launch_id, m, raw) in enumerate(todo, 1):
            try:
                row = extract_one(client, args.model, launch_id, m, raw)
                out.write(json.dumps(row) + "\n")
                out.flush()
                ok += 1
                cache = row.get("_cache_read_tokens", 0)
                print(f"  [{i}/{len(todo)}] {launch_id:40s} "
                      f"in={row['_input_tokens']} cache={cache} "
                      f"out={row['_output_tokens']}")
            except Exception as e:
                failed += 1
                print(f"  FAIL {launch_id}: {e}", file=sys.stderr)
                time.sleep(2)

    write_parquet()
    print(f"\nDone. ok={ok}  failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
