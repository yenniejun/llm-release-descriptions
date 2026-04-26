# LLM Release Descriptions vs. Apple Hardware Launches

A comparative corpus analysis of how AI labs and Apple talk about new releases.

## Research question

**Are modern LLM launches starting to look like recent iPhone launches?**

Both spaces appear to have moved from category-creating launches (iPhone 2007,
ChatGPT 2022) to incremental "better/faster/sleeker" launches (camera bumps,
benchmark bumps). This project tests that hypothesis with text data.

The Apple corpus spans ~20 years (2005-2025) so the *full* arc is captured —
early iPod / iPhone launches genuinely introduced new categories, while recent
launches are dominated by spec bumps. That gives us a baseline against which
to measure where modern LLM launches sit on the curve.

## Data

Two parallel corpora of official launch announcement text:

- **LLM launches** (`data/launches/llm_launches.csv`): ~80-100 launches from
  OpenAI, Anthropic, Google/DeepMind, Meta, Mistral, xAI, DeepSeek, Qwen,
  Microsoft, Cohere, Amazon, NVIDIA, Stability, HuggingFace (2022-2026).
- **Apple launches** (`data/launches/apple_launches.csv`): ~120+ launches across
  iPod, iPhone, iPad, Mac (MacBook, MacBook Pro, MacBook Air, iMac, Mac Mini,
  Mac Studio, Mac Pro), and Apple Silicon (M1-M5) (2005-2025).

For each launch we collect the **full announcement text** (Wayback Machine
fallback for older / dead URLs) and extract structured features.

## Extracted features

Done via the Claude API (see `scripts/extract_features.py`). For each post:

- **date** (YYYY-MM-DD), **company**, **product_name**, **domain** (llm/apple)
- **model_type** for LLMs (thinking, flash, image-gen, multimodal, science,
  major-version vs incremental); **product_category** for Apple
- **benchmarks_mentioned** — list of benchmarks/tests cited (MMLU, GPQA, etc.
  for LLMs; Geekbench, battery hours, camera specs for Apple)
- **use_cases_mentioned** — list of use cases / workflows highlighted
- **adjectives** — top descriptive adjectives applied to the product
- **novelty_signals** — count of "first ever", "introducing", "new category",
  "never before possible" type phrases
- **incremental_signals** — count of "faster", "improved", "X% better",
  "more efficient" type phrases
- **superlatives** — count of "smartest ever", "most powerful", etc.
- **comparatives** — count of "smarter than", "X% faster than", etc.
- **comparison_targets** — what the product is compared to, classified as
  `internal` (own previous gen) vs `external` (competitor)
- **quantification_density** — numbers / percentages per 100 words
- **use_case_vs_spec_ratio** — words about user activity vs. words about
  product specs/internals
- **new_capability_claims** — qualitative list of "things this can do that
  the previous version couldn't"

## Pipeline

1. `scripts/discover_urls.py` — verifies + fills in canonical URLs for each
   launch in the master CSVs (Wayback Machine fallback).
2. `scripts/scrape.py` — fetches each URL, extracts main text, stores under
   `data/raw/<launch_id>.json` (alongside the HTML).
3. `scripts/extract_features.py` — calls Claude API per post, writes structured
   features to `data/processed/features.parquet`.
4. `notebooks/` — analysis (timelines, comparisons, charts) lives here later.

## Storage layout

```
data/
  launches/
    llm_launches.csv      # master list (id, company, product, date, url)
    apple_launches.csv
  raw/
    <launch_id>.json      # {url, fetched_at, text, html_path, source}
    html/<launch_id>.html
  processed/
    features.parquet      # one row per launch, all extracted features
scripts/
  discover_urls.py
  scrape.py
  extract_features.py
notebooks/
```

## Running

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python scripts/discover_urls.py     # verify/fill URLs
python scripts/scrape.py            # fetch all posts
python scripts/extract_features.py  # extract features via Claude
```
