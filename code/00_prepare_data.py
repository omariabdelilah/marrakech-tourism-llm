"""
00_prepare_data.py
------------------
Produces the released, de-identified dataset from the raw scraped file.

This script is included for transparency only. It cannot be re-run from the
contents of this repository, because its input (the raw scrape, which contains
reviewer names and full review texts) is not redistributed here. Review texts
are not redistributed in accordance with TripAdvisor's terms of service.

Input : LLM_FULL_ANALYSIS.csv       (raw, not released)
Output: data/llm_extracted_signals.csv  (released)

Dropped columns: username, title, text, contributions.
Added columns  : review_id, stars, gt, year, era.
"""

import re
import pandas as pd

RAW = "LLM_FULL_ANALYSIS.csv"
OUT = "data/llm_extracted_signals.csv"

DROP = ["username", "title", "text", "contributions"]


def parse_rating(v):
    """'4 of 5 bubbles' -> 4.0"""
    if pd.isna(v):
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"(\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else float("nan")


def stars_to_label(s):
    """Three-class mapping used as ground truth throughout the paper."""
    if pd.isna(s):
        return None
    if s >= 4:
        return "Positive"
    if s == 3:
        return "Neutral"
    return "Negative"


def era(year):
    if year <= 2019:
        return "Pre-COVID"
    if year == 2020:
        return "COVID"
    return "Post-COVID"


def main():
    df = pd.read_csv(RAW)

    df["stars"] = df["rating"].apply(parse_rating)
    df["gt"] = df["stars"].apply(stars_to_label)

    written = df["written_date"].str.replace("Written ", "", regex=False)
    df["year"] = pd.to_datetime(written, errors="coerce").dt.year
    df["era"] = df["year"].apply(era)

    df = df.drop(columns=[c for c in DROP if c in df.columns])
    df.insert(0, "review_id", range(1, len(df) + 1))

    cols = [
        "review_id", "place", "stars", "gt", "year", "era",
        "trip_date", "trip_type", "written_date",
        "llm_sentiment", "llm_score", "llm_emotion",
        "llm_safety", "llm_pricing", "llm_service",
        "llm_cleanliness", "llm_atmosphere",
        "llm_problems",
        "llm_price_perception", "llm_recommendation", "llm_revisit",
        "llm_keywords",
    ]
    df = df[cols]

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(df):,} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
