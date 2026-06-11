"""
Converts the Kaggle Fragrantica dataset (fra_cleaned.csv) to the format
expected by build_index.py.

Run:
    python3 -m src.ingest_kaggle
"""

import pandas as pd
from pathlib import Path

from src.clean import LUXURY_BRANDS, AFFORDABLE_BRANDS

INPUT = "data/raw/fra_cleaned.csv"
OUTPUT = "data/raw/fragrances_kaggle.csv"


def _parse_notes(raw) -> str:
    """Convert comma-separated notes string to pipe-delimited format."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    notes = [n.strip().lower() for n in raw.split(",") if n.strip()]
    return "|".join(notes)


def _infer_tier(brand_slug: str) -> str:
    # Brand slugs are like "creed", "tom-ford", "jean-paul-gaultier"
    normalized = brand_slug.replace("-", " ").strip().lower()
    if normalized in LUXURY_BRANDS:
        return "luxury"
    if normalized in AFFORDABLE_BRANDS:
        return "affordable"
    return "unknown"


def _parse_accords(row: pd.Series) -> str:
    accords = []
    for col in ["mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"]:
        val = row.get(col, "")
        if isinstance(val, str) and val.strip() and val.strip().lower() != "unknown":
            accords.append(val.strip().lower())
    return "|".join(accords)


def ingest(input_path: str = INPUT, output_path: str = OUTPUT) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=";", encoding="latin-1", on_bad_lines="skip")

    out = pd.DataFrame()
    out["name"] = df["Perfume"].str.replace("-", " ").str.strip()
    out["brand"] = df["Brand"].str.replace("-", " ").str.strip()
    out["price_tier"] = df["Brand"].apply(_infer_tier)
    out["url"] = df["url"].fillna("")
    out["concentration"] = "Unknown"
    out["top_notes"] = df["Top"].apply(_parse_notes)
    out["middle_notes"] = df["Middle"].apply(_parse_notes)
    out["base_notes"] = df["Base"].apply(_parse_notes)
    out["main_accords"] = df.apply(_parse_accords, axis=1)

    # Stats
    total = len(out)
    known_tier = out[out["price_tier"] != "unknown"]
    print(f"Total rows: {total}")
    print(f"  Luxury:     {(out['price_tier'] == 'luxury').sum()}")
    print(f"  Affordable: {(out['price_tier'] == 'affordable').sum()}")
    print(f"  Unknown:    {(out['price_tier'] == 'unknown').sum()} (excluded from dupe results)")
    print(f"  Has notes:  {out[out['top_notes'] != ''].shape[0]}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return out


if __name__ == "__main__":
    ingest()
