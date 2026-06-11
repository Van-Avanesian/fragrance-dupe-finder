"""
Full pipeline: raw CSV → clean → featurize → cosine similarity matrix → save.

Run after scraping:
    python build_index.py

Or point to a different raw file:
    python build_index.py --raw data/raw/my_file.csv --out data/processed
"""

import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.clean import clean
from src.features import build
from src.recommender import DupeFinder


def main(raw_path: str, out_dir: str) -> None:
    print("=== Step 1: Clean ===")
    df = clean(raw_path)
    print(f"Clean dataset: {len(df)} fragrances")
    print(f"  Luxury: {(df['price_tier'] == 'luxury').sum()}")
    print(f"  Affordable: {(df['price_tier'] == 'affordable').sum()}")

    print("\n=== Step 2: Feature engineering ===")
    matrix, note_binarizer, accord_encoder, accord_categories = build(df)
    print(f"Feature matrix: {matrix.shape}")
    print(f"  Note vocab size: {len(note_binarizer.classes_)}")
    print(f"  Scent families: {accord_categories}")

    print("\n=== Step 3: Cosine similarity ===")
    sim = cosine_similarity(matrix).astype(np.float32)
    print(f"Similarity matrix: {sim.shape} — {sim.nbytes / 1e6:.1f} MB")

    print("\n=== Step 4: Save ===")
    finder = DupeFinder(df, sim, note_binarizer, accord_encoder, accord_categories)
    finder.save(out_dir)

    # Quick sanity check against known dupe pairs
    print("\n=== Sanity check (known dupe pairs) ===")
    known_pairs = [
        ("Aventus", "Club de Nuit Intense Man"),
        ("Sauvage", "Asad"),
        ("Baccarat Rouge 540", "Fakhar"),
    ]
    for luxury_name, affordable_name in known_pairs:
        results = finder.find_dupes(luxury_name, n=10, affordable_only=False, with_descriptions=False)
        match = next((r for r in results if affordable_name.lower() in r["name"].lower()), None)
        if match:
            print(f"  ✓ {luxury_name} → {match['name']} ({match['similarity_pct']} similarity)")
        else:
            print(f"  ? {luxury_name}: '{affordable_name}' not in top-10 (may not be in dataset)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/fragrances_raw.csv")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    main(args.raw, args.out)
