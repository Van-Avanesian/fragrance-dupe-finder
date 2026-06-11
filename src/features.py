"""
Turns the cleaned DataFrame into a numeric feature matrix for cosine similarity.

Strategy
--------
1. Combine note pyramid into a weighted list: base notes × 3, middle × 2, top × 1.
   Base notes define the dry-down and linger longest — they contribute more to
   perceived similarity than top notes, which evaporate in ~15 minutes.
2. Binarize the weighted note list with MultiLabelBinarizer (500–800 dimensions).
3. One-hot encode the top main accord (scent family) — prevents cross-family false
   matches where two fragrances share a common note but smell nothing alike.
4. Concatenate and return as a float32 array alongside the fitted binarizer and
   accord encoder so we can transform a query fragrance at lookup time.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


def _weighted_notes(row: pd.Series) -> list[str]:
    """Repeat each note according to its pyramid weight."""
    return (
        list(row["base_notes"]) * 3
        + list(row["middle_notes"]) * 2
        + list(row["top_notes"]) * 1
    )


def build(df: pd.DataFrame) -> tuple[np.ndarray, MultiLabelBinarizer, LabelEncoder, list[str]]:
    """
    Returns
    -------
    matrix : float32 ndarray of shape (n_fragrances, n_features)
    note_binarizer : fitted MultiLabelBinarizer
    accord_encoder : fitted LabelEncoder
    accord_categories : list of accord category strings (for decoding)
    """
    # --- Notes (weighted) ---
    weighted = df.apply(_weighted_notes, axis=1).tolist()
    note_binarizer = MultiLabelBinarizer()
    note_matrix = note_binarizer.fit_transform(weighted).astype(np.float32)

    # --- Scent family (primary accord) ---
    # Use only the first (strongest) accord per fragrance as the family label.
    primary_accord = df["main_accords"].apply(
        lambda accords: accords[0] if accords else "unknown"
    )
    accord_encoder = LabelEncoder()
    accord_encoded = accord_encoder.fit_transform(primary_accord)
    accord_categories = list(accord_encoder.classes_)

    # One-hot encode the accord integer
    n_accords = len(accord_categories)
    accord_matrix = np.zeros((len(df), n_accords), dtype=np.float32)
    accord_matrix[np.arange(len(df)), accord_encoded] = 1.0

    # Upweight accords slightly so family mismatches are penalised
    accord_matrix *= 2.0

    matrix = np.hstack([note_matrix, accord_matrix])
    return matrix, note_binarizer, accord_encoder, accord_categories


def transform_query(
    top: list[str],
    middle: list[str],
    base: list[str],
    primary_accord: str,
    note_binarizer: MultiLabelBinarizer,
    accord_encoder: LabelEncoder,
    accord_categories: list[str],
) -> np.ndarray:
    """Transform a single fragrance's features into the same vector space."""
    weighted = base * 3 + middle * 2 + top * 1
    note_vec = note_binarizer.transform([weighted]).astype(np.float32)

    n_accords = len(accord_categories)
    accord_vec = np.zeros((1, n_accords), dtype=np.float32)
    if primary_accord in accord_categories:
        idx = accord_encoder.transform([primary_accord])[0]
        accord_vec[0, idx] = 2.0

    return np.hstack([note_vec, accord_vec])
