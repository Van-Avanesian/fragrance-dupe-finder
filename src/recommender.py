"""
Dupe lookup + Claude-powered sensory descriptions.

Usage
-----
from src.recommender import DupeFinder
finder = DupeFinder.load("data/processed")
results = finder.find_dupes("Aventus", n=5)
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import anthropic
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


class DupeFinder:
    def __init__(
        self,
        df: pd.DataFrame,
        similarity_matrix: np.ndarray,
        note_binarizer,
        accord_encoder,
        accord_categories: list[str],
    ):
        self.df = df.reset_index(drop=True)
        self.sim = similarity_matrix  # shape: (n, n)
        self.note_binarizer = note_binarizer
        self.accord_encoder = accord_encoder
        self.accord_categories = accord_categories
        # Build a name → row index lookup (lowercase for fuzzy matching)
        self._name_index: dict[str, int] = {
            f"{row['name'].lower()} by {row['brand'].lower()}": i
            for i, row in df.iterrows()
        }
        self._name_only: dict[str, int] = {
            row["name"].lower(): i for i, row in df.iterrows()
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, data_dir: str = "data/processed") -> None:
        out = Path(data_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(out / "fragrances.parquet", index=False)
        np.save(out / "similarity_matrix.npy", self.sim)
        with open(out / "encoders.pkl", "wb") as f:
            pickle.dump(
                {
                    "note_binarizer": self.note_binarizer,
                    "accord_encoder": self.accord_encoder,
                    "accord_categories": self.accord_categories,
                },
                f,
            )
        print(f"Saved index to {data_dir}/")

    @classmethod
    def load(cls, data_dir: str = "data/processed") -> "DupeFinder":
        p = Path(data_dir)
        df = pd.read_parquet(p / "fragrances.parquet")
        # Parquet may deserialise list columns as numpy arrays — normalise to Python lists
        for col in ("top_notes", "middle_notes", "base_notes", "main_accords"):
            df[col] = df[col].apply(
                lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else (
                    x.strip("[]").replace("'", "").split(", ") if isinstance(x, str) and x.strip("[]") else []
                )
            )
        sim = np.load(p / "similarity_matrix.npy")
        with open(p / "encoders.pkl", "rb") as f:
            enc = pickle.load(f)
        return cls(df, sim, enc["note_binarizer"], enc["accord_encoder"], enc["accord_categories"])

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _resolve_index(self, query: str) -> Optional[int]:
        q = query.strip().lower()
        if q in self._name_index:
            return self._name_index[q]
        if q in self._name_only:
            return self._name_only[q]
        # Partial match — return first hit
        for key, idx in self._name_only.items():
            if q in key:
                return idx
        return None

    def find_dupes(
        self,
        query: str,
        n: int = 5,
        affordable_only: bool = True,
        with_descriptions: bool = True,
    ) -> list[dict]:
        """
        Given a fragrance name, return the top-N most similar fragrances.

        Parameters
        ----------
        query : str — fragrance name (e.g. "Aventus" or "Aventus by Creed")
        n : int — number of results
        affordable_only : bool — filter to affordable tier dupes
        with_descriptions : bool — call Claude API for each result

        Returns
        -------
        List of dicts with keys: name, brand, price_tier, similarity,
        top_notes, middle_notes, base_notes, main_accords, description, url
        """
        idx = self._resolve_index(query)
        if idx is None:
            return []

        scores = self.sim[idx]  # similarity to all fragrances
        input_row = self.df.iloc[idx]

        # Build candidate list sorted by similarity (descending)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for candidate_idx, score in ranked:
            if candidate_idx == idx:
                continue  # skip the query itself
            row = self.df.iloc[candidate_idx]
            if affordable_only and row["price_tier"] != "affordable":
                continue
            results.append(
                {
                    "name": row["name"],
                    "brand": row["brand"],
                    "price_tier": row["price_tier"],
                    "similarity": round(float(score), 4),
                    "similarity_pct": f"{float(score) * 100:.1f}%",
                    "concentration": row.get("concentration", ""),
                    "top_notes": row["top_notes"],
                    "middle_notes": row["middle_notes"],
                    "base_notes": row["base_notes"],
                    "main_accords": row["main_accords"],
                    "url": row.get("url", ""),
                    "description": "",
                }
            )
            if len(results) >= n:
                break

        if with_descriptions and results:
            for rec in results:
                rec["description"] = _generate_description(input_row, rec)

        return results

    def all_fragrance_names(self) -> list[str]:
        return sorted(
            f"{r['name']} by {r['brand']}"
            for _, r in self.df.iterrows()
        )


# ------------------------------------------------------------------
# Claude API — sensory comparison
# ------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a master perfumer with deep sensory expertise. "
    "When given two fragrances and their notes, write exactly two sentences: "
    "one explaining what olfactory character they share, one describing any key difference. "
    "Be vivid and specific — name actual notes. Never use filler phrases like 'both fragrances' "
    "at the start. Keep the total under 60 words."
)


def _format_notes(row) -> str:
    def _get(key):
        val = row[key] if isinstance(row, dict) else row.get(key, [])
        return ", ".join(val) if isinstance(val, list) else str(val)

    parts = []
    if _get("top_notes"):
        parts.append(f"top: {_get('top_notes')}")
    if _get("middle_notes"):
        parts.append(f"heart: {_get('middle_notes')}")
    if _get("base_notes"):
        parts.append(f"base: {_get('base_notes')}")
    return "; ".join(parts) or "notes unknown"


def _generate_description(input_row: pd.Series, dupe: dict) -> str:
    try:
        client = _get_client()
        user_msg = (
            f"Original: {input_row['name']} by {input_row['brand']} "
            f"({_format_notes(input_row)})\n"
            f"Dupe: {dupe['name']} by {dupe['brand']} "
            f"({_format_notes(dupe)})"
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=120,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        return f"(Description unavailable: {exc})"
