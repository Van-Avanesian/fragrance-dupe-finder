"""
Cleans and normalizes raw Fragrantica CSV into a analysis-ready DataFrame.
"""

from typing import Optional

import pandas as pd

# Canonical note name mapping — handles Fragrantica's inconsistencies
NOTE_ALIASES: dict[str, str] = {
    "oak moss": "oakmoss",
    "oak-moss": "oakmoss",
    "musk white": "white musk",
    "white musks": "white musk",
    "musks": "musk",
    "ambergris": "ambrette",
    "amber": "ambergris",  # true amber is ambergris-like in perfumery
    "iso e super": "iso e super",
    "labdanum": "labdanum",
    "lavendel": "lavender",
    "lavande": "lavender",
    "iris root": "orris root",
    "orris": "orris root",
    "vetiver": "vetiver",
    "vetyver": "vetiver",
    "sandalwood": "sandalwood",
    "sandal": "sandalwood",
    "oud": "oud",
    "aoud": "oud",
    "bergamote": "bergamot",
    "lemon zest": "lemon",
    "citron": "lemon",
    "pink peppercorn": "pink pepper",
    "rose absolute": "rose",
    "rose de mai": "rose",
    "patchouli leaf": "patchouli",
    "dark patchouli": "patchouli",
    "cistus": "labdanum",
    "elemi": "elemi resin",
    "benzoin": "benzoin resin",
    "styrax": "benzoin resin",
    "tonka": "tonka bean",
    "tonka beans": "tonka bean",
    "cedar": "cedarwood",
    "cedrus": "cedarwood",
    "atlas cedarwood": "cedarwood",
    "virginia cedarwood": "cedarwood",
    "jasmine absolute": "jasmine",
    "jasminum": "jasmine",
    "ylang ylang": "ylang-ylang",
    "ylang": "ylang-ylang",
    "neroli": "neroli",
    "orange blossom": "neroli",
    "petitgrain": "petitgrain",
    "petit grain": "petitgrain",
    "guaiac wood": "guaiac wood",
    "gaiac wood": "guaiac wood",
    "black pepper": "pepper",
    "white pepper": "pepper",
    "agarwood": "oud",
    "frankincense": "frankincense",
    "olibanum": "frankincense",
    "incense": "frankincense",
    "galbanum": "galbanum",
    "angelica root": "angelica",
    "cardamom": "cardamom",
    "cardamon": "cardamom",
    "coriander": "coriander",
    "ginger": "ginger",
    "clary sage": "sage",
    "common sage": "sage",
    "cistus labdanum": "labdanum",
    "benzyl benzoate": "benzoin resin",
    "coumarin": "coumarin",
    "heliotrope": "heliotrope",
    "violet leaf": "violet",
    "violets": "violet",
}

# Concentration normalization
CONCENTRATION_MAP = {
    "eau de parfum": "EDP",
    "edp": "EDP",
    "eau de toilette": "EDT",
    "edt": "EDT",
    "parfum": "Parfum",
    "extrait": "Parfum",
    "extrait de parfum": "Parfum",
    "cologne": "EDC",
    "eau de cologne": "EDC",
    "edc": "EDC",
}

# Brand → price tier overrides (in case the scraper's tier is wrong)
LUXURY_BRANDS = {
    # Ultra niche / $200+
    "creed", "xerjoff", "amouage", "frederic malle", "by kilian",
    "initio parfums prives", "clive christian", "serge lutens", "roja dove",
    "memo paris", "tiziana terenzi", "bond no 9", "boadicea the victorious",
    "m micallef", "mancera", "montale", "penhaligon s", "penhaligons",
    # Designer luxury / $100–200
    "tom ford", "maison margiela", "parfums de marly", "diptyque",
    "jo malone london", "chanel", "dior", "guerlain", "hermes", "loewe",
    "acqua di parma", "bvlgari", "yves saint laurent", "giorgio armani",
    "givenchy", "carolina herrera", "kenzo", "jean paul gaultier", "lancome",
    "donna karan", "estee lauder", "dolce gabbana", "mugler", "marc jacobs",
    "gucci", "issey miyake", "caron", "burberry", "nina ricci", "molinard",
    "fragonard", "l occitane en provence", "swiss arabian", "salvatore ferragamo",
    "maison francis kurkdjian", "nasomatto",
}
AFFORDABLE_BRANDS = {
    # Middle East affordable niche
    "armaf", "lattafa perfumes", "al haramain perfumes", "rasasi",
    "fragrance world", "maison alhambra", "afnan perfumes", "al rehab",
    "ajmal", "wadi al khaleej", "dua fragrances",
    # Western affordable / drugstore
    "alexandre j", "zara", "paco rabanne", "calvin klein", "hugo boss",
    "davidoff", "azzaro", "nautica", "adidas", "designer imposters",
    "avon", "oriflame", "jeanne arthes", "faberlic", "la rive",
    "bath body works", "victoria s secret", "yves rocher", "brocard",
    "antonio banderas", "o boticario", "natura", "eudora",
}


def _normalize_note(note: str) -> str:
    n = note.strip().lower()
    return NOTE_ALIASES.get(n, n)


def _parse_pipe_list(value) -> list[str]:
    """Parse 'note1|note2|note3' string into a cleaned list."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [_normalize_note(n) for n in value.split("|") if n.strip()]


def _normalize_concentration(raw: str) -> str:
    if not isinstance(raw, str):
        return "Unknown"
    key = raw.strip().lower()
    return CONCENTRATION_MAP.get(key, raw.strip() or "Unknown")


def _infer_tier(brand: str) -> Optional[str]:
    b = brand.strip().lower()
    if b in LUXURY_BRANDS:
        return "luxury"
    if b in AFFORDABLE_BRANDS:
        return "affordable"
    return None


def clean(raw_path: str = "data/raw/fragrances_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # Parse pipe-delimited note lists into Python lists
    df["top_notes"] = df["top_notes"].apply(_parse_pipe_list)
    df["middle_notes"] = df["middle_notes"].apply(_parse_pipe_list)
    df["base_notes"] = df["base_notes"].apply(_parse_pipe_list)
    df["main_accords"] = df["main_accords"].apply(_parse_pipe_list)

    # Normalize concentration
    df["concentration"] = df["concentration"].apply(_normalize_concentration)

    # Fix price tier using brand-level override where possible
    df["price_tier"] = df.apply(
        lambda r: _infer_tier(r["brand"]) or r.get("price_tier", "unknown"),
        axis=1,
    )

    # Drop rows with no note pyramid (useless for similarity)
    has_notes = df.apply(
        lambda r: len(r["top_notes"]) + len(r["middle_notes"]) + len(r["base_notes"]) > 0,
        axis=1,
    )
    dropped = (~has_notes).sum()
    df = df[has_notes].reset_index(drop=True)
    print(f"Dropped {dropped} rows with no notes. {len(df)} remain.")

    # Keep all fragrances in the index — unknown-tier ones still contribute to
    # similarity scoring; they're just excluded when filtering dupe results.
    unknown_count = (df["price_tier"] == "unknown").sum()
    print(f"  {unknown_count} fragrances have unknown price tier (kept in index, excluded from dupe results).")

    # Deduplicate by (name, brand)
    before = len(df)
    df = df.drop_duplicates(subset=["name", "brand"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicates. {len(df)} remain.")

    return df
