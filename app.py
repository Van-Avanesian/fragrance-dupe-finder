"""
Fragrance Dupe Finder — Streamlit UI

Run:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data/processed"

st.set_page_config(
    page_title="Fragrance Dupe Finder",
    page_icon="🌹",
    layout="wide",
)

# ------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ------------------------------------------------------------------

@st.cache_resource
def load_finder():
    from src.recommender import DupeFinder
    return DupeFinder.load(DATA_DIR)


def data_ready() -> bool:
    return (
        Path(DATA_DIR, "fragrances.parquet").exists()
        and Path(DATA_DIR, "similarity_matrix.npy").exists()
        and Path(DATA_DIR, "encoders.pkl").exists()
    )



# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.title("🌹 Fragrance Dupe Finder")
st.caption("Find affordable alternatives to luxury fragrances using note-profile similarity.")

if not data_ready():
    st.warning(
        "No index found. Run the scraper and then `python build_index.py` to build the database.",
        icon="⚠️",
    )
    st.code("python -m scraper.fragrantica\npython build_index.py", language="bash")
    st.stop()

finder = load_finder()

# Sidebar options
with st.sidebar:
    st.header("Options")
    n_results = st.slider("Number of dupes", 1, 10, 5)
    affordable_only = st.checkbox("Affordable dupes only", value=True)
    show_ai = st.checkbox("AI sensory descriptions (uses Claude API)", value=True)
    st.divider()
    st.caption(
        "Similarity is cosine similarity on the weighted note pyramid + scent family vector. "
        "Base notes are weighted 3×, heart notes 2×, top notes 1×."
    )

# Fragrance search
all_names = finder.all_fragrance_names()
query = st.selectbox(
    "Search for a fragrance",
    options=[""] + all_names,
    index=0,
    placeholder="Start typing a fragrance name…",
)

if not query:
    st.info("Select a fragrance above to find its dupes.")
    st.stop()

# Check API key if AI descriptions requested
if show_ai and not os.getenv("ANTHROPIC_API_KEY"):
    st.warning(
        "Set `ANTHROPIC_API_KEY` in your `.env` file to enable AI descriptions.",
        icon="🔑",
    )
    show_ai = False

with st.spinner("Finding dupes…"):
    results = finder.find_dupes(
        query,
        n=n_results,
        affordable_only=affordable_only,
        with_descriptions=show_ai,
    )

if not results:
    st.error(f"No dupes found for '{query}'. The fragrance may not be in the database.")
    st.stop()

# Show the input fragrance details
input_idx = finder._resolve_index(query.lower().split(" by ")[0] if " by " not in query.lower() else query.lower())
if input_idx is not None:
    input_row = finder.df.iloc[input_idx]
    with st.expander("Input fragrance details", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Brand", input_row["brand"])
        col2.metric("Concentration", input_row.get("concentration", "—"))
        col3.metric("Price tier", input_row["price_tier"].capitalize())
        if input_row["main_accords"]:
            st.write("**Main accords:**", " · ".join(input_row["main_accords"]))
        if input_row["top_notes"]:
            st.write("**Top notes:**", ", ".join(input_row["top_notes"]))
        if input_row["middle_notes"]:
            st.write("**Heart notes:**", ", ".join(input_row["middle_notes"]))
        if input_row["base_notes"]:
            st.write("**Base notes:**", ", ".join(input_row["base_notes"]))

st.divider()
st.subheader(f"Top {len(results)} {'affordable ' if affordable_only else ''}dupes")

for i, rec in enumerate(results, 1):
    with st.container():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"### {i}. {rec['name']} — *{rec['brand']}*")
        with col_b:
            st.metric("Similarity", rec["similarity_pct"])

        meta_cols = st.columns(3)
        if rec["concentration"]:
            meta_cols[0].write(f"**Concentration:** {rec['concentration']}")
        if rec["main_accords"]:
            meta_cols[1].write(f"**Accords:** {' · '.join(rec['main_accords'][:3])}")
        meta_cols[2].write(f"**Tier:** {rec['price_tier'].capitalize()}")

        # Notes
        notes_col1, notes_col2, notes_col3 = st.columns(3)
        if rec["top_notes"]:
            notes_col1.write(f"**Top:** {', '.join(rec['top_notes'])}")
        if rec["middle_notes"]:
            notes_col2.write(f"**Heart:** {', '.join(rec['middle_notes'])}")
        if rec["base_notes"]:
            notes_col3.write(f"**Base:** {', '.join(rec['base_notes'])}")

        # AI description
        if show_ai and rec.get("description"):
            st.info(rec["description"], icon="🤖")

        st.divider()
