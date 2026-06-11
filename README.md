# Fragrance Dupe Finder

A recommendation engine that finds affordable alternatives to luxury fragrances using note-profile similarity. Built with Python, scikit-learn, and Streamlit.

## How It Works

Every fragrance has a **note pyramid** — top notes (what you smell first), heart notes (the core), and base notes (the dry-down that lingers for hours). This app encodes each fragrance as a weighted vector across a vocabulary of 1,600+ notes, then uses cosine similarity to find fragrances with the most similar scent profiles.

**Feature engineering:**
- Base notes weighted 3×, heart notes 2×, top notes 1× — base notes define how a fragrance actually wears, so they contribute more to similarity than top notes which evaporate in ~15 minutes
- Scent family (woody, oriental, fresh, etc.) one-hot encoded and concatenated to prevent cross-family false matches
- 23,800+ fragrances across luxury and affordable brands

**Recommendation:**
- Cosine similarity matrix precomputed across all fragrances
- Query fragrance → ranked list of affordable dupes filtered by price tier
- Optional AI-generated sensory comparison via the Claude API

## Demo

Search any luxury fragrance — Creed, Tom Ford, Chanel, Dior — and get the top affordable dupes with similarity scores and note breakdowns.

| Input | Top Dupe | Similarity |
|-------|----------|------------|
| Bleu de Chanel | Blue de Chance (Maison Alhambra) | 97.3% |
| Sauvage (Dior) | Magnificent Blue Pour Homme (Armaf) | 81.2% |
| Eros (Versace) | Versencia Oro (Maison Alhambra) | 93.3% |

## Stack

- **Python** — pandas, scikit-learn, numpy
- **ML** — MultiLabelBinarizer, cosine similarity, weighted feature encoding
- **Data** — 23,800+ fragrances from Fragrantica via Kaggle
- **AI** — Claude API (Anthropic) for sensory descriptions
- **UI** — Streamlit

## Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download the dataset**

Download [this Kaggle dataset](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset) and place `fra_cleaned.csv` in `data/raw/`.

**3. Build the index**
```bash
python src/ingest_kaggle.py
python build_index.py --raw data/raw/fragrances_kaggle.csv
```

**4. Run the app**
```bash
streamlit run app.py
```

**5. (Optional) Enable AI descriptions**

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```
