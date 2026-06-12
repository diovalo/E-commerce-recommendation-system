# E-Commerce Product Recommendation System

Content-based product recommendations over **1.4M real Amazon listings** using TF-IDF and Word2Vec : both methods stay under 10 ms per query by restricting similarity search to the product's own category.

**[Live Dashboard →](https://e-commerce-recommendation-system-8zjl6qf4zau7hofwckwjaf.streamlit.app/)**

---

## Results

| Method | Median latency | P95 latency | Sub-100ms |
|--------|---------------|-------------|-----------|
| TF-IDF | **6.2 ms** | 14.7 ms | 100% |
| Word2Vec | **3.5 ms** | 10.9 ms | 100% |

Benchmarked on the full 1,426,337-product dataset — 50 trials × 5 randomly sampled products.

---

## How it works

Both recommenders are **category-aware**: instead of running cosine similarity across all 1.4M products, each query is scoped to the ~5,700-product slice that shares the queried item's category. This single optimisation is enough to hit sub-10 ms with exact cosine similarity — no approximate nearest-neighbour index needed.

**TF-IDF**
- 50K-feature bigram matrix over cleaned product titles (`sublinear_tf=True`)
- Query: `linear_kernel` on the sparse category submatrix only

**Word2Vec**
- 100-dimensional model (window=5, min_count=2, epochs=5) — 117,653-word vocabulary
- Each product = mean of its token vectors, L2-normalised
- Query: dot product on the normalised category submatrix (equivalent to cosine similarity)

---

## Project structure

```
.
├── recommendation_system.ipynb  # Full analysis notebook (all sections executed)
├── app.py                       # Streamlit dashboard (Search, Overview, Benchmark tabs)
├── precompute.py                # One-time script — trains models, writes artifacts to models/
├── create_sample.py             # Generates stratified 50K-product demo sample
├── requirements.txt             # Runtime deps (pandas, plotly only)
├── data/
│   ├── amazon_products.csv      # Full dataset — not tracked (see Dataset section)
│   ├── amazon_categories.csv    # Category ID → name mapping — not tracked
│   └── sample_products.csv      # Stratified 50K sample — tracked, used by app
└── models/                      # Pre-computed artifacts — tracked
    ├── products.csv             # Cleaned product subset
    ├── recs.csv.gz              # Pre-computed top-10 recs per product (both methods)
    ├── doc_vectors_norm.npy     # Normalised Word2Vec document vectors
    ├── w2v_vectors.kv           # Gensim KeyedVectors
    ├── category_index.json      # Category → row index mapping
    └── meta.json                # Vocab size, product count
```

---

## Dataset

[Amazon Products 2023](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products) — 1,426,337 products across 248 categories.

Download via kagglehub:

```python
import kagglehub
path = kagglehub.dataset_download("asaniczka/amazon-products-dataset-2023-1-4m-products")
```

Place `amazon_products.csv` and `amazon_categories.csv` in `data/` before running the notebook or `precompute.py`.

---

## Getting started

```bash
git clone https://github.com/diovalo/E-commerce-recommendation-system.git
cd E-commerce-recommendation-system
pip install pandas numpy scikit-learn gensim scipy matplotlib seaborn jupyter
```

**Option A — run the notebook (full analysis):**
```bash
jupyter notebook recommendation_system.ipynb
# Full execution ~10–15 min (Word2Vec training dominates)
```

**Option B — launch the Streamlit dashboard locally:**
```bash
pip install streamlit plotly
python create_sample.py      # generate data/sample_products.csv (once)
python precompute.py         # train models, write models/ (once, ~10 min)
streamlit run app.py
```

The Streamlit app loads from pre-computed artifacts in `models/` — no ML packages needed at runtime.

---

## Notebook sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Data Loading | Imports, load + merge CSVs, print schema |
| 2 | Exploratory Data Analysis | Category distribution, title-length histogram |
| 3 | Text Preprocessing | `clean_text` function, apply to titles, drop short rows |
| 4 | TF-IDF Recommender | Build sparse matrix, category index, `get_tfidf_recommendations` |
| 5 | Word2Vec Recommender | Train model, build normalised doc-vector matrix, `get_w2v_recommendations` |
| 6 | Performance Benchmark | 50 × 5 timed trials — mean / median / P95 / sub-100ms % |
| 7 | Demo | Top-5 from each method across 3 categories |
| 8 | Summary | Dataset size, categories, and latency in one block |

---

## Design decisions

- **No ANN index (FAISS/HNSW):** Category scoping cuts the search space by ~248×, making exact cosine similarity fast enough without extra infrastructure.
- **`linear_kernel` over `cosine_similarity`:** Skips a redundant normalisation pass on already unit-norm sparse vectors.
- **`sublinear_tf=True`:** Dampens keyword repetition in product titles (e.g. "wireless wireless headphones").
- **Pre-normalised Word2Vec matrix:** Dot product of two unit-norm vectors equals cosine similarity — no division at query time.
- **Pre-computed artifacts:** `precompute.py` runs once offline; the Streamlit app loads serialised results so runtime only needs pandas and plotly, avoiding a 20-minute cold start on Streamlit Cloud.
