# E-Commerce Product Recommendation System

Content-based product recommendations over 1.4M Amazon products using TF-IDF and Word2Vec — both methods stay under 10 ms per query by restricting similarity search to the queried product's category.

---

## Results

| Method   | Median latency | P95 latency | Sub-100ms |
|----------|---------------|-------------|-----------|
| TF-IDF   | 6.2 ms        | —           | 100%      |
| Word2Vec | 3.5 ms        | —           | 100%      |

Tested on 1,426,337 products across 248 categories.

---

## How it works

Both recommenders are **category-aware**: instead of computing cosine similarity across all 1.4M products, each query is scoped to the ~5,700-product category slice of the query item. This is the only optimization needed to hit sub-10 ms latency without approximate nearest-neighbour indices.

**TF-IDF recommender**
- Builds a 50K-feature bigram TF-IDF matrix over cleaned product titles
- At query time, calls `linear_kernel` on the sparse category submatrix only

**Word2Vec recommender**
- Trains a 100-dimensional Word2Vec model (window=5, min_count=2, epochs=5)
- Represents each product as the mean of its token vectors, then L2-normalises
- At query time, uses a matrix dot-product on the normalised category submatrix

---

## Project structure

```
.
├── data/
│   └── amazon_products.csv      # not tracked by git (see .gitignore)
├── recommendation_system.ipynb  # single notebook, fully executable
├── requirements.txt
└── .gitignore
```

---

## Dataset

[Amazon Products 2023](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products) from Kaggle — 1.4M products.

Download and place the CSV at `data/amazon_products.csv` before running the notebook. The notebook auto-discovers column names in Section 1, so it adapts if the schema changes.

Key columns used:

| Column         | Role                        |
|----------------|-----------------------------|
| `asin`         | Unique product identifier   |
| `title`        | Product name (text input)   |
| `categoryName` | Category for index scoping  |

---

## Installation

```bash
pip install -r requirements.txt
```

All dependencies:

```
numpy  pandas  matplotlib  seaborn  scikit-learn  gensim  scipy  jupyter
```

---

## Running the notebook

```bash
jupyter notebook recommendation_system.ipynb
```

Or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace recommendation_system.ipynb
```

Execution time on the full 1.4M dataset is roughly 10–15 minutes (Word2Vec training dominates).

---

## Notebook sections

| Section | What it does |
|---------|-------------|
| 1. Setup & Data Loading | Imports, loads CSV, prints schema and missing-value counts |
| 2. Exploratory Data Analysis | Category distribution bar chart, title-length histogram |
| 3. Text Preprocessing | `clean_text` function, applies to titles, drops short rows |
| 4. TF-IDF Recommender | Builds sparse matrix, category index, `get_tfidf_recommendations` |
| 5. Word2Vec Recommender | Trains model, builds normalised doc-vector matrix, `get_w2v_recommendations` |
| 6. Performance Benchmark | 50 × 5 timed trials, prints mean / median / P95 / sub-100ms% |
| 7. Demo | Top-5 from each method for 3 products across 3 categories |
| 8. Summary | One-block summary of dataset size, categories, and latency |

---

## Design decisions

- **No approximate nearest-neighbour (ANN):** Category scoping alone cuts the search space by ~248×, making exact cosine similarity fast enough without FAISS or HNSW.
- **`linear_kernel` over `cosine_similarity`:** Avoids a redundant normalisation pass on already-unit-norm vectors; marginally faster on large sparse slices.
- **`sublinear_tf=True`:** Dampens term-frequency explosion on product titles that repeat keywords (e.g. "wireless wireless headphones").
- **Pre-normalised Word2Vec matrix:** Dot product of two unit-norm vectors equals cosine similarity, so no division at query time.
