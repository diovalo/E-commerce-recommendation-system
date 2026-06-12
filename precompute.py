"""Run once locally to generate models/ artifacts used by app.py."""
import json
import os
import re
import string

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

os.makedirs("models", exist_ok=True)

_PUNCT_DIGITS = re.compile(f"[{re.escape(string.punctuation)}0-9]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _PUNCT_DIGITS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


print("Loading sample...")
df = pd.read_csv("data/sample_products.csv")
df["clean_text"] = df["title"].apply(clean_text)
df = df[df["clean_text"].str.len() >= 3].reset_index(drop=True)
print(f"  {len(df):,} products  |  {df['categoryName'].nunique()} categories")

df[["title", "categoryName"]].to_csv("models/products.csv", index=False)
print("  Saved models/products.csv")

print("Building TF-IDF matrix...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])
print(f"  Shape {tfidf_matrix.shape}  |  {tfidf_matrix.nnz:,} non-zeros")

print("Building category index...")
category_index: dict[str, list[int]] = {}
for row_idx, cat in enumerate(df["categoryName"]):
    category_index.setdefault(cat, []).append(row_idx)
print(f"  {len(category_index)} categories")

print("Training Word2Vec...")
tokenized = [text.split() for text in df["clean_text"]]
w2v_model = Word2Vec(
    sentences=tokenized,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=5,
    seed=42,
)
print(f"  Vocab: {len(w2v_model.wv):,}")

print("Computing document vectors...")


def get_doc_vector(tokens: list[str]) -> np.ndarray:
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vecs:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vecs, axis=0)


doc_vectors = np.vstack([get_doc_vector(t) for t in tokenized])
doc_vectors_norm = normalize(doc_vectors).astype(np.float32)
print(f"  Shape {doc_vectors_norm.shape}")

print("Pre-computing TF-IDF recommendations (batch by category)...")
tfidf_rows: list[tuple] = []
for indices in category_index.values():
    cat_mat = tfidf_matrix[indices]
    sim = linear_kernel(cat_mat, cat_mat)
    for local_i, global_i in enumerate(indices):
        row = sim[local_i]
        rank = 0
        for pos in np.argsort(row)[::-1]:
            if indices[pos] == global_i:
                continue
            tfidf_rows.append((global_i, indices[pos], round(float(row[pos]), 4)))
            rank += 1
            if rank == 10:
                break
print(f"  {len(tfidf_rows):,} TF-IDF recommendation rows")

print("Pre-computing Word2Vec recommendations (batch by category)...")
w2v_rows: list[tuple] = []
for indices in category_index.values():
    idx_arr = np.array(indices)
    vecs = doc_vectors_norm[idx_arr]
    sim = vecs @ vecs.T
    for local_i, global_i in enumerate(indices):
        row = sim[local_i]
        rank = 0
        for pos in np.argsort(row)[::-1]:
            if int(idx_arr[pos]) == global_i:
                continue
            w2v_rows.append((global_i, int(idx_arr[pos]), round(float(row[pos]), 4)))
            rank += 1
            if rank == 10:
                break
print(f"  {len(w2v_rows):,} Word2Vec recommendation rows")

tfidf_df = pd.DataFrame(tfidf_rows, columns=["product_idx", "rec_idx", "score"])
tfidf_df["method"] = "tfidf"
w2v_df = pd.DataFrame(w2v_rows, columns=["product_idx", "rec_idx", "score"])
w2v_df["method"] = "w2v"
recs = pd.concat([tfidf_df, w2v_df], ignore_index=True)
recs.to_csv("models/recs.csv.gz", index=False, compression="gzip")
print(f"  Saved models/recs.csv.gz  ({len(recs):,} rows)")

meta = {
    "w2v_vocab_size": len(w2v_model.wv),
    "n_products": len(df),
    "n_categories": df["categoryName"].nunique(),
}
with open("models/meta.json", "w") as f:
    json.dump(meta, f)
print("  Saved models/meta.json")

print("\nAll artifacts saved to models/")
for fname in sorted(os.listdir("models")):
    size_mb = os.path.getsize(f"models/{fname}") / 1_048_576
    print(f"  {fname:<30} {size_mb:.1f} MB")
