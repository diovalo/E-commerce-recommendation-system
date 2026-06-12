"""Run once locally to generate models/ artifacts used by app.py."""
import json
import os
import pickle
import re
import string

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
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
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f, protocol=5)
save_npz("models/tfidf_matrix.npz", tfidf_matrix)
print(f"  Shape {tfidf_matrix.shape}  |  {tfidf_matrix.nnz:,} non-zeros")

print("Building category index...")
category_index: dict[str, list[int]] = {}
for row_idx, cat in enumerate(df["categoryName"]):
    category_index.setdefault(cat, []).append(row_idx)
with open("models/category_index.json", "w") as f:
    json.dump(category_index, f)
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
w2v_model.wv.save("models/w2v_vectors.kv")
print(f"  Vocab: {len(w2v_model.wv):,}")

print("Computing document vectors...")


def get_doc_vector(tokens: list[str]) -> np.ndarray:
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vecs:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vecs, axis=0)


doc_vectors = np.vstack([get_doc_vector(t) for t in tokenized])
doc_vectors_norm = normalize(doc_vectors).astype(np.float32)
np.save("models/doc_vectors_norm.npy", doc_vectors_norm)
print(f"  Shape {doc_vectors_norm.shape}  |  dtype {doc_vectors_norm.dtype}")

print("\nAll artifacts saved to models/")
for fname in sorted(os.listdir("models")):
    size_mb = os.path.getsize(f"models/{fname}") / 1_048_576
    print(f"  {fname:<30} {size_mb:.1f} MB")
