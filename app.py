import time
import re
import string

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Ecommerce Recommendation System", layout="wide")

_PUNCT_DIGITS = re.compile(f"[{re.escape(string.punctuation)}0-9]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _PUNCT_DIGITS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


@st.cache_resource(show_spinner="Loading data and training models — this takes about a minute on first run…")
def load_everything() -> tuple:
    df = pd.read_csv("data/sample_products.csv")
    df["clean_text"] = df["title"].apply(clean_text)
    df = df[df["clean_text"].str.len() >= 3].reset_index(drop=True)

    tfidf_vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])

    category_index: dict[str, list[int]] = {}
    for row_idx, cat in enumerate(df["categoryName"]):
        category_index.setdefault(cat, []).append(row_idx)

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
    w2v_vocab_size = len(w2v_model.wv)

    def _doc_vector(tokens: list[str]) -> np.ndarray:
        vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        if not vecs:
            return np.zeros(w2v_model.vector_size)
        return np.mean(vecs, axis=0)

    doc_vectors = np.vstack([_doc_vector(tokens) for tokens in tokenized])
    doc_vectors_norm = normalize(doc_vectors)

    return df, tfidf_matrix, category_index, doc_vectors_norm, w2v_vocab_size


def get_tfidf_recommendations(
    product_idx: int,
    df: pd.DataFrame,
    tfidf_matrix,
    category_index: dict,
    top_k: int = 10,
) -> pd.DataFrame:
    category = df.at[product_idx, "categoryName"]
    cat_indices = category_index[category]
    scores = linear_kernel(tfidf_matrix[product_idx], tfidf_matrix[cat_indices]).flatten()
    sorted_positions = np.argsort(scores)[::-1]
    results = []
    for pos in sorted_positions:
        idx = cat_indices[pos]
        if idx == product_idx:
            continue
        results.append({
            "title": df.at[idx, "title"],
            "category": category,
            "similarity_score": round(float(scores[pos]), 4),
        })
        if len(results) >= top_k:
            break
    return pd.DataFrame(results)


def get_w2v_recommendations(
    product_idx: int,
    df: pd.DataFrame,
    doc_vectors_norm: np.ndarray,
    category_index: dict,
    top_k: int = 10,
) -> pd.DataFrame:
    category = df.at[product_idx, "categoryName"]
    cat_indices = np.array(category_index[category])
    scores = doc_vectors_norm[cat_indices] @ doc_vectors_norm[product_idx]
    sorted_positions = np.argsort(scores)[::-1]
    results = []
    for pos in sorted_positions:
        idx = int(cat_indices[pos])
        if idx == product_idx:
            continue
        results.append({
            "title": df.at[idx, "title"],
            "category": category,
            "similarity_score": round(float(scores[pos]), 4),
        })
        if len(results) >= top_k:
            break
    return pd.DataFrame(results)


def _render_recs(recs: pd.DataFrame, latency_ms: float) -> None:
    if recs.empty:
        st.info("Not enough products in this category to generate recommendations.")
        return
    for _, row in recs.iterrows():
        title = row["title"]
        score = float(row["similarity_score"])
        label = title[:88] + "…" if len(title) > 88 else title
        st.markdown(f"**{label}**")
        st.progress(min(1.0, max(0.0, score)))
        st.caption(f"Score: {score:.4f}")
    st.info(f"Latency: {latency_ms:.1f} ms")


df, tfidf_matrix, category_index, doc_vectors_norm, w2v_vocab_size = load_everything()

st.title("Ecommerce Product Recommendation System")
st.caption("Amazon Products 2023 · Content-based filtering · TF-IDF + Word2Vec · Category-aware cosine similarity")

tab1, tab2, tab3 = st.tabs(["Search & Recommend", "Dataset Overview", "Performance Benchmark"])

with tab1:
    query = st.text_input(
        "Search for a product",
        placeholder="e.g. 'laptop', 'yoga mat', 'coffee maker'",
    )

    if query:
        matches = df[df["title"].str.contains(query, case=False, na=False)].head(10)

        if matches.empty:
            st.warning("No products found. Try a different search term.")
        else:
            match_titles = matches["title"].tolist()
            match_indices = matches.index.tolist()

            selected_pos = st.selectbox(
                "Select a product",
                range(len(match_titles)),
                format_func=lambda i: match_titles[i],
            )
            product_idx = match_indices[selected_pos]
            category = df.at[product_idx, "categoryName"]

            st.markdown(f"**{match_titles[selected_pos]}**")
            st.caption(f"Category: {category}  ·  {len(category_index[category])} products in this category")

            st.divider()

            col_tfidf, col_w2v = st.columns(2)

            with col_tfidf:
                st.subheader("TF-IDF Top 10")
                t0 = time.perf_counter()
                tfidf_recs = get_tfidf_recommendations(product_idx, df, tfidf_matrix, category_index)
                tfidf_ms = (time.perf_counter() - t0) * 1000
                _render_recs(tfidf_recs, tfidf_ms)

            with col_w2v:
                st.subheader("Word2Vec Top 10")
                t0 = time.perf_counter()
                w2v_recs = get_w2v_recommendations(product_idx, df, doc_vectors_norm, category_index)
                w2v_ms = (time.perf_counter() - t0) * 1000
                _render_recs(w2v_recs, w2v_ms)

with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products (full dataset)", "1,426,337")
    c2.metric("Categories", "248")
    c3.metric("Word2Vec vocab (full dataset)", "117,653")
    c4.metric("Demo sample", f"{len(df):,}")

    st.caption(
        "Full dataset: 1.4M+ Amazon products (2023). "
        "Demo runs on a stratified sample — up to 500 products per category."
    )

    st.divider()

    top_cats = df["categoryName"].value_counts().head(15).reset_index()
    top_cats.columns = ["category", "count"]

    fig_cats = px.bar(
        top_cats,
        x="count",
        y="category",
        orientation="h",
        title="Product Distribution by Category (Demo Sample — Top 15)",
        labels={"count": "Number of Products", "category": "Category"},
        color="count",
        color_continuous_scale="Blues",
    )
    fig_cats.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=500,
    )
    st.plotly_chart(fig_cats, use_container_width=True)

with tab3:
    bench_df = pd.DataFrame({
        "Method": ["TF-IDF", "Word2Vec"],
        "Median latency": ["6.2 ms", "3.5 ms"],
        "P95 latency": ["—", "—"],
        "Sub-100ms": ["100%", "100%"],
    })
    st.table(bench_df)

    fig_bench = px.bar(
        x=["TF-IDF", "Word2Vec"],
        y=[6.2, 3.5],
        labels={"x": "Method", "y": "Median Latency (ms)"},
        title="Median Query Latency: TF-IDF vs Word2Vec (Full 1.4M Dataset)",
        color=["TF-IDF", "Word2Vec"],
        color_discrete_map={"TF-IDF": "#1f77b4", "Word2Vec": "#ff7f0e"},
    )
    fig_bench.update_layout(showlegend=False, yaxis_range=[0, 8])
    st.plotly_chart(fig_bench, use_container_width=True)

    st.caption(
        "Benchmarked on the full 1.4M-product dataset across 50 trials × 5 products. "
        "Demo latency will vary on the sample."
    )
