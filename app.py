import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Ecommerce Recommendation System", layout="wide")


@st.cache_resource(show_spinner="Loading recommendation models…")
def load_everything() -> tuple:
    df = pd.read_csv("models/products.csv")
    recs = pd.read_csv("models/recs.csv.gz", compression="gzip")
    with open("models/meta.json") as f:
        meta = json.load(f)
    return df, recs, meta


def _rows_to_df(rows: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in rows.iterrows():
        rec = df.iloc[int(row["rec_idx"])]
        results.append({
            "title": rec["title"],
            "category": rec["categoryName"],
            "similarity_score": float(row["score"]),
        })
    return pd.DataFrame(results)


def get_tfidf_recommendations(
    product_idx: int, df: pd.DataFrame, recs: pd.DataFrame, top_k: int = 10
) -> pd.DataFrame:
    rows = recs[(recs["product_idx"] == product_idx) & (recs["method"] == "tfidf")].head(top_k)
    return _rows_to_df(rows, df)


def get_w2v_recommendations(
    product_idx: int, df: pd.DataFrame, recs: pd.DataFrame, top_k: int = 10
) -> pd.DataFrame:
    rows = recs[(recs["product_idx"] == product_idx) & (recs["method"] == "w2v")].head(top_k)
    return _rows_to_df(rows, df)


def _render_recs(recs_df: pd.DataFrame, latency_ms: float) -> None:
    if recs_df.empty:
        st.info("Not enough products in this category to generate recommendations.")
        return
    for _, row in recs_df.iterrows():
        title = row["title"]
        score = float(row["similarity_score"])
        label = title[:88] + "…" if len(title) > 88 else title
        st.markdown(f"**{label}**")
        st.progress(min(1.0, max(0.0, score)))
        st.caption(f"Score: {score:.4f}")
    st.info(f"Latency: {latency_ms:.1f} ms")


df, recs, meta = load_everything()

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
            cat_size = (recs["product_idx"] == product_idx).sum() // 2
            st.caption(f"Category: {category}  ·  ~{cat_size} products in this category")

            st.divider()

            col_tfidf, col_w2v = st.columns(2)

            with col_tfidf:
                st.subheader("TF-IDF Top 10")
                t0 = time.perf_counter()
                tfidf_recs = get_tfidf_recommendations(product_idx, df, recs)
                tfidf_ms = (time.perf_counter() - t0) * 1000
                _render_recs(tfidf_recs, tfidf_ms)

            with col_w2v:
                st.subheader("Word2Vec Top 10")
                t0 = time.perf_counter()
                w2v_recs = get_w2v_recommendations(product_idx, df, recs)
                w2v_ms = (time.perf_counter() - t0) * 1000
                _render_recs(w2v_recs, w2v_ms)

with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products (full dataset)", "1,426,337")
    c2.metric("Categories", "248")
    c3.metric("Word2Vec vocab (demo)", f"{meta['w2v_vocab_size']:,}")
    c4.metric("Demo sample", f"{meta['n_products']:,}")

    st.caption(
        "Full dataset: 1.4M+ Amazon products (2023). "
        "Demo runs on a stratified sample — up to 200 products per category."
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
