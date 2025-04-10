import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
FILE_MAP = {
    "Fine-grained (0.5 threshold, ~9000)": "labeled_clusters_dt0.5.json",
    "Moderate (1.0 threshold, ~3000)": "labeled_clusters_dt1.json",
    "Broad (1.5 threshold, ~1600)": "labeled_clusters_dt1.5.json"
}
DATA_DIR = "./labeled_cluster_results"

# === Load all datasets once ===
@st.cache_data
def load_all_clusters():
    data = {}
    for label, file in FILE_MAP.items():
        path = os.path.join(DATA_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            clusters = json.load(f)
            data[label] = clusters
    return data

cluster_data = load_all_clusters()

# === Load embeddings for semantic search ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

@st.cache_data
def get_all_keywords_and_embeddings(cluster_data):
    all_keywords = list(set(kw for data in cluster_data.values() for kw in data.keys()))
    embeddings = model.encode(all_keywords, show_progress_bar=False)
    return all_keywords, embeddings

all_keywords, all_embeddings = get_all_keywords_and_embeddings(cluster_data)

# === UI ===
st.title("📈 Materials Keyword Trend Tool")

selected_granularity = st.selectbox("Select specificity level:", list(FILE_MAP.keys()))
selected_data = cluster_data[selected_granularity]

allow_multi = st.checkbox("Allow multi-select")
query = st.text_input("Type keyword to search:")

filtered_keywords = sorted([kw for kw in selected_data if query.lower() in kw.lower()])

if allow_multi:
    selected_keywords = st.multiselect("Select keywords:", filtered_keywords)
else:
    selected_keywords = [st.selectbox("Select keyword:", filtered_keywords)] if filtered_keywords else []



# === Plot ===
def plot_keywords(keywords, data):
    years = list(range(2011, 2025))
    df = pd.DataFrame(index=years)

    for kw in keywords:
        year_counts = data.get(kw, {})
        df[kw] = [year_counts.get(str(y), 0) for y in years]

    st.line_chart(df)
    st.bar_chart(df)
    return df

if selected_keywords:
    st.subheader("📊 Trend over years")
    df = plot_keywords(selected_keywords, selected_data)

    # Save plot button
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_title("Keyword Trends")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="keyword_trends.png", mime="image/png")

# === Semantic Search ===
st.markdown("---")
st.subheader("🔍 Semantic Similar Keywords")

sem_kw = st.text_input("Enter a keyword to find similar ones:", key="sem_kw")
similar_keywords = []

if sem_kw:
    emb = model.encode([sem_kw], convert_to_tensor=True)
    similarities = util.cos_sim(emb, all_embeddings)[0]
    top_k = min(10, len(similarities))
    top_indices = torch.topk(similarities, k=top_k).indices.numpy()
    similar_keywords = [all_keywords[i] for i in top_indices if all_keywords[i] != sem_kw]

    st.write("Top similar keywords found:")
    selected_similar_keywords = st.multiselect(
        "Select keywords to add to plot:", similar_keywords, key="semantic_multiselect"
    )

    if selected_similar_keywords:
        # Optionally merge with the currently selected keywords from earlier
        st.write("✅ Selected for plotting:", selected_similar_keywords)

        if st.button("Plot selected similar keywords"):
            plot_keywords(selected_similar_keywords, selected_data)
