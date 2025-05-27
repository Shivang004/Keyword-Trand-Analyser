import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import numpy as np
from sentence_transformers import SentenceTransformer
import numpy.linalg as LA

st.set_page_config(page_title="Research Analytics Dashboard", layout="wide")
QUICK_GUIDE = """
## 🚀 Quick Start Guide

### Step 1: Choose Your Data Granularity
Select your preferred detail level from the **Summary Granularity** section in the sidebar:
- **Fine-grained (~9000)** - Most detailed analysis
- **Moderate (~4000)** - Balanced view (recommended)
- **Broad (~1800)** - High-level overview

### Step 2: Navigate Through Analysis Types
Click on any analysis from the **Navigation** menu in the sidebar:

**📅 Year-Based Analysis**
- **"1. Year → Keywords"** - See what topics were popular in a specific year
- **"2. Year → Countries"** - Find which countries were most active in a year

**🔑 Keyword-Based Analysis** 
- **"3. Keyword → Years"** - Track how topics evolved over time
- **"4. Keyword → Countries"** - See which countries research specific topics

**🌍 Country-Based Analysis**
- **"5. Country → Years"** - View a country's research activity over time  
- **"6. Country → Keywords"** - Discover what topics countries focus on

**🔄 Combined Analysis**
- **"7. Year + Keyword → Countries"** - Which countries researched X topic in Y year?
- **"8. Year + Country → Keywords"** - What did X country research in Y year?
- **"9. Keyword + Country → Years"** - When was X country most active in Y topic?

### Step 3: Use Interactive Features
- **Select multiple items** where available for comparisons
- **Adjust sliders** to show more/fewer results
- **Hover over charts** for detailed values
- **View data tables** below charts for exact numbers

### Step 4: Try Semantic Search
Click **"Semantic Search"** for AI-powered keyword discovery:
1. Type any research term or phrase
2. Get 20 most similar keywords with similarity scores
3. Select interesting keywords for automatic analysis
4. View trends, countries, and timeline data instantly

---
*💡 **Pro Tip**: Start with "Moderate" granularity and use Semantic Search to discover relevant keywords you might not have thought of!*
"""

# Insert this in your main code after the title:
st.title("📊 Research Analytics Dashboard")
# Option 1: Expandable section
with st.expander(" 📖 How to Use This Dashboard - Quick Start Guide"):
    st.markdown(QUICK_GUIDE)

st.markdown(
    """
    **Explore research trends by year, country, and keyword.**
    - Use the sidebar to select the **summary granularity** and navigate between different types of analyses.
    - Select multiple keywords or countries where possible for comparative analytics!
    """
)
# --- App config ---
SUMMARY_OPTIONS = {
    "Fine-grained (~9000)": "output_summary_0.5",
    "Moderate (~4000)": "output_summary_1.0",
    "Broad (~1800)": "output_summary_1.5"
}

# --- Sidebar: Summary selection ---
st.sidebar.title("Summary Granularity")
summary_type = st.sidebar.radio(
    "Choose clustering granularity:",
    list(SUMMARY_OPTIONS.keys())
)
DATA_DIR = SUMMARY_OPTIONS[summary_type]

# --- Load main data ---
@st.cache_resource(show_spinner=True)
def load_data(data_dir):
    with open(os.path.join(data_dir, 'years_summary.json'), encoding='utf-8') as f:
        years_summary = json.load(f)
    with open(os.path.join(data_dir, 'keywords_summary.json'), encoding='utf-8') as f:
        keywords_summary = json.load(f)
    with open(os.path.join(data_dir, 'countries_summary.json'), encoding='utf-8') as f:
        countries_summary = json.load(f)
    return years_summary, keywords_summary, countries_summary

years_summary, keywords_summary, countries_summary = load_data(DATA_DIR)

def top_n_items(d, n=20):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:n])

# --- Load embeddings and keywords for semantic search ---
# Show loading spinner
@st.cache_resource(show_spinner=True)
def load_embeddings_and_keywords(granularity_key):
    file_prefix_map = {
        "Fine-grained (~9000)": "fine-grained",
        "Moderate (~4000)": "moderate",
        "Broad (~1800)": "broad"
    }
    prefix = file_prefix_map[granularity_key]
    emb_file = os.path.join(SUMMARY_OPTIONS[granularity_key], f"{prefix}_embeddings.npy")
    kw_file = os.path.join(SUMMARY_OPTIONS[granularity_key], f"{prefix}_keywords.json")
    embeddings = np.load(emb_file)
    with open(kw_file, 'r', encoding='utf-8') as f:
        keywords = json.load(f)
    return embeddings, keywords

# --- Load SentenceTransformer model ---
# Show loading spinner
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def semantic_search(query, embeddings, keywords, model, top_k=20):
    query_emb = model.encode([query])[0]
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_emb_norm = query_emb / np.linalg.norm(query_emb)
    sims = np.dot(emb_norm, query_emb_norm)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(keywords[i], sims[i]) for i in top_idx]

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to Section",
    [
        "1. Year → Keywords",
        "2. Year → Countries",
        "3. Keyword → Years",
        "4. Keyword → Countries",
        "5. Country → Years",
        "6. Country → Keywords",
        "7. Year + Keyword → Countries",
        "8. Year + Country → Keywords",
        "9. Keyword + Country → Years",
        "Semantic Search"
    ]
)

st.info("ℹ️ **Note:** Data for the year 2019 is not available and has been excluded from the charts.")

# --- Sections 1 to 9 (same as your original code, using Plotly for charts) ---

if section == "1. Year → Keywords":
    st.header("Most Popular Keywords in a Selected Year")
    year = st.selectbox("Select Year", sorted(years_summary.keys()))
    n = st.slider("Top N Keywords", 5, 30, 15)
    all_keywords = list(years_summary.get(year, {}).get('keywords', {}).keys())
    selected_keywords = st.multiselect("Filter to specific keywords (optional):", sorted(all_keywords), default=[])
    data = years_summary.get(year, {}).get('keywords', {})
    if selected_keywords:
        data = {k: v for k, v in data.items() if k in selected_keywords}
    data = top_n_items(data, n)
    df = pd.DataFrame({"Keyword": list(data.keys()), "Count": list(data.values())})

    fig = px.bar(df, x="Keyword", y="Count", title=f"Top {n} Keywords in {year}")
    st.plotly_chart(fig, use_container_width=True)
        
    st.dataframe(df, use_container_width=True)

elif section == "2. Year → Countries":
    st.header("Most Active Countries in a Selected Year")
    st.warning("Note: Country data is missing for 2018.")
    year = st.selectbox("Select Year", sorted(years_summary.keys()))
    n = st.slider("Top N Countries", 5, 30, 15)
    all_countries = list(years_summary.get(year, {}).get('countries', {}).keys())
    selected_countries = st.multiselect("Filter to specific countries (optional):", sorted(all_countries), default=[])
    data = years_summary.get(year, {}).get('countries', {})
    if selected_countries:
        data = {k: v for k, v in data.items() if k in selected_countries}
    data = top_n_items(data, n)
    df = pd.DataFrame({"Country": list(data.keys()), "Count": list(data.values())})
    fig = px.bar(df, x="Country", y="Count", title=f"Top {n} Countries in {year}")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.choropleth(df, locations='Country',
                    locationmode='country names',  # use country names to identify countries
                    color='Count',
                    color_continuous_scale='Viridis',
                    title='Country Intensity Map')

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)

elif section == "3. Keyword → Years":
    st.header("Keyword Popularity Over Time")
    all_keywords = sorted(list(keywords_summary.keys()))
    selected_keywords = st.multiselect("Select Keywords", all_keywords, default=all_keywords[:1])
    chart_df = pd.DataFrame()
    for kw in selected_keywords:
        data = keywords_summary[kw]['years']
        df = pd.DataFrame({"Year": list(data.keys()), kw: list(data.values())})
        df = df.sort_values('Year')
        df.set_index("Year", inplace=True)
        chart_df = pd.concat([chart_df, df], axis=1)
    if not chart_df.empty:
        st.subheader("Trend Over Time")
        st.line_chart(chart_df, use_container_width=True)

        st.subheader("Year-wise Distribution")
        fig = px.bar(chart_df, title="Year-wise Distribution")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please select at least one keyword.")

elif section == "4. Keyword → Countries":
    st.header("Countries Researching Keyword(s)")
    st.warning("Note: Country data is missing for 2018.")
    all_keywords = sorted(list(keywords_summary.keys()))
    selected_keywords = st.multiselect("Select Keywords", all_keywords, default=all_keywords[:1])
    n = st.slider("Top N Countries", 5, 30, 15)
    if selected_keywords:
        chart_df = pd.DataFrame()
        for kw in selected_keywords:
            data = keywords_summary[kw]['countries']
            data = top_n_items(data, n)
            df = pd.DataFrame({"Country": list(data.keys()), kw: list(data.values())})
            df.set_index("Country", inplace=True)
            chart_df = pd.concat([chart_df, df], axis=1)
        fig = px.bar(chart_df, title="Countries Researching Keyword(s)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Create choropleth map using the data for the last keyword
        choropleth_df = pd.DataFrame({"Country": list(data.keys()), "Value": list(data.values())})
        fig = px.choropleth(choropleth_df, 
                    locations='Country',
                    locationmode='country names',
                    color='Value',
                    color_continuous_scale='Viridis',
                    title=f'Country Intensity Map for {kw}')

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(chart_df.reset_index(), use_container_width=True)
    else:
        st.info("Please select at least one keyword.")

elif section == "5. Country → Years":
    st.header("Country's Research Activity Over Time")
    st.warning("Note: Country data is missing for 2018.")
    all_countries = sorted(list(countries_summary.keys()))
    selected_countries = st.multiselect("Select Countries", all_countries, default=all_countries[:1])
    chart_df = pd.DataFrame()
    for country in selected_countries:
        data = countries_summary[country]['years']
        df = pd.DataFrame({"Year": list(data.keys()), country: list(data.values())})
        df = df.sort_values('Year')
        df.set_index("Year", inplace=True)
        chart_df = pd.concat([chart_df, df], axis=1)
    if not chart_df.empty:
        st.subheader("Trend Over Time")
        st.line_chart(chart_df, use_container_width=True)
        st.subheader("Year-wise Distribution")
        fig = px.bar(chart_df, title="Year-wise Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(chart_df.reset_index(), use_container_width=True)
    else:
        st.info("Please select at least one country.")

elif section == "6. Country → Keywords":
    st.header("Country's Top Research Focus Areas")
    st.warning("Note: Country data is missing for 2018.")
    all_countries = sorted(list(countries_summary.keys()))
    selected_countries = st.multiselect("Select Countries", all_countries, default=all_countries[:1])
    n = st.slider("Top N Keywords", 5, 30, 15)
    if selected_countries:
        chart_df = pd.DataFrame()
        for country in selected_countries:
            data = countries_summary[country]['keywords']
            data = top_n_items(data, n)
            df = pd.DataFrame({"Keyword": list(data.keys()), country: list(data.values())})
            df.set_index("Keyword", inplace=True)
            chart_df = pd.concat([chart_df, df], axis=1)
        fig = px.bar(chart_df, title="Country's Top Research Focus Areas")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(chart_df.reset_index(), use_container_width=True)
    else:
        st.info("Please select at least one country.")

elif section == "7. Year + Keyword → Countries":
    st.header("Which Countries Researched a Keyword in a Year?")
    st.warning("Note: Country data is missing for 2018.")
    year = st.selectbox("Select Year", sorted(years_summary.keys()))
    all_keywords = sorted(list(years_summary[year]['keywords'].keys()))
    selected_keywords = st.multiselect("Select Keywords", all_keywords, default=all_keywords[:1])
    n = st.slider("Top N Countries", 5, 30, 15)
    chart_df = pd.DataFrame()
    for keyword in selected_keywords:
        data = years_summary[year]['keyword_countries'].get(keyword, {})
        data = top_n_items(data, n)
        df = pd.DataFrame({"Country": list(data.keys()), keyword: list(data.values())})
        df.set_index("Country", inplace=True)
        chart_df = pd.concat([chart_df, df], axis=1)
    if not chart_df.empty:
        fig = px.bar(chart_df, title="Which Countries Researched a Keyword in a Year?")
        st.plotly_chart(fig, use_container_width=True)

        # Create choropleth map using the data for the last keyword
        choropleth_df = pd.DataFrame({"Country": list(data.keys()), "Value": list(data.values())})
        fig = px.choropleth(choropleth_df, 
                    locations='Country',
                    locationmode='country names',
                    color='Value',
                    color_continuous_scale='Viridis',
                    title=f'Country Intensity Map for {keyword}')

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(chart_df.reset_index(), use_container_width=True)
    else:
        st.warning("No data available for this selection.")

elif section == "8. Year + Country → Keywords":
    st.header("What Did a Country Research in a Year?")
    st.warning("Note: Country data is missing for 2018.")
    year = st.selectbox("Select Year", sorted(years_summary.keys()))
    all_countries = sorted(list(years_summary[year]['countries'].keys()))
    selected_countries = st.multiselect("Select Countries", all_countries, default=all_countries[:1])
    n = st.slider("Top N Keywords", 5, 30, 15)
    chart_df = pd.DataFrame()
    for country in selected_countries:
        data = years_summary[year]['country_keywords'].get(country, {})
        data = top_n_items(data, n)
        df = pd.DataFrame({"Keyword": list(data.keys()), country: list(data.values())})
        df.set_index("Keyword", inplace=True)
        chart_df = pd.concat([chart_df, df], axis=1)
    if not chart_df.empty:
        fig = px.bar(chart_df, title="What Did a Country Research in a Year?")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(chart_df.reset_index(), use_container_width=True)
    else:
        st.warning("No data available for this selection.")

elif section == "9. Keyword + Country → Years":
    st.header("When Was a Country Most Active in a Keyword?")
    st.warning("Note: Country data is missing for 2018.")
    all_keywords = sorted(list(keywords_summary.keys()))
    all_countries = sorted(list(countries_summary.keys()))
    selected_keywords = st.multiselect("Select Keywords", all_keywords, default=all_keywords[:1])
    selected_countries = st.multiselect("Select Countries", all_countries, default=all_countries[:1])
    chart_df = pd.DataFrame()
    
    for keyword in selected_keywords:
        for country in selected_countries:
            trend = {}
            for year, yearinfo in years_summary.items():
                count = yearinfo['keyword_countries'].get(keyword, {}).get(country, 0)
                if count > 0:
                    trend[year] = count
            if trend:
                df = pd.DataFrame({"Year": list(trend.keys()), f"{keyword} ({country})": list(trend.values())})
                df = df.sort_values('Year')
                df.set_index("Year", inplace=True)
                chart_df = pd.concat([chart_df, df], axis=1)
    
    if not chart_df.empty:
        st.subheader("Trend Over Time")
        st.line_chart(chart_df, use_container_width=True)
        st.subheader("Year-wise Distribution")
        fig = px.bar(chart_df, title="Year-wise Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(chart_df.reset_index(), use_container_width=True)
    else:
        st.warning("No data available for this selection.")

# --- Semantic Search Section ---
if section == "Semantic Search":
    st.header("Semantic Search for Keywords")

    # Use granularity selected in sidebar automatically
    granularity_for_search = summary_type
    
    embeddings, embed_keywords = load_embeddings_and_keywords(granularity_for_search)
    model = load_model()
    
    query = st.text_input("Enter keyword or phrase to search:")
    
    if query:
        results = semantic_search(query, embeddings, embed_keywords, model)
        matched_keywords = [kw for kw, score in results]
        scores = [score for kw, score in results]
        
        st.write(f"Top {len(matched_keywords)} matched keywords:")
        for kw, score in results:
            st.write(f"{kw} — similarity: {score:.3f}")
        
        selected_keywords = st.multiselect("Select keywords for analysis", matched_keywords)
        
        if selected_keywords:
            st.write("### Keyword-based analyses")
            
            # Section 3: Keyword → Years
            chart_df = pd.DataFrame()
            for kw in selected_keywords:
                data = keywords_summary.get(kw, {}).get('years', {})
                df = pd.DataFrame({"Year": list(data.keys()), kw: list(data.values())})
                df = df.sort_values('Year')
                df.set_index("Year", inplace=True)
                chart_df = pd.concat([chart_df, df], axis=1)
            if not chart_df.empty:
                st.subheader("Trend Over Time (Keyword → Years)")
                st.line_chart(chart_df, use_container_width=True)
            
            # Section 4: Keyword → Countries
            n = st.slider("Top N Countries for Keyword Research", 5, 30, 15, key="ss_n")
            chart_df2 = pd.DataFrame()
            for kw in selected_keywords:
                data = keywords_summary.get(kw, {}).get('countries', {})
                data = top_n_items(data, n)
                df = pd.DataFrame({"Country": list(data.keys()), kw: list(data.values())})
                df.set_index("Country", inplace=True)
                chart_df2 = pd.concat([chart_df2, df], axis=1)
            if not chart_df2.empty:
                st.subheader("Countries Researching Keyword(s)")
                fig2 = px.bar(chart_df2, title="Keyword → Countries")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Section 9: Keyword + Country → Years
            all_countries = sorted(list(countries_summary.keys()))
            selected_countries = st.multiselect("Select Countries for Yearly Keyword Activity", all_countries, default=all_countries[:1], key="ss_countries")
            chart_df3 = pd.DataFrame()
            for keyword in selected_keywords:
                for country in selected_countries:
                    trend = {}
                    for year, yearinfo in years_summary.items():
                        count = yearinfo.get('keyword_countries', {}).get(keyword, {}).get(country, 0)
                        if count > 0:
                            trend[year] = count
                    if trend:
                        df = pd.DataFrame({"Year": list(trend.keys()), f"{keyword} ({country})": list(trend.values())})
                        df = df.sort_values('Year')
                        df.set_index("Year", inplace=True)
                        chart_df3 = pd.concat([chart_df3, df], axis=1)
            if not chart_df3.empty:
                st.subheader("Keyword + Country → Years Trend")
                st.line_chart(chart_df3, use_container_width=True)


# --- Footer ---
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Made with ❤️ using Streamlit | Shivang Agarwal")
