import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Shopping Assistant", layout="wide")

# ================== CUSTOM CSS ==================
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .product-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: scale(1.02);
    }
    .product-title {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .price {
        font-size: 16px;
        font-weight: bold;
        color: #00ffcc;
        margin-bottom: 8px;
    }
    .desc {
        font-size: 14px;
        color: #cccccc;
    }
    </style>
""", unsafe_allow_html=True)

# ================== LOAD DATA ==================
df = pd.read_csv("products.csv")

# Combine text for AI
df["combined"] = df["name"].astype(str) + " " + df["description"].astype(str)

# TF-IDF model
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# ================== HEADER ==================
st.markdown("""
# 🛒 AI Shopping Assistant  
### Find the best products instantly with AI 🚀
""")

# ================== DATA PREVIEW ==================
df_display = df.copy()
df_display["name"] = df_display["name"].astype(str).str[:40] + "..."
df_display["description"] = df_display["description"].astype(str).str[:60] + "..."

st.markdown("## 📦 Product Data Preview")
st.dataframe(df_display.head())

# ================== USER INPUT ==================
user_input = st.text_input("💬 Enter your query:")

# ================== MAIN LOGIC ==================
if user_input:
    query = user_input.lower()

    # Extract price
    price_match = re.findall(r'\d+', query)
    max_price = int(price_match[0]) if price_match else None

    # Bot response
    st.markdown("---")
    if max_price:
        st.write(f"🤖 Here are some smart recommendations under ₹{max_price} for you:")
    else:
        st.write("🤖 Here are some relevant products for you:")

    # Convert query to vector
    query_vec = vectorizer.transform([query])

    # Compute similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix)

    # Get top matches
    top_indices = similarity[0].argsort()[-10:][::-1]

    # Get results
    result = df.iloc[top_indices]

    # Apply price filter
    if max_price:
        result = result[result["price"] <= max_price]

    # ================== DISPLAY ==================
    if not result.empty:
        result = result.head(6)

        st.markdown("## 🔥 Top Picks For You")

        cols = st.columns(2)

        for idx, row in enumerate(result.itertuples()):
            col = cols[idx % 2]

            with col:
                st.markdown(f"""
                <div class="product-card">
                    <div class="product-title">🛍️ {row.name}</div>
                    <div class="price">💰 ₹{row.price}</div>
                    <div class="desc">{str(row.description)[:140]}...</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.write("😔 No matching products found. Try a different query!")