import streamlit as st
import pandas as pd
from app.reco_engine.tf_idf_recommender import recommend_movies
from app.reco_engine.hybrid_reco_engine import HybridRecoEngine
from app.etl.data import prepare_movies_data

# --------------------------
# Setup
# --------------------------
st.title("🎥 Movie Recommender System")

st.write("Choose a movie and optionally describe your preferences to get hybrid recommendations.")

merged_movies_df = prepare_movies_data()
if merged_movies_df.empty:
    st.error("No movies data available. Please check the dataset.")
    st.stop()

movie_titles = merged_movies_df['title'].sort_values().tolist()

# Select mode
mode = st.radio("Select recommendation mode:", ("Content-based (TF-IDF)", "Hybrid (TF-IDF + LLM)"))

# Common movie selector
selected_title = st.selectbox("Choose a base movie", movie_titles)

# User preference text (optional)
user_pref_text = ""
if mode == "Hybrid (TF-IDF + LLM)":
    user_pref_text = st.text_area("Describe your preferences (optional, for LLM filtering)", "")

num_recommendations = st.slider("Number of recommendations", 1, 20, 5)

# --------------------------
# Recommendation logic
# --------------------------
if st.button("Show Recommendations"):
    if mode == "Content-based (TF-IDF)":
        recommendations = recommend_movies(selected_title, num_recommendations=num_recommendations)
        if not recommendations.empty:
            st.write("## Top Content-Based Recommendations")
            st.dataframe(recommendations.reset_index(drop=True))
        else:
            st.write("No recommendations found.")
    else:
        # Hybrid mode
        hybrid_reco = HybridRecoEngine(content_k=20, llm_final_k=15, hybrid_final_k=num_recommendations)
        if not user_pref_text.strip():
            st.warning("Please describe your preferences for LLM filtering to enable hybrid recommendations.")
        else:
            hybrid_df = hybrid_reco.hybrid_recommendations(
                user_text=user_pref_text,
                content_title=selected_title
            )
            if not hybrid_df.empty:
                st.write("## Top Hybrid Recommendations")
                st.dataframe(hybrid_df.reset_index(drop=True))
            else:
                st.write("No recommendations found.")
