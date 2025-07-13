import streamlit as st
import pandas as pd
from app.reco_engine.tf_idf_recommender import merged_movies_df, recommend_movies

st.title("🎥 Movie Recommender (TF-IDF Based)")

st.write("Select a movie to get similar recommendations based on title and genres.")

# Sort titles to show nicely in select box
movie_titles = merged_movies_df['title'].sort_values().tolist()

selected_title = st.selectbox("Choose a movie", movie_titles)

num_recommendations = st.slider("Number of recommendations", 1, 20, 5)

if st.button("Show Recommendations"):
    recommendations = recommend_movies(selected_title, num_recommendations=num_recommendations)
    
    if not recommendations.empty:
        st.write("## Top Recommendations")
        st.dataframe(recommendations.reset_index(drop=True))
    else:
        st.write("No recommendations found for the selected movie.")
