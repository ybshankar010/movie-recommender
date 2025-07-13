import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.logs.logger import SimpleLogger
from app.utils.constants import DATA_FOLDER

logger = SimpleLogger(__name__, level="debug")

# Load the dataset
movies_df = pd.read_csv(DATA_FOLDER)

movies_df = movies_df[movies_df['genres'] != '(no genres listed)']

merged_movies_df = (
    movies_df.groupby('title', as_index=False)
    .agg({
        'movieId': 'first',
        'genres': lambda x: '|'.join(sorted(set('|'.join(x).split('|'))))
    })
)

merged_movies_df['combined_text'] = merged_movies_df['title'] + " " + merged_movies_df['genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(merged_movies_df['combined_text'])

merged_movies_df = merged_movies_df.reset_index(drop=True)
indices = pd.Series(merged_movies_df.index, index=merged_movies_df['title']).drop_duplicates()


def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        logger.error("Movie not found in dataset.")
        return pd.DataFrame()

    idx = indices[title]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # skip itself

    movie_indices = [i[0] for i in sim_scores]
    return merged_movies_df.iloc[movie_indices][['title', 'genres']]

def main():
    title = "Toy Story (1995)"
    num_recommendations = 5
    recommendations = recommend_movies(title, num_recommendations=num_recommendations)
    if recommendations.empty:
        print("No recommendations found.")
    else:
        print("\nTop recommendations:")
        print(recommendations)

if __name__ == "__main__":
    main()