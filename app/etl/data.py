import pandas as pd

from app.logs.logger import SimpleLogger
from app.utils.constants import DATA_FOLDER


logger = SimpleLogger("Prepare_Data", level="debug")

def prepare_movies_data():
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

    return merged_movies_df