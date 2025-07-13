import pandas as pd
from app.reco_engine.tf_idf_recommender import recommend_movies
from app.reco_engine.llm_based_reco_engine import LLMBasedRecoEngine

from app.logs.logger import SimpleLogger
from app.utils.constants import DATA_FOLDER
from app.etl.data import prepare_movies_data

class HybridRecoEngine:

    def __init__(self, content_k=20, llm_final_k=10, hybrid_final_k=10):
        self.content_k = content_k
        self.llm_final_k = llm_final_k
        self.hybrid_final_k = hybrid_final_k
        self.logger = SimpleLogger(self.__class__.__name__, level="debug")
        self.llm_reco_engine = LLMBasedRecoEngine()
        self.logger.info("Hybrid Recommender initialized.")


    def hybrid_recommendations(self, user_text, content_title):
        """
        Combine TF-IDF content-based and LLM-based recommendations.
        """
        movies_df = prepare_movies_data()
        if movies_df.empty:
            self.logger.warning("No movies data available for recommendations.")
            return pd.DataFrame()
        
        # Step 1: Content-based recommendations
        content_df = recommend_movies(content_title, num_recommendations=self.content_k)
        content_titles = content_df['title'].tolist() if not content_df.empty else []
        self.logger.debug(f"Content-based titles: {content_titles}")

        # Step 2: LLM-based refined titles
        llm_titles = self.llm_reco_engine.refine_recommendations(user_text, final_k=self.llm_final_k)
        self.logger.debug(f"LLM-based titles: {llm_titles}")

        # Step 3: Combine and deduplicate
        combined_titles = list(set(content_titles + llm_titles))
        self.logger.debug(f"Combined titles: {combined_titles}")

        # Step 4: Create final DataFrame
        final_df = movies_df[movies_df['title'].isin(combined_titles)][['title', 'genres']]

        # Limit final set if needed
        final_df = final_df.head(self.hybrid_final_k)
        return final_df

def main():
    # Create an instance of the hybrid recommender
    hybrid_reco = HybridRecoEngine(content_k=20, llm_final_k=10, hybrid_final_k=10)

    # Example user preference text
    user_pref = "I love psychological thrillers with strong female leads and surprising plot twists. Avoid heavy violence."

    # Example content-based movie title
    content_title = "Gone Girl (2014)"

    # Get hybrid recommendations
    final_df = hybrid_reco.hybrid_recommendations(
        user_text=user_pref,
        content_title=content_title
    )

    print("✨ Final Hybrid Recommendations:")
    print(final_df)

if __name__ == "__main__":
    main()

