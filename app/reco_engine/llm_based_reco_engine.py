from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate

from app.logs.logger import SimpleLogger
from app.db.chromadb import ChromaDB
from app.utils.common_utils import setup_llm


class LLMBasedRecoEngine:
    def __init__(self, temperature=0.7):
        self.model_name = "llama3.1:8b"
        self.temperature = temperature
        self.logger = SimpleLogger(self.__class__.__name__, level="debug")
        self.logger.info(f"LLM Based Recommender initialized with model: {self.model_name}")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.db = ChromaDB()
        self.filter_template = """
You are a movie expert. Here is a list of movie titles:

{movies_list}

User preference: "{user_preference}"

From this list, please select {final_k} movies that best match the user's preference.
Return as a bullet-point list of movie titles only, no explanations.
"""
        self.prompt = PromptTemplate(
            input_variables=["movies_list", "user_preference", "final_k"],
            template=self.filter_template
        )
        self.llm = setup_llm(model_name=self.model_name, temperature=self.temperature)

        self.refinement_chain = (
            self.prompt
            | self.llm
        )

    
    def retrieve_movies_chroma(self,user_text, top_k=50):
        # Embed user preference
        # user_embedding = self.embedding_model.encode([user_text], normalize_embeddings=True).tolist()[0]
        
        # Query Chroma
        results = self.db.query_data(
            query=user_text,
            n_results=top_k
        )
        
        # Extract titles
        titles = [m['title'] for m in results['metadatas'][0]]
        if not titles:
            self.logger.warning("No movies found for the given user text.")
            return []
        
        self.logger.info(f"Top {len(titles)} movie titles retrieved successfully.")
        return titles
    
    def refine_recommendations(self, user_text, final_k=5):
        """
        Refine recommendations based on user input.
        This method can be extended to include more complex logic.
        """
        self.logger.debug(f"Refining recommendations for user text: {user_text}")
        movies_list = self.retrieve_movies_chroma(user_text, top_k=50)
        movies_text = "\n".join(movies_list)    
    
        prompt = f"""
        You are a movie expert. Here is a list of movie titles:
        {movies_text}
        
        User preference: "{user_text}"
        
        From this list, please select {final_k} movies that best match the user's preference.
        Return as a bullet-point list of movie titles only, no explanation.
        """
        response = self.refinement_chain.invoke({
            "movies_list": movies_text,
            "user_preference": user_text,
            "final_k": final_k
        })

        filtered_titles = [line.strip("•- ").strip() for line in response.content.strip().split("\n") if line.strip()]
        return filtered_titles
    

def main():
    # Create an instance of your recommender
    reco_engine = LLMBasedRecoEngine(temperature=0.7)
    
    # User preference text example
    user_pref = "I like thought-provoking sci-fi movies with strong emotional depth and surprising twists."

    # Get refined recommendations
    recommendations = reco_engine.refine_recommendations(user_pref, final_k=5)

    print("Final Recommendations:")
    for i, title in enumerate(recommendations, 1):
        print(f"{i}. {title}")

if __name__ == "__main__":
    main()