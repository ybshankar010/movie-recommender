import pandas as pd

from app.logs.logger import SimpleLogger
from app.db.chromadb import ChromaDB
from app.etl.data import prepare_movies_data

class DataIngestor:
    def __init__(self):
        self.logger = SimpleLogger(DataIngestor.__name__, level="debug")
        self.logger.info("DataIngestor getting initialized.")
        self.db = ChromaDB()
        self.db.init_database()
        self.logger.info("DataIngestor initialized with ChromaDB instance.")
        
    def ingest_data(self):
        self.logger.info("Starting data ingestion process.")
        try:
            # Load the dataset
            movies_df = prepare_movies_data()
            
            if movies_df.empty:
                self.logger.warning("No data found to ingest.")
                return
            
            self.logger.info(f"Loaded {len(movies_df)} movies from dataset.")
            for _, row in movies_df.iterrows():
                content = row['title'] + " " + row['genres']
                metadata = {
                    'movieId': row['movieId'],
                    'title': row['title'],
                    'genres': row['genres']
                }
                self.db.insert_data(content=content, metadata=metadata)
                self.logger.debug(f"Inserted movie: {row['title']} with ID: {row['movieId']}")
            self.logger.info("Data ingestion completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during data ingestion: {e}")
        
        return
    
def main():
    print("Starting data ingestion...")
    ingestor = DataIngestor()
    ingestor.ingest_data()
    ids,documents,metadatas,embeddings = ingestor.db.get_all_data()
    print(f"Total documents in the database: {len(documents)}")
    for id,document,_,_ in zip(ids,documents,metadatas,embeddings):
        print("==========",id, "==========",document[:10])

if __name__ == "__main__":
    main()