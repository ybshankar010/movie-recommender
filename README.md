# Movie Recommendation System

A hybrid movie recommendation system that combines traditional content-based filtering with GenAI-powered recommendations using a local LLM on consumer hardware.

## 🎬 Overview

This project implements a sophisticated movie recommendation system that leverages both traditional content-based filtering techniques and modern GenAI capabilities. The system analyzes movie features (genres, titles, metadata) to provide personalized recommendations with natural language explanations.

### Key Features

- **Hybrid Approach**: Combines content-based filtering with GenAI recommendations
- **Local LLM**: Runs entirely on consumer hardware (8GB GPU)
- **Cold Start Solution**: Handles new users effectively using GenAI
- **Natural Language Explanations**: Provides reasoning for recommendations
- **Interactive UI**: Streamlit-based web interface
- **Real-time Recommendations**: Fast inference with optimized models

## 🏗️ Architecture

### System Components

1. **Traditional Content-Based Filtering**

   - TF-IDF vectorization of movie features
   - Cosine similarity between movies
   - Feature engineering from genres, titles, and metadata

2. **GenAI-Enhanced Recommender**

   - Local quantized LLM for contextual understanding
   - Embedding-based similarity search using movie content
   - Natural language user profile generation from rating patterns

3. **Hybrid Fusion System**
   - Weighted combination of content-based and GenAI approaches
   - Dynamic weight adjustment based on user data availability
   - Intelligent fallback mechanisms for new users

## 📊 Dataset

**Source**: [Kaggle Movie Recommendation System Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)

**Files**:

- `movies.csv`: Movie metadata (movieId, title, genres)
- `ratings.csv`: User ratings (userId, movieId, rating, timestamp)

## 🔧 Technical Requirements

### Hardware

- **GPU**: 8GB VRAM (RTX 3070/4060 Ti or better)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and data

### Software

- **Python**: 3.9+
- **CUDA**: Compatible GPU drivers
- **Package Manager**: uv (for fast dependency management)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
uv sync
```

### 2. Download Dataset

```bash
# Configure Kaggle credentials first
# Place kaggle.json in ~/.kaggle/ or set environment variables

# Download dataset
uv run python scripts/download_data.py
```

### 3. Run the Application

```bash
# Start the Streamlit app
uv run streamlit run app/streamlit_app.py
```

## 💡 Key Implementation Details

### Content-Based Filtering with TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Combine movie features into single text
def create_movie_features(movies_df):
    movies_df['combined_features'] = (
        movies_df['genres'].fillna('') + ' ' +
        movies_df['title'].fillna('') + ' ' +
        movies_df.get('keywords', '').fillna('')
    )
    return movies_df

# Create TF-IDF vectors
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2)
)

# Fit and transform movie features
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get recommendations
def get_content_recommendations(movie_title, cosine_sim, movies_df, n_recommendations=10):
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
    return movies_df.iloc[movie_indices]['title'].tolist()
```

### GenAI Model Configuration

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### Hybrid Scoring

```python
def hybrid_score(content_score, genai_score, user_data_availability, alpha=0.7):
    if user_data_availability < 10:  # Cold start - rely more on content
        return 0.3 * content_score + 0.7 * genai_score
    else:
        return alpha * content_score + (1 - alpha) * genai_score
```

## 📈 Performance Metrics

- **Response Time**: < 2 seconds for recommendations
- **GPU Memory**: < 7GB usage
- **Accuracy**: Precision@10, Recall@10, NDCG@10
- **Diversity**: Intra-list diversity and novelty metrics

## 🧪 Evaluation

**Offline Evaluation**: Historical data split and cross-validation

## 🔧 Configuration

### Environment Variables

```bash
# .env file
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
MODEL_PATH=./models/genai/
CHROMA_DB_PATH=./data/chroma_db/
```

## 🚀 Usage Examples

### Basic Recommendations

```python
from src.hybrid_system import HybridRecommender

# Initialize recommender
recommender = HybridRecommender()

# Get recommendations for a user
user_id = 123
recommendations = recommender.get_recommendations(
    user_id=user_id,
    n_recommendations=10
)

# Print recommendations with explanations
for movie, score, explanation in recommendations:
    print(f"{movie}: {score:.2f} - {explanation}")
```

### Cold Start (New User)

```python
# For new users without rating history
preferences = {
    'genres': ['Action', 'Sci-Fi'],
    'keywords': ['superhero', 'space'],
    'min_rating': 7.0
}

recommendations = recommender.get_cold_start_recommendations(
    preferences=preferences,
    n_recommendations=10
)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is designed for educational and research purposes. For production use, consider scalability, security, and performance optimizations.
