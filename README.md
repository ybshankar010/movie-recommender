# 🚀 Movie Recommendation System

A hybrid movie recommendation system that combines traditional content-based filtering with local LLM-enhanced GenAI recommendations using consumer hardware.

---

## 🎬 Overview

This system leverages both **classic content-based filtering** (TF-IDF on titles and genres) and **GenAI capabilities** (local LLM reasoning and vector search) to provide refined and personalized movie recommendations.

---

### ⚡ Key Features

- **Hybrid Approach**: Combines TF-IDF content similarity with vector-based GenAI refinement.
- **Local LLM Integration**: Uses a quantized local language model via LangChain for filtering and nuanced reasoning.
- **Supports Cold Start**: Allows users to input free-text preferences without prior ratings.
- **Interactive UI**: Intuitive Streamlit interface to explore recommendations.
- **Database-Restricted Results**: Final movies always come from your stored database.

---

## 🏗️ Architecture

### Components

#### ✅ Content-Based Filtering

- TF-IDF vectorization on titles and genres.
- Cosine similarity to find movies similar to a given anchor title.

#### ✅ GenAI-Enhanced Recommender

- Local language model (accessed via LangChain) to interpret user free-text preferences.
- Vector similarity search on pre-computed embeddings (ChromaDB).
- LLM-based filtering and re-ranking of vector search results.

#### ✅ Hybrid Layer

- Combines TF-IDF content recommendations with LLM-refined results.
- Deduplicates and merges, prioritizing nuanced user preferences.

---

## 📊 Dataset

**Source**: [Kaggle Movie Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)

**Key Files**:

- `movies.csv`: Movie metadata (movieId, title, genres).
- `ratings.csv`: User ratings (userId, movieId, rating, timestamp).

---

## 💻 Technical Requirements

### Hardware

- **GPU**: 8GB VRAM recommended (e.g., RTX 3070, 4060 Ti, or Apple M-series with Metal support).
- **RAM**: 16GB+ suggested.
- **Disk**: 5GB+ for model files and embeddings.

### Software

- **Python**: 3.9+
- **CUDA/Metal**: Optional, for local LLM acceleration.

---

## 🚀 Quick Start

### 1️⃣ Setup

```bash
# Clone repo
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

---

### 2️⃣ Download and prepare data

```bash
# Configure Kaggle credentials first if needed

# Example placeholder
python scripts/download_data.py
```

---

### 3️⃣ Run Streamlit App

```bash
streamlit run recommender.py
```

---

## 💬 Usage

### Content-Based Example

```python
from tf_idf_recommender import recommend_movies

recs = recommend_movies("Inception (2010)", num_recommendations=5)
print(recs)
```

---

### LLM-Enhanced Example

```python
from llm_based_reco_engine import LLMBasedRecoEngine

reco_engine = LLMBasedRecoEngine()
user_text = "I enjoy sci-fi thrillers with deep psychological twists and minimal violence."
recs = reco_engine.refine_recommendations(user_text, final_k=5)
print(recs)
```

---

### Hybrid Example

```python
from hybrid_reco_engine import HybridRecoEngine

hybrid_engine = HybridRecoEngine()
user_text = "I like sci-fi thrillers with strong female leads and philosophical themes."
anchor_movie = "Arrival (2016)"

final_recs = hybrid_engine.hybrid_recommendations(
    user_text=user_text,
    content_title=anchor_movie,
    content_k=20,
    llm_final_k=10,
    hybrid_final_k=10
)
print(final_recs)
```

---

## ⚙️ Environment Config

```bash
# .env (example)
CHROMA_DB_PATH=./data/chroma_db/
MODEL_PATH=./models/local_llm/
```

---

## 📈 Metrics & Evaluation

- **Precision\@5, Recall\@5**: Evaluate using held-out user likes.
- **Cold Start Success**: Ability to recommend using only text preferences.
- **Interactive User Feedback**: Check relevance from real-time testing in UI.

---

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

**Note**: This project is designed for educational and research purposes. For production use, co
