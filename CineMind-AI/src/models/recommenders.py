import pandas as pd
import numpy as np
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from configs.settings import MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
        self.tfidf_matrix = None
        self.movies_df = None
        self.titles_cleaned = []

    def _clean_title(self, title: str) -> str:
        if not isinstance(title, str):
            return ""
        import re
        title = title.lower()
        title = re.sub(r"[^a-z0-9]", "", title)
        return title.strip()

    def fit(self, movies_df: pd.DataFrame):
        logger.info("Fitting TF-IDF Content-Based Recommender...")
        t0 = time.time()
        self.movies_df = movies_df.reset_index(drop=True)
        
        self.titles_cleaned = [self._clean_title(t) for t in self.movies_df['title']]
        
        features = (
            self.movies_df['overview'].fillna("") + " " +
            self.movies_df['tagline'].fillna("") + " " +
            self.movies_df['genres_cleaned'].fillna("") + " " +
            self.movies_df['keywords_cleaned'].fillna("") + " " +
            self.movies_df['cast_cleaned'].fillna("") + " " +
            self.movies_df['director_cleaned'].fillna("")
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(features)
        logger.info(f"TF-IDF Matrix built in {time.time() - t0:.2f}s with shape {self.tfidf_matrix.shape}")

    def recommend(self, query: str, top_n: int = 10):
        t0 = time.time()
        if not query or self.tfidf_matrix is None:
            return []
            
        clean_query = self._clean_title(query)
        
        movie_idx = None
        if clean_query in self.titles_cleaned:
            movie_idx = self.titles_cleaned.index(clean_query)
            
        if movie_idx is not None:
            logger.info(f"Recommending movies similar to: {self.movies_df.iloc[movie_idx]['title']}")
            query_vector = self.tfidf_matrix[movie_idx]
            similarity = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            indices = np.argsort(similarity)[::-1]
            indices = [idx for idx in indices if idx != movie_idx][:top_n]
            explanation = "Similar Movie Recommendation (TF-IDF)"
        else:
            logger.info(f"Recommending movies for search query: '{query}'")
            query_vector = self.vectorizer.transform([query])
            similarity = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            indices = np.argsort(similarity)[::-1][:top_n]
            explanation = "Search Phrase Match (TF-IDF)"
            
        latency_ms = (time.time() - t0) * 1000
        
        recs = []
        for idx in indices:
            row = self.movies_df.iloc[idx]
            sim_score = similarity[idx]
            
            actors = row.get('cast_cleaned', '')
            if isinstance(actors, str) and len(actors) > 100:
                actors = ", ".join(actors.split(", ")[:3])
                
            recs.append({
                "title": row.get("title", ""),
                "genres": row.get("genres_cleaned", ""),
                "overview": row.get("overview", "No synopsis available."),
                "vote_average": str(row.get("vote_average", "N/A")),
                "runtime": f"{int(row['runtime'])} min" if pd.notna(row.get('runtime')) else "N/A",
                "actors": actors,
                "box_office": f"${int(row['revenue']):,}" if pd.notna(row.get('revenue')) and row['revenue'] > 0 else "N/A",
                "explanation": explanation,
                "confidence": round(float(sim_score) * 100, 2),
                "latency_ms": round(latency_ms, 2)
            })
            
        return recs

    def save(self):
        save_path = MODEL_DIR / "content_recommender.pkl"
        joblib.dump({
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "movies_df": self.movies_df,
            "titles_cleaned": self.titles_cleaned
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls):
        save_path = MODEL_DIR / "content_recommender.pkl"
        instance = cls()
        data = joblib.load(save_path)
        instance.vectorizer = data["vectorizer"]
        instance.tfidf_matrix = data["tfidf_matrix"]
        instance.movies_df = data["movies_df"]
        instance.titles_cleaned = data["titles_cleaned"]
        logger.info(f"Model loaded successfully from {save_path}")
        return instance
