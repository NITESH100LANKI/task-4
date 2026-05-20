import sys
import httpx
import asyncio
import re
import time
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional

from configs.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, TMDB_API_KEY, OMDB_API_KEY
from src.models.recommenders import ContentBasedRecommender

app = FastAPI(
    title="CineMind AI - Movie API",
    description="Lightweight Content-Based Movie Recommendation Engine",
    version="1.0.0"
)

# Global variables
movies_df = None
cb_model = None
metadata_cache = {}

@app.on_event("startup")
def load_models():
    global movies_df, cb_model
    try:
        movies_path = PROCESSED_DATA_DIR / "movies.parquet"
        if movies_path.exists():
            movies_df = pd.read_parquet(movies_path)
            print(f"Loaded {len(movies_df)} movies for API metadata lookup.")
        else:
            print("Cleaned movies parquet file not found.")
            
        try:
            cb_model = ContentBasedRecommender.load()
            print("Content-based TF-IDF model loaded successfully.")
        except Exception as e:
            print(f"Content-based TF-IDF model not found or failed to load: {e}")
    except Exception as e:
        print(f"Error loading models on startup: {e}")

class RecommendationRequest(BaseModel):
    query: str
    top_n: int = 10

async def fetch_omdb_poster(title: str):
    """Fetch poster dynamically from OMDb API if key is available."""
    if not OMDB_API_KEY or OMDB_API_KEY == "your_actual_omdb_key_here":
        return None
        
    url = "http://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "t": title}
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(url, params=params, timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                if data.get("Response") == "True" and data.get("Poster") != "N/A":
                    return data.get("Poster")
        except:
            pass
    return None

async def fetch_tmdb_poster(title: str):
    """Fetch poster dynamically from TMDB API if key is available."""
    if not TMDB_API_KEY or TMDB_API_KEY == "your_actual_tmdb_key_here":
        return None
    
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(url, params=params, timeout=3.0)
            if res.status_code == 200:
                results = res.json().get('results', [])
                if results and results[0].get('poster_path'):
                    return f"https://image.tmdb.org/t/p/w500{results[0].get('poster_path')}"
        except:
            pass
    return None

async def get_poster(title: str):
    if title in metadata_cache:
        return metadata_cache[title]
        
    poster = await fetch_tmdb_poster(title)
    if not poster:
        poster = await fetch_omdb_poster(title)
        
    metadata_cache[title] = poster
    return poster

async def enrich_recommendations(recs: List[dict]):
    """Fetch posters in parallel to keep API response times very fast."""
    tasks = [get_poster(r['title']) for r in recs]
    posters = await asyncio.gather(*tasks)
    
    for r, poster in zip(recs, posters):
        r['poster_path'] = poster
    return recs

@app.get("/")
def health_check():
    return {"status": "ok", "message": "CineMind AI Movie API is running."}

@app.post("/recommend")
async def get_recommendations(req: RecommendationRequest):
    if not cb_model:
        raise HTTPException(status_code=503, detail="Recommendation model not loaded")
        
    try:
        recs = cb_model.recommend(req.query, req.top_n)
        if not recs:
            return {"recommendations": []}
            
        recs = await enrich_recommendations(recs)
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies")
def get_movies_list():
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    titles = sorted(movies_df['title'].dropna().unique().tolist())
    return {"movies": titles}

@app.get("/search")
async def search_movies(query: str, limit: int = 10):
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    mask = movies_df['title'].str.contains(query, case=False, na=False)
    results_df = movies_df[mask].head(limit)
    
    results = []
    for _, row in results_df.iterrows():
        results.append({
            "title": row.get("title", ""),
            "genres": row.get("genres_cleaned", ""),
            "overview": row.get("overview", "No synopsis available."),
            "vote_average": str(row.get("vote_average", "N/A")),
            "runtime": f"{int(row['runtime'])} min" if pd.notna(row.get('runtime')) else "N/A",
            "actors": row.get("cast_cleaned", ""),
            "box_office": f"${int(row['revenue']):,}" if pd.notna(row.get('revenue')) and row['revenue'] > 0 else "N/A",
            "explanation": "Search Result",
            "confidence": 100.0,
            "latency_ms": 0.0
        })
        
    results = await enrich_recommendations(results)
    return {"results": results}
